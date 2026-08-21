#!/usr/bin/env python3
"""Splice a bench run against the frozen baseline cache (never re-run).

  python3 tools/bench_compare.py <NAME>            # compare + write csv
  python3 tools/bench_compare.py <NAME> --promote  # ...and cache NAME's
                                                   # rows as a baseline

Inputs:  cache/nonuniform/bench_<NAME>.log   (from script/bench_scheme.sh)
         experiments/mosaic_x1/bench_baselines.csv
Outputs: experiments/mosaic_x1/bench_runs/<NAME>.csv
         stdout comparison table (vs sota + vs best-other per cell)

Baselines are FROZEN: they are written once (archives scrape + one
canonical SOTA run) and only ever extended via --promote. The gate rule
(CLAUDE.md): SOTA claims need per-cell >= sota - 5% and zero FAILED;
breaches must be argued as an aggregate trade, 41-group style.
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'experiments/mosaic_x1/bench_baselines.csv')
RUNS = os.path.join(ROOT, 'experiments/mosaic_x1/bench_runs')


def parse_log(name):
    path = os.path.join(ROOT, 'cache/nonuniform/bench_%s.log' % name)
    rows = {}
    for line in open(path):
        m = re.match(r'CELL (\S+) validation (PASSED|FAILED)'
                     r'.*?completion_us ([0-9.]+)', line)
        if m:
            drops = re.search(r'queue_drops (\d+)', line)
            rows[m.group(1)] = (float(m.group(3)), m.group(2),
                                int(drops.group(1)) if drops else -1)
    return rows


def load_baselines():
    base = {}
    if not os.path.exists(BASE):
        return base
    for r in csv.DictReader(open(BASE)):
        base.setdefault(r['cell'], {})[r['scheme']] = (
            float(r['completion_us']) if r['completion_us'] else None,
            r.get('source', ''))
    return base


def main():
    name = sys.argv[1]
    promote = '--promote' in sys.argv
    rows = parse_log(name)
    base = load_baselines()
    os.makedirs(RUNS, exist_ok=True)
    out = csv.writer(open(os.path.join(RUNS, name + '.csv'), 'w'))
    out.writerow(['cell', 'scheme', 'completion_us', 'status', 'drops',
                  'bound_us', 'vs_sota', 'vs_best_other', 'best_other'])
    print('%-18s %10s %7s | %10s %8s | %10s %8s %s' % (
        'cell', name, 'status', 'sota(U2)', 'x_sota', 'best_other',
        'x_other', 'gate'))
    worst, fails, wins, losses = None, 0, 0, 0
    for cell in sorted(rows):
        comp, st, drops = rows[cell]
        b = base.get(cell, {})
        bound = b.get('bound', (None, ''))[0]
        sota = b.get('sota_u2', (None, ''))[0]
        others = {k: v[0] for k, v in b.items()
                  if k not in ('bound', 'sota_u2') and v[0]}
        bestk = min(others, key=lambda k: others[k]) if others else ''
        bestv = others.get(bestk)
        vs = (sota / comp) if (sota and comp) else None
        vo = (bestv / comp) if (bestv and comp) else None
        gate = ''
        if st == 'FAILED':
            gate, fails = 'FAIL', fails + 1
        elif vs is not None:
            if vs < 0.95:
                gate = 'BREACH'
            wins += vs >= 1.005
            losses += vs <= 0.995
            if worst is None or vs < worst[1]:
                worst = (cell, vs)
        out.writerow([cell, name, comp, st, drops, bound or '',
                      '%.3f' % vs if vs else '',
                      '%.3f' % vo if vo else '', bestk])
        print('%-18s %10.1f %7s | %10s %8s | %10s %8s %s' % (
            cell, comp, st,
            '%.1f' % sota if sota else '-',
            '%.3f' % vs if vs else '-',
            '%.1f' % bestv if bestv else '-',
            '%.3f' % vo if vo else '-', gate))
    print('\nvs SOTA: wins=%d losses=%d worst=%s fails=%d'
          % (wins, losses, worst, fails))
    print('gate: %s' % ('PASS (all cells >= sota-5%, no FAILED)'
                        if fails == 0 and (worst is None
                                           or worst[1] >= 0.95)
                        else 'NOT PASSED as-is -- argue the aggregate '
                             'trade (41-group precedent) or fix'))
    if promote:
        exists = os.path.exists(BASE)
        with open(BASE, 'a') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['cell', 'scheme', 'completion_us', 'source'])
            for cell, (comp, st, drops) in rows.items():
                if st == 'PASSED':
                    w.writerow([cell, name, comp, 'bench_%s.log' % name])
        print('promoted %d rows of %s into baselines cache' %
              (len(rows), name))


if __name__ == '__main__':
    main()
