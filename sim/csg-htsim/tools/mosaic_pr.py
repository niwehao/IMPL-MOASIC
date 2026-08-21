#!/usr/bin/env python3
# MOSAIC-O-PR: candidate-family + exact-async-evaluation selector for the
# PARTIAL (per-port) reconfiguration semantics of htsim_optical.
#
# Scheme definition (the OCS twin of the EPS "compose sub-schedules and
# tune their shares" idea): the executed schedule is a time-weighted
# composition of circuit configurations; under per-port billing a port
# pays R only when ITS circuit changes and never waits for a barrier.
# The planner fields three candidate families
#   rotation  Cay(Z_n,{1,n-1}) (+ optional extra generators): each pair
#             rides its shorter direction; the hot column's two natural
#             ingress edges make the gateway skeleton embedded,
#   ron-style the RoN contraction sequence for every d (1..n-1),
#   static    the theta-opt single topology (Harvest plan file, if given),
# evaluates EVERY candidate by exact simulation under the async executor
# (same license as MOSAIC-O's scratch-fabric evaluation: known state
# only), and executes the argmin. Wall-clock cost = |candidates| fast
# chunk-level sims (~seconds at n=32).
#
# Usage:
#   mosaic_pr.py --demand-file F --reconf-us R [--static-plan HV.txt]
#                [--bin ./sim/tests/htsim_optical] [--keep-plan OUT]
import argparse, os, re, subprocess, sys, tempfile

def run_async(bin_, plan):
    out = subprocess.run([bin_, '-plan', plan, '-async'],
                         capture_output=True, text=True).stdout
    m = re.search(r'optical (PASSED|FAILED) async.*port_reconfigs (\d+) '
                  r'completion_us ([0-9.e+]+)', out)
    if not m or m.group(1) != 'PASSED':
        return None
    return float(m.group(3)), int(m.group(2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demand-file', required=True)
    ap.add_argument('--reconf-us', type=float, required=True)
    ap.add_argument('--nodes', type=int, default=32)
    ap.add_argument('--bytes-per-pair', type=int, default=2097152)
    ap.add_argument('--static-plan', default='',
                    help='optional theta-opt plan file (offline artifact)')
    ap.add_argument('--bin', default='./sim/tests/htsim_optical')
    ap.add_argument('--keep-plan', default='',
                    help='copy the winning plan here')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = tempfile.mkdtemp(prefix='mosaic_pr_',
                           dir=os.path.join(os.path.dirname(here),
                                            'cache', 'optical'))
    cands = []                      # (name, plan_path)

    def gen(tool, args, out):
        subprocess.run([sys.executable, os.path.join(here, tool)] + args
                       + ['--output', out],
                       capture_output=True, text=True)
        return out

    for g in ('1,%d' % (a.nodes - 1), '1,5,%d' % (a.nodes - 1),
              '1,7,25,%d' % (a.nodes - 1) if a.nodes == 32
              else '1,%d,%d' % (a.nodes // 4, a.nodes - 1)):
        p = os.path.join(tmp, 'rot_%s.txt' % g.replace(',', '_'))
        gen('port_plan_gen.py',
            ['--nodes', str(a.nodes), '--demand-file', a.demand_file,
             '--reconf-us', str(a.reconf_us), '--mode', 'rot',
             '--gens', g], p)
        cands.append(('rot{%s}' % g, p))
    for d in range(1, a.nodes):
        p = os.path.join(tmp, 'ron_d%d.txt' % d)
        gen('ron_gen.py',
            ['--nodes', str(a.nodes),
             '--bytes-per-pair', str(a.bytes_per_pair),
             '--reconf-us', str(a.reconf_us), '--strategy', 'd%d' % d,
             '--demand-file', a.demand_file], p)
        cands.append(('ron-d%d' % d, p))
    if a.static_plan:
        # re-emit with THIS cell's reconfiguration delay (the artifact
        # file may carry another R in its header)
        p = os.path.join(tmp, 'static.txt')
        with open(a.static_plan) as f, open(p, 'w') as g2:
            for line in f:
                if line.startswith('gbps'):
                    line = re.sub(r'reconf_us [0-9.e+]+',
                                  'reconf_us %g' % a.reconf_us, line)
                g2.write(line)
        cands.append(('static-theta', p))

    best = None
    for name, p in cands:
        if not os.path.exists(p):
            continue
        r = run_async(a.bin, p)
        if r is None:
            continue
        t, pr = r
        sys.stderr.write('cand %-12s %10.1f us  port_reconfigs %d\n'
                         % (name, t, pr))
        if best is None or t < best[1]:
            best = (name, t, pr, p)
    if best is None:
        sys.stderr.write('no feasible candidate\n')
        sys.exit(1)
    if a.keep_plan:
        with open(best[3]) as f, open(a.keep_plan, 'w') as g2:
            g2.write(f.read())
    print('mosaic_pr PASSED winner %s completion_us %.1f port_reconfigs %d'
          % (best[0], best[1], best[2]))

if __name__ == '__main__':
    main()
