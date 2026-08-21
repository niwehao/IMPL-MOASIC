#!/usr/bin/env python3
# x45: build experiments/mosaic_x1/45_spectrum/{results.csv,plot.png}
# from cache/optical/x45_spectrum.log. Opponent numbers are FROZEN
# archives (35 barrier best / 39 PR / 43 O-A / 40 drift), hard-coded
# here on purpose: rerunning opponents is forbidden by the bench rules.
import os, re, sys

HT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
LOG = os.path.join(HT, 'cache/optical/x45_spectrum.log')
OUT = os.path.join(HT, 'experiments/mosaic_x1/45_spectrum')

LB = {'uniform': 650.1, 'zipf10': 5097.7, 'zipf20': 12658.0,
      'persrc': 1058.1}
BARRIER = {('uniform', 10): 975.617, ('uniform', 100): 2695.65,
           ('uniform', 1000): 7496.71, ('uniform', 10000): 20649.9,
           ('zipf10', 10): 5423.16, ('zipf10', 100): 7895.28,
           ('zipf10', 1000): 13272.3, ('zipf10', 10000): 22272.3,
           ('zipf20', 10): 12983.5, ('zipf20', 100): 15085.9,
           ('zipf20', 1000): 15985.9, ('zipf20', 10000): 24985.9,
           ('persrc', 10): 1583.74, ('persrc', 100): 4158.83,
           ('persrc', 1000): 10004.1, ('persrc', 10000): 19004.1}
PR39 = {('uniform', 10): 960.617, ('uniform', 100): 2674.65,
        ('uniform', 1000): 7376.71, ('uniform', 10000): 20409.9,
        ('zipf10', 10): 5408.16, ('zipf10', 100): 6620.06,
        ('zipf10', 1000): 10220.1, ('zipf10', 10000): 22272.3,
        ('zipf20', 10): 12688.5, ('zipf20', 100): 12958.5,
        ('zipf20', 1000): 14777.3, ('zipf20', 10000): 24985.9,
        ('persrc', 10): 1583.74, ('persrc', 100): 4131.85,
        ('persrc', 1000): 8253.41, ('persrc', 10000): 19004.1}
OA43 = {('uniform', 10): 2078.67, ('uniform', 100): 5363.62,
        ('uniform', 1000): 7453.1, ('uniform', 10000): 20522.2,
        ('zipf10', 10): 6748.7, ('zipf10', 100): 8475.5,
        ('zipf10', 1000): 10275.5, ('zipf10', 10000): 22676.1,
        ('zipf20', 10): 14002.7, ('zipf20', 100): 14182.7,
        ('zipf20', 1000): 15982.7, ('zipf20', 10000): 26886.0,
        ('persrc', 10): 2865.6, ('persrc', 100): 5728.7,
        ('persrc', 1000): 7528.7, ('persrc', 10000): 22205.2}
# drift opponents (39/40 archives): serial async, pr_online, O-W
DR_SER = 16717.0
DR_PRO = {3000: 12922.6, 5000: 14644.7, 8000: 19695.2, 9500: 16913.3}
DR_OW = {3000: 13043.5, 5000: 14470.4, 8000: 14522.3, 9500: 15018.0}
# n64 archives (39/40): barrier best, PR, O-W; LB from x39 report
N64 = {'zipf10_1000': (43515.7, 20589.6, 22635.3, 8759.7),
       'zipf10_100': (14245.5, 11902.0, 20835.3, 8759.7),
       'persrc_1000': (35737.4, 20098.7, 32244.3, 2270.0)}

rows = []
md5 = ''
for line in open(LOG):
    m = re.match(r'# binary_md5 (\S+)', line)
    if m:
        md5 = m.group(1)
    m = re.match(r'CELL (\S+) (\d+) optical (PASSED|FAILED) spectrum '
                 r'winner (\S+) .*?planning_ms ([0-9.]+)'
                 r'(?: replan_ms [0-9.]+)?'
                 r'(?: port_reconfigs (\d+))? completion_us ([0-9.]+)',
                 line)
    if m:
        dm, r = m.group(1), int(m.group(2))
        st, wname = m.group(3), m.group(4)
        plan_ms, us = float(m.group(5)), float(m.group(7))
        pr = m.group(6) or '-'
        cut = 'OK' if us + 1e-6 >= LB[dm] + r else 'VIOLATION'
        rows.append((dm, r, 'mosaic_os',
                     'winner=%s portR=%s plan_ms=%.0f' % (wname, pr,
                                                          plan_ms),
                     us, st,
                     '%.3f' % (BARRIER[(dm, r)] / us),
                     '%.3f' % (PR39[(dm, r)] / us),
                     '%.3f' % (OA43[(dm, r)] / us), cut))
    m = re.match(r'DRIFT (\d+) optical (PASSED|FAILED) spectrum '
                 r'winner (\S+) .*?completion_us ([0-9.]+)', line)
    if m:
        t, st = int(m.group(1)), m.group(2)
        us = float(m.group(4))
        rows.append(('drift_z10_at%d' % t, 1000, 'mosaic_os',
                     'winner=%s' % m.group(3), us, st,
                     '%.3f' % (DR_SER / us),
                     '%.3f' % (DR_PRO[t] / us),
                     '%.3f' % (DR_OW[t] / us), 'OK'))
    m = re.match(r'N64 (\S+) optical (PASSED|FAILED) spectrum '
                 r'winner (\S+) .*?planning_ms ([0-9.]+).*?'
                 r'completion_us ([0-9.]+)', line)
    if m:
        key, st = m.group(1), m.group(2)
        us = float(m.group(5))
        bb, pr39, ow, lb = N64[key]
        r = int(key.split('_')[-1])
        cut = 'OK' if us + 1e-6 >= lb + r else 'VIOLATION'
        rows.append((key + '_n64', r, 'mosaic_os',
                     'winner=%s plan_ms=%.0f' % (m.group(3),
                                                 float(m.group(4))),
                     us, st, '%.3f' % (bb / us), '%.3f' % (pr39 / us),
                     '%.3f' % (ow / us), cut))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, 'results.csv'), 'w') as f:
    f.write('demand,R_us,scheme,detail,completion_us,status,'
            'vs_barrier35,vs_pr39,vs_oa43,cut_ok\n')
    f.write('_meta,0,frozen_binary_md5,%s,0,-,,,,\n' % md5)
    f.write('_meta,0,opponents,frozen archives 35(barrier)+39(PR)'
            '+43(O-A)+40(drift/O-W); none rerun,0,-,,,,\n')
    for row in rows:
        f.write(','.join(str(x) for x in row) + '\n')
print('rows:', len(rows))
bad = [r for r in rows if r[5] != 'PASSED' or r[9] != 'OK']
print('non-passed-or-cut:', bad if bad else 'none')
judge = [r for r in rows if r[0] in ('zipf10', 'zipf20', 'persrc')
         and float(r[6]) < 1.0]
print('judged cells below 1.0x vs barrier:', judge if judge else 'none')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    RS = [10, 100, 1000, 10000]
    for ax, dm in zip(axes, ['uniform', 'zipf10', 'zipf20', 'persrc']):
        os_v = {r: None for r in RS}
        for row in rows:
            if row[0] == dm:
                os_v[row[1]] = row[4]
        ax.plot(RS, [BARRIER[(dm, r)] for r in RS], 'o--',
                label='barrier best (35)', color='#888')
        ax.plot(RS, [PR39[(dm, r)] for r in RS], 's--',
                label='PR (39)', color='#c77')
        ax.plot(RS, [OA43[(dm, r)] for r in RS], '^--',
                label='O-A (43)', color='#7a7')
        ax.plot(RS, [os_v[r] for r in RS], 'D-',
                label='MOSAIC-O-S (45)', color='#26c', lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(dm)
        ax.set_xlabel('R (us)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('completion (us)')
    axes[0].legend(fontsize=8)
    fig.suptitle('x45 MOSAIC-O-S vs frozen opponents (n=32, k=1, '
                 '800G, 4MB chunk, pay-initial)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'plot.png'), dpi=110)
    print('plot saved')
except Exception as e:
    print('plot skipped:', e)
