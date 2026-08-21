#!/usr/bin/env python3
# Consolidate x32_*.log batch logs into results.csv (32 组归档).
import re, csv, os
SRC = 'cache/nonuniform'
OUT = 'experiments/mosaic_x1/32_paradox_pacing/results.csv'
rows = []
for log, sect in [('x32_acct.log','acct'), ('x32_p1rail.log','p1rail'),
                  ('x32_p1pod.log','p1pod'), ('x32_p23.log','p23'),
                  ('x32_p23b.log','p23b'), ('x32_mech.log','mech')]:
    path = os.path.join(SRC, log)
    if not os.path.exists(path):
        continue
    binmd5 = ''
    for l in open(path):
        if l.startswith('# binary'):
            binmd5 = l.split()[2][:8]
            continue
        m = re.match(r'(.+?) validation (\w+) rounds \d+ completion_us ([\d.]+)'
                     r'(?:\s+p99_us ([\d.]+))?', l)
        if m:
            rows.append([sect, m.group(1).strip(), m.group(2),
                         m.group(3), m.group(4) or '', binmd5])
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['section', 'label', 'validation', 'completion_us',
                'p99_us', 'binary_md5'])
    w.writerows(rows)
print(f"{len(rows)} rows -> {OUT}")
