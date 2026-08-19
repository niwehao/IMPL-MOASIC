#!/usr/bin/env python3
"""割下界(cut lower bound)—— 对齐模拟器 tools/x1_bounds.py 的口径。

每个 GPU 的入口/出口各有两条并行管:NV 管(机内 + 中转)与 NIC 管。
divisible load 下某 dst 的完成时间不可能小于
    max( same_d/β_V , (same_d+cross_d)/(β_V+β_N) )
第一项 = 机内流量只能走 NV 管;第二项 = 两管一起用的注水界。出口侧同理。
再加每台 host 的 G 块 vNIC 聚合入口界。全局下界 = 三者取 max。

用法: build/bench_a2a --dump-demand ... | tools/hw_bounds.py
"""
import sys

lines = [l.strip() for l in sys.stdin if l.strip()]
hdr = dict(kv.split('=') for kv in lines[0].lstrip('# ').split())
n, G = int(hdr['n']), int(hdr['G'])
bV = float(hdr['nv']) * 1e9 / 8
bN = float(hdr['nic']) * 1e9 / 8
D = [[int(x) for x in l.split(',')] for l in lines[1:1 + n]]
host = lambda r: r // G

worst, who = 0.0, ''
for d in range(n):
    same = sum(D[s][d] for s in range(n) if s != d and host(s) == host(d))
    cross = sum(D[s][d] for s in range(n) if host(s) != host(d))
    t = max(same / bV, (same + cross) / (bV + bN)) * 1e6
    if t > worst: worst, who = t, f'ingress@{d}'
for s in range(n):
    same = sum(D[s][d] for d in range(n) if d != s and host(d) == host(s))
    cross = sum(D[s][d] for d in range(n) if host(d) != host(s))
    t = max(same / bV, (same + cross) / (bV + bN)) * 1e6
    if t > worst: worst, who = t, f'egress@{s}'
for h in range(n // G):
    cross = sum(D[s][d] for s in range(n) for d in range(n) if host(d) == h and host(s) != h)
    t = cross / (G * bN) * 1e6
    if t > worst: worst, who = t, f'host{h}-nic-in'
print(f"{worst:.1f}  ({who})")

# ---- 细分诊断:DIRECT-only 的结构上限 vs 两管注水界 ----
if '-v' in sys.argv:
    print(f"{'dst':>4} {'same(MB)':>9} {'cross(MB)':>10} {'DIRECT界(us)':>13} {'两管界(us)':>12} {'可得倍数':>9}")
    for d in range(n):
        same = sum(D[s][d] for s in range(n) if s != d and host(s) == host(d))
        cross = sum(D[s][d] for s in range(n) if host(s) != host(d))
        # DIRECT-only:机内走 NV、跨机只能进自己那块 vNIC,两者并行
        t_direct = max(same / bV, cross / bN) * 1e6
        # 两管注水:跨机流量可以经中转改走 NV 管(RX/SPREAD)
        t_pipe = max(same / bV, (same + cross) / (bV + bN)) * 1e6
        print(f"{d:>4} {same/1e6:>9.2f} {cross/1e6:>10.2f} {t_direct:>13.1f} {t_pipe:>12.1f} {t_direct/max(t_pipe,1e-9):>9.2f}x")
