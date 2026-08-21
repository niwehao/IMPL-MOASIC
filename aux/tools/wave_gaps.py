#!/usr/bin/env python3
"""wave_gaps.py —— 从 mosaic trace 环量化两件事(Raven wave 编排的归因输入):

1. **NIC 发送空窗**:collective 窗口内相邻 TE_SLICE_TX(跨机)之间 > gap_us 的
   间隙合计 = "NIC 在等下一波/下一片"的时间;按 (coll, wave) 归组,
   区分「wave 间空窗」(gap 两侧 wave 不同)与「wave 内空窗」。
2. **两腿重叠**:机内腿(hop path 全同 host 的 TE_SLICE_POST..DELIV)与跨机腿
   (TE_SLICE_TX..RECV/DELIV)的活跃区间并集/交集,重叠率 = 交/较短腿。

用法:
  ./tools/wave_gaps.py /path/mosaic_rank0.trace [--gap-us 20] [--g 4]
输出:每 collective 一行(coll, 总时长, nic 忙, nic 空窗(wave间/内), 腿重叠率)
口径:单 rank 视角、TE_* 时戳均为该 rank 本地钟;空窗阈值默认 20µs。
"""
import argparse
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<QHHIQQ")   # t_ns, ev, rank, a, b, c
EV = {1: "WAVE_EMIT", 2: "POST", 3: "DELIV", 9: "TX", 10: "RECV",
      11: "TXDONE", 12: "REPOST", 13: "CBEGIN", 14: "CEND"}


def load(path):
    with open(path, "rb") as f:
        head = f.read(64)
        cur = struct.unpack("<Q", head[:8])[0]
        cap = struct.unpack("<I", head[8:12])[0]
        data = f.read()
    n = min(cur, cap)
    start = (cur - n) % cap if cap else 0
    out = []
    for i in range(n):
        off = ((start + i) % cap) * 32
        out.append(REC.unpack_from(data, off))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-us", type=float, default=20.0)
    ap.add_argument("--g", type=int, default=4, help="gpus_per_host(判机内/跨机)")
    a = ap.parse_args()
    recs = load(a.trace)

    # 按 CBEGIN/CEND 切窗口
    colls = []
    cur = None
    for t, ev, rank, aa, b, c in recs:
        if ev == 13:
            cur = {"id": aa, "t0": t, "t1": None, "tx": [], "wave_of": {},
                   "nv": [], "nic": []}
            colls.append(cur)
        elif ev == 14 and cur is not None:
            cur["t1"] = t
        elif cur is None:
            continue
        elif ev == 9:      # TX:跨机腿投出
            cur["tx"].append(t)
            cur["nic"].append((t, t))          # 起点;终点由 TXDONE/RECV 侧近似
        elif ev == 11 and cur["nic"]:
            s, _ = cur["nic"][-1]
            cur["nic"][-1] = (s, t)
        elif ev == 2:      # POST:含机内(hop 判 src/dst 同 host)
            src, dst = aa >> 16, aa & 0xFFFF
            if src // a.g == dst // a.g:
                cur["nv"].append((t, t))
        elif ev == 3:
            src, dst = aa >> 16, aa & 0xFFFF
            if src // a.g == dst // a.g and cur["nv"]:
                s, _ = cur["nv"][-1]
                cur["nv"][-1] = (s, t)

    def spans_union(spans):
        if not spans:
            return 0, []
        spans = sorted(spans)
        merged = [list(spans[0])]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return sum(e - s for s, e in merged), merged

    print(f"{'coll':>5} {'total_us':>9} {'nic_busy':>9} {'gap_us':>8} "
          f"{'ngaps':>5} {'nv_us':>8} {'overlap%':>8}")
    for cw in colls:
        if cw["t1"] is None or not cw["tx"]:
            continue
        total = (cw["t1"] - cw["t0"]) / 1e3
        txs = sorted(cw["tx"])
        gaps = 0.0
        ngap = 0
        for i in range(1, len(txs)):
            g = (txs[i] - txs[i - 1]) / 1e3
            if g > a.gap_us:
                gaps += g
                ngap += 1
        nic_busy, nic_m = spans_union(cw["nic"])
        nv_busy, nv_m = spans_union(cw["nv"])
        inter = 0
        i = j = 0
        while i < len(nic_m) and j < len(nv_m):
            s = max(nic_m[i][0], nv_m[j][0])
            e = min(nic_m[i][1], nv_m[j][1])
            if e > s:
                inter += e - s
            if nic_m[i][1] < nv_m[j][1]:
                i += 1
            else:
                j += 1
        shorter = min(nic_busy, nv_busy)
        ov = (inter / shorter * 100) if shorter else float("nan")
        print(f"{cw['id']:>5} {total:>9.1f} {nic_busy/1e3:>9.1f} {gaps:>8.1f} "
              f"{ngap:>5} {nv_busy/1e3:>8.1f} {ov:>8.1f}")


if __name__ == "__main__":
    main()
