#!/usr/bin/env python3
"""trace2csv.py —— 解 MOSAIC 的二进制 trace 环(internal.h:127-146 引用的那个工具)。

**这个文件长期被 internal.h 与多份报告引用但一直不存在**(2026-08-19 补上)。

文件布局(src/util.cc:mtrace_init):
  [0..8)    uint64  写游标 = 累计写入条数(可 > cap,环形覆盖)
  [8..12)   uint32  cap(条数,默认 1<<20,MOSAIC_TRACE_CAP 可改)
  [64..)    cap 条 TraceRec,每条 32B packed:
            uint64 t_ns; uint16 ev; uint16 rank; uint32 a; uint64 b; uint64 c;

用法:
  # 转 CSV(时间轴按各文件自己的 CLOCK_MONOTONIC_RAW,跨机不可直接相减)
  ./tools/trace2csv.py /tmp/mosaic_rank*.trace -o trace.csv

  # 单片生命周期统计(自动配对 src / dst 两个文件)
  ./tools/trace2csv.py /tmp/mosaic_rank0.trace /tmp/mosaic_rank1.trace --stats

  # 只看某个 collective(按 TE_COLL_BEGIN/END 切窗口)
  ./tools/trace2csv.py ... --stats --coll 7
"""
import argparse
import os
import struct
import sys
from collections import defaultdict

REC = struct.Struct("<QHHIQQ")
assert REC.size == 32

EV = {
    1: "WAVE_EMIT", 2: "SLICE_POST", 3: "SLICE_DELIV", 4: "CHA_PUB",
    5: "CHB_PUB", 6: "STAGGER", 7: "DET", 8: "DUMP",
    9: "SLICE_TX", 10: "SLICE_RECV", 11: "TXDONE", 12: "REPOST",
    13: "COLL_BEGIN", 14: "COLL_END", 15: "LOOP",
}
SLICE_EVS = {2, 3, 9, 10, 11, 12}          # a/b/c 是 (src,dst)/off/(len,hop)


def read_trace(path):
    """→ (rank, [rec…]),按写入顺序(已解环)。"""
    with open(path, "rb") as f:
        blob = f.read()
    if len(blob) < 64:
        raise SystemExit(f"{path}: 太短,不像 trace 文件")
    cur = struct.unpack_from("<Q", blob, 0)[0]
    cap = struct.unpack_from("<I", blob, 8)[0]
    if cap == 0 or 64 + cap * 32 > len(blob) + 32:
        raise SystemExit(f"{path}: cap={cap} 与文件大小不符")
    n = min(cur, cap)
    # 环形:cur>cap 时最老的一条在 cur%cap
    start = (cur - n) % cap
    out = []
    for k in range(n):
        i = (start + k) % cap
        t, ev, rank, a, b, c = REC.unpack_from(blob, 64 + i * 32)
        if t == 0:
            continue
        out.append((t, ev, rank, a, b, c))
    rank = out[0][2] if out else -1
    dropped = cur - n
    return rank, out, dropped


def decode(rec):
    t, ev, rank, a, b, c = rec
    d = {"t_ns": t, "ev": EV.get(ev, str(ev)), "rank": rank, "a": a, "b": b, "c": c}
    if ev in SLICE_EVS:
        d["src"], d["dst"] = a >> 16, a & 0xFFFF
        d["off"], d["len"], d["hop"] = b, c >> 8, c & 0xFF
    return d


def to_csv(files, out):
    w = open(out, "w") if out else sys.stdout
    w.write("file,rank,t_ns,ev,a,b,c,src,dst,off,len,hop\n")
    for p in files:
        rank, recs, dropped = read_trace(p)
        if dropped:
            print(f"# {p}: 环已覆盖,丢弃最老 {dropped} 条", file=sys.stderr)
        base = os.path.basename(p)
        for r in recs:
            d = decode(r)
            w.write(f"{base},{d['rank']},{d['t_ns']},{d['ev']},{d['a']},{d['b']},{d['c']},"
                    f"{d.get('src','')},{d.get('dst','')},{d.get('off','')},"
                    f"{d.get('len','')},{d.get('hop','')}\n")
    if out:
        w.close()
        print(f"→ {out}", file=sys.stderr)


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * q))]


def fmt(name, v, unit="µs"):
    if not v:
        return f"  {name:<34} —(0 样本)"
    return (f"  {name:<34} n={len(v):<5} p50={pct(v,.5):8.1f} "
            f"p90={pct(v,.9):8.1f} max={max(v):8.1f} {unit}")


def stats(files, coll=None):
    """单片生命周期:src 本地闭合段 + dst 本地闭合段 + 由减法得到的跨机段。"""
    per = {}
    for p in files:
        rank, recs, dropped = read_trace(p)
        per[rank] = [decode(r) for r in recs]
        if dropped:
            print(f"# rank{rank}: 环覆盖,丢弃最老 {dropped} 条", file=sys.stderr)

    for rank in sorted(per):
        ev = per[rank]
        # 按 collective 切窗口
        wins = []
        beg = None
        for e in ev:
            if e["ev"] == "COLL_BEGIN":
                beg = e
            elif e["ev"] == "COLL_END" and beg is not None:
                wins.append((beg, e))
                beg = None
        if coll is not None:
            wins = [w for w in wins if w[0]["a"] == coll]
        print(f"\n=== rank {rank} · {len(wins)} 个完整 collective 窗口 ===")
        if not wins:
            print("  (没有成对的 COLL_BEGIN/END —— 是不是没开 MOSAIC_TRACE=1?)")
            continue

        # 逐片配对:key = (src,dst,off)
        seg = defaultdict(list)
        loop = []
        coll_us, begin2first, last2end = [], [], []
        for b, e in wins:
            lo, hi = b["t_ns"], e["t_ns"]
            win = [x for x in ev if lo <= x["t_ns"] <= hi]
            byk = defaultdict(dict)
            for x in win:
                if "off" in x:
                    byk[(x["src"], x["dst"], x["off"])][x["ev"]] = x["t_ns"]
                elif x["ev"] == "LOOP" and x["b"]:
                    loop.append(x["b"] / 1e3)
            for k, m in byk.items():
                def d(p_, q_):
                    if p_ in m and q_ in m and m[q_] >= m[p_]:
                        seg[f"{p_} → {q_}"].append((m[q_] - m[p_]) / 1e3)
                d("SLICE_POST", "SLICE_TX")      # 入队 → D2H 完成、WR 实投(src)
                d("SLICE_TX", "TXDONE")          # WR 投出 → 发送完成(src,含对端 ack)
                d("SLICE_RECV", "SLICE_DELIV")   # CQ 命中 → H2D 落 recvbuf(dst)
                d("SLICE_RECV", "REPOST")        # CQ 命中 → credit 释放(dst)
            # 片间节拍(src 本地):TX(i) → POST(i+1)
            txs = sorted([x for x in win if x["ev"] == "SLICE_TX"], key=lambda z: z["t_ns"])
            posts = sorted([x for x in win if x["ev"] == "SLICE_POST"], key=lambda z: z["t_ns"])
            for i in range(len(txs) - 1):
                seg["TX(i) → TX(i+1) 片间节拍"].append((txs[i + 1]["t_ns"] - txs[i]["t_ns"]) / 1e3)
            for i in range(min(len(txs), len(posts)) - 1):
                if posts[i + 1]["t_ns"] >= txs[i]["t_ns"]:
                    seg["TX(i) → POST(i+1)"].append((posts[i + 1]["t_ns"] - txs[i]["t_ns"]) / 1e3)
            # collective 首尾
            coll_us.append((hi - lo) / 1e3)
            if posts:
                begin2first.append((min(x["t_ns"] for x in posts) - lo) / 1e3)
            delivs = [x for x in win if x["ev"] == "SLICE_DELIV"]
            if delivs:
                last2end.append((hi - max(x["t_ns"] for x in delivs)) / 1e3)

        print(fmt("COLL_BEGIN → COLL_END(全长)", coll_us))
        print(fmt("COLL_BEGIN → 首片 POST", begin2first))
        for k in ("SLICE_POST → SLICE_TX", "SLICE_TX → TXDONE",
                  "SLICE_RECV → SLICE_DELIV", "SLICE_RECV → REPOST",
                  "TX(i) → TX(i+1) 片间节拍", "TX(i) → POST(i+1)"):
            if k in seg:
                print(fmt(k, seg[k]))
        print(fmt("末片 DELIV → COLL_END", last2end))
        if loop:
            print(fmt("事件循环单圈(TE_LOOP)", loop))


def align(files):
    """跨机时钟偏移估计 + 跨机段(两台机器各自的 CLOCK_MONOTONIC_RAW 无公共 epoch)。

    方法(NTP 式,复用已有事件,不加探针):对同一片,
    src 有 TX(WR 投出)与 TXDONE(send CQ 完成 = 对端 NIC 已 ack),
    dst 有 RECV(recv CQ 命中)。设去/回程对称,该片"到达"在 src 时钟上
    ≈ (TX+TXDONE)/2,于是 offset ≈ median( RECV_dst − (TX+TXDONE)/2 )。

    **两个方向分别算**:理论上 off(0→1) = −off(1→0),两者之和即为
    「不对称 + 轮询」引入的残余误差,直接打印出来当误差棒。
    **仅在每片单独在飞时可信**(片一多 TXDONE 含排队,估计失效)。
    """
    ev_by_rank = {}
    for f in files:
        rank, recs, _ = read_trace(f)
        ev_by_rank[rank] = [decode(r) for r in recs]
    if len(ev_by_rank) < 2:
        print("对时需要两个 rank 的 trace 文件"); return

    def idxed(rank, name):
        """{(src,dst,off,第几次出现): t_ns}"""
        cnt, out = defaultdict(int), {}
        for d in ev_by_rank[rank]:
            if d.get("ev") == name and "off" in d:
                k0 = (d["src"], d["dst"], d["off"])
                out[k0 + (cnt[k0],)] = d["t_ns"]
                cnt[k0] += 1
        return out

    ranks = sorted(ev_by_rank)
    offs_dir = {}
    for s_r, d_r in ((ranks[0], ranks[1]), (ranks[1], ranks[0])):
        tx, txd = idxed(s_r, "SLICE_TX"), idxed(s_r, "TXDONE")
        rcv, dlv = idxed(d_r, "SLICE_RECV"), idxed(d_r, "SLICE_DELIV")
        keys = [k for k in tx if k[0] == s_r and k in txd and k in rcv]
        if not keys:
            continue
        o = sorted(rcv[k] - (tx[k] + txd[k]) // 2 for k in keys)
        off = o[len(o) // 2]
        offs_dir[(s_r, d_r)] = off
        hr = sorted((txd[k] - tx[k]) / 2e3 for k in keys)
        print(f"\n=== 方向 rank{s_r} → rank{d_r}(n={len(keys)} 片)===")
        print(f"  offset(dst−src)   = {off/1e3:10.1f} µs   "
              f"[p10 {o[len(o)//10]/1e3:.1f} .. p90 {o[len(o)*9//10]/1e3:.1f}]")
        print(f"  半程 (TXDONE−TX)/2 = {hr[len(hr)//2]/1:10.1f} µs")
        print(fmt("TX(src) → RECV(dst) 跨机", sorted((rcv[k] - off - tx[k]) / 1e3 for k in keys)))
        kk = [k for k in keys if k in dlv]
        if kk:
            print(fmt("TX(src) → DELIV(dst) 跨机", sorted((dlv[k] - off - tx[k]) / 1e3 for k in kk)))
    if len(offs_dir) == 2:
        a_, b_ = list(offs_dir.values())
        print(f"\n  自洽性检验:两方向 offset 之和 = {(a_ + b_)/1e3:.1f} µs "
              f"(理想 0;这就是残余误差,跨机段的数字只精确到这个量级)")


def main():
    ap = argparse.ArgumentParser(description="解 MOSAIC trace 环")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", help="写 CSV 到文件(默认 stdout)")
    ap.add_argument("--stats", action="store_true", help="打单片生命周期统计,不出 CSV")
    ap.add_argument("--coll", type=int, help="只统计这个 coll_id")
    ap.add_argument("--align", action="store_true", help="跨机对时 + 跨机段(需两个 rank 的文件)")
    a = ap.parse_args()
    if a.align:
        align(a.files)
    elif a.stats:
        stats(a.files, a.coll)
    else:
        to_csv(a.files, a.out)


if __name__ == "__main__":
    main()
