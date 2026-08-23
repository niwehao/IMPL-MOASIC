"""NCCL / MOSAIC 的真机对拍(world=2,每机一张 L40S,RoCE 100GbE 不限速)。

定位(Imp_Plan/04 §2):这是**真实硬件参考点**,不是主对比。
主对比(mosaic-rt/aux/scripts/sweep.sh)跑在限速的 8-rank 模拟 fabric 上,
测的是调度算法;这里测的是"我们的运行时相对 NCCL 还差多少工程开销"。

NCCL 侧用 `dist.all_to_all_single` 的非均匀 splits 形式 —— 这正是 PyTorch/
Megatron 的 MoE dispatch 走的路径,也是 NCCL 做 alltoallv 的官方写法
(ncclGroupStart + 逐 peer Send/Recv)。

注意 NCCL 的结构性限制:一个 communicator 里**不允许同一张卡上有多个 rank**,
所以 NCCL 基线只能 world=2;而 MOSAIC 的模拟 fabric 是 8 rank 共 2 张卡。
两者不能直接比时间,只能各自与自己的下界比 —— 这一点在报告里必须写清楚。

跑法(rank0 在 gpu-02):
    MASTER_ADDR=192.168.100.2 MASTER_PORT=29511 RANK=0 WORLD_SIZE=2 python bench_nccl.py
"""
import argparse
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

# 本文件在 Impl/mosaic-rt/nccl/ 下,绑定层在 ../aux/python(Raven 模板另经
# PYTHONPATH 提供,两者并存无害)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "aux", "python"))


def build_demand(n, msg, alpha, seed, mode="hotdst", gran=4096):
    """与 mosaic-rt/bench/bench_a2a.cc 的 build_demand 同构(同一 xorshift 与舍入)。"""
    def xs(s):
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 7
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        return s
    s = 0x9E3779B97F4A7C15 ^ (seed + 1)
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        s = xs(s)
        j = s % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    rank_of = [0] * n
    for r, node in enumerate(perm):
        rank_of[node] = r
    w = [1.0 if mode == "uniform" else 1.0 / ((rank_of[d] + 1) ** alpha) for d in range(n)]
    total = (n - 1) * msg
    M = [[0] * n for _ in range(n)]
    for src in range(n):
        wsum = sum(w[d] for d in range(n) if d != src)
        assigned, hottest = 0, -1
        for d in range(n):
            if d == src:
                continue
            b = max(gran, int(total * w[d] / wsum) // gran * gran)
            M[src][d] = b
            assigned += b
            if hottest < 0 or w[d] > w[hottest]:
                hottest = d
        if assigned < total and hottest >= 0:
            M[src][hottest] += total - assigned
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="nccl", choices=["nccl", "mosaic"])
    ap.add_argument("--msg", type=int, default=8 << 20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="hotdst")
    ap.add_argument("--sched", default="mosaic")
    # 相位(mosaic.h 契约):本项目每个 a2a 必居其一 —— build_demand 的各模式
    # 都是行全等构造 ⇒ 默认 1 = dispatch(src 决策);2 = combine(dst 决策)
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2])
    a = ap.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    # R16 同款:多卡节点上不能硬编码 cuda:0(否则同机 4 rank 挤一张卡,
    # NCCL 直接报 Duplicate GPU)。本机 world=2 每机一卡时 local 恒 0,行为不变。
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    ndev = torch.cuda.device_count()
    torch.cuda.set_device(local % ndev if ndev else 0)
    ROW = 4096                                     # 每"行"字节数(bf16 × 2048 hidden)
    D = build_demand(world, a.msg, a.alpha, a.seed, a.mode, gran=ROW)
    srow = [D[rank][j] // ROW for j in range(world)]
    rrow = [D[i][rank] // ROW for i in range(world)]
    send = torch.randn(sum(srow), ROW // 2, dtype=torch.bfloat16, device="cuda")
    recv = torch.empty(sum(rrow), ROW // 2, dtype=torch.bfloat16, device="cuda")

    if a.backend == "nccl":
        dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank)
        def run():
            dist.all_to_all_single(recv, send, rrow, srow)
            torch.cuda.synchronize()
        def sync():
            dist.barrier(); torch.cuda.synchronize()
    else:
        import mosaic_py as mp
        hosts = os.environ.get("MOSAIC_HOSTS", "192.168.100.2,192.168.100.1")
        comm = mp.Comm(rank=rank, nranks=world, gpus_per_host=int(os.environ.get("MOSAIC_G", 1)),
                       hosts=hosts, sched=a.sched, job=int(os.environ.get("MOSAIC_JOB", 7)))
        phase = a.phase   # run() 内部的位移累加变量会遮蔽 argparse 的 a,先取出
        def run():
            st = torch.cuda.current_stream().cuda_stream or 1
            sc = [r * ROW for r in srow]; rc2 = [r * ROW for r in rrow]
            sd = []; rd = []; a = 0; b = 0
            for x in sc: sd.append(a); a += x
            for x in rc2: rd.append(b); b += x
            comm.alltoallv(send.data_ptr(), sc, sd, recv.data_ptr(), rc2, rd,
                           stream=st, phase=phase)
            # 只等 caller 流(= 门核放行 = recvbuf 就绪,与训练同语义)。
            # 设备级 synchronize 会连消费者核一起等,把 recv-即-开门量没了。
            torch.cuda.current_stream().synchronize()
        def sync():
            # 计时窗外吸收上一笔的残余排空(stats(sync=True) 走 wait_coll)——
            # 训练里这段与计算重叠,不该算进 a2a
            comm.stats(sync=True)
            comm.barrier()

    # ── 端到端校验(一次,计时前):行首 8B 写入 (src<<16|dst_row_global) 指纹,
    # 收端逐 src 段抽查首/末行。没有它,QP 静默死 + 数据丢失也会给出"漂亮"的
    # 计时(29425220-033016 实测教训:W=2 α2 下 r15 QP 死,其余 rank 假完成)。
    def verify():
        rows_out = 0
        for j in range(world):
            for k in range(srow[j]):
                send[rows_out + k, 0] = float(rank * 16 + j)   # ≤255,bf16 精确
            rows_out += srow[j]
        sync(); run(); sync()
        rin = 0
        bad = 0
        for s in range(world):
            for k in (0, rrow[s] - 1) if rrow[s] else ():
                v = float(recv[rin + k, 0].item())
                if abs(v - (s * 16 + rank)) > 0.5:
                    bad += 1
            rin += rrow[s]
        if bad:
            print(f"[prank {rank}] VERIFY FAILED bad={bad}", flush=True)
        return bad == 0
    okv = verify()

    ts = []
    for it in range(a.warmup + a.iters):
        sync()
        t0 = time.perf_counter()
        run()
        t1 = time.perf_counter()
        if it >= a.warmup:
            ts.append((t1 - t0) * 1e6)
    tot = sum(D[rank]) - D[rank][rank]
    med = statistics.median(ts)
    # 尾窗中位(后 1/4):在线自适应(W-arm)的学习期不摊进稳态口径
    tail = statistics.median(ts[-max(1, len(ts)//4):])
    if not okv:
        med = float("nan")     # 校验失败的数字不许被引用
    # 每 rank 完成结构(冷 dst 早完成 = MOSAIC 对 NCCL 的差异化轴;两后端同尺):
    # 各 rank 自报 barrier→本 rank 调用返回的墙钟中位。skew 下看分布,不只 max。
    print(f"[prank {rank}] med={med:.1f}us col_in={sum(D[s][rank] for s in range(world)) - D[rank][rank]}",
          flush=True)
    if rank == 0:
        print(f"backend={a.backend:6s} world={world} msg={a.msg>>20}MB/pair "
              f"med={med:9.1f}us tail_med={tail:9.1f}us p99={max(ts):9.1f}us  单 rank 出向 {tot/med*8/1000:6.2f} Gbps")
    if a.backend == "nccl":
        dist.destroy_process_group()
    else:
        comm.close()


if __name__ == "__main__":
    main()
