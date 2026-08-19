"""mosaic_torch 正确性测试(world=2,双机,GPU 张量)。

判据(全部 bit-exact,以 NCCL 的 torch.distributed.all_to_all_single 为 oracle):
  1) mt.all_to_all_single 前向;2) autograd backward(= splits 转置的 a2a);
  3) 非均匀 splits 与空行(MoE dispatch 真实形态)。

用法:MT_BACKEND=mosaic bash baseline-rt/run2.sh mosaic-rt/tests/test_mosaic_torch.py
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import mosaic_torch as mt  # noqa: E402


def log(rank, *a):
    print(f"[r{rank}]", *a, flush=True)


def main():
    backend = os.environ.get("MT_BACKEND", "mosaic")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")
    torch.manual_seed(1234 + rank)
    mt.init(backend=backend, gpus_per_host=int(os.environ.get("MOSAIC_G", "1")))

    H = 64
    fails = 0
    for name, in_rows_by_rank in [
        ("uniform", [[8, 8], [8, 8]]),
        ("skewed", [[3, 13], [11, 5]]),
        ("empty-row", [[0, 16], [9, 0]]),
    ]:
        in_splits = in_rows_by_rank[rank]
        out_splits = [in_rows_by_rank[r][rank] for r in range(world)]
        x = torch.randn(sum(in_splits), H, device=dev)

        ref = torch.empty(sum(out_splits), H, device=dev)
        dist.all_to_all_single(ref, x.contiguous(), out_splits, in_splits)
        torch.cuda.synchronize()

        got = torch.empty_like(ref)
        mt.all_to_all_single(got, x.contiguous(), out_splits, in_splits)
        ok = torch.equal(got, ref)
        log(rank, f"[{backend}] {name} fwd bit-exact:", "PASS" if ok else "FAIL")
        fails += 0 if ok else 1

        # autograd:mosaic 前后向 vs NCCL 手工前后向,比 input.grad
        x1 = x.clone().requires_grad_(True)
        y1 = mt.all_to_all_autograd(x1, out_splits, in_splits)
        gup = torch.randn_like(y1)
        if y1.numel():
            y1.backward(gup)
        gback = torch.empty(sum(in_splits), H, device=dev)
        dist.all_to_all_single(gback, gup.contiguous(), in_splits, out_splits)
        torch.cuda.synchronize()
        ok2 = (x1.grad is None and gback.numel() == 0) or torch.equal(x1.grad, gback)
        log(rank, f"[{backend}] {name} autograd bit-exact:", "PASS" if ok2 else "FAIL")
        fails += 0 if ok2 else 1

    dist.barrier()
    log(rank, "ALL", "PASS" if fails == 0 else f"FAIL({fails})")
    mt.destroy()
    dist.destroy_process_group()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
