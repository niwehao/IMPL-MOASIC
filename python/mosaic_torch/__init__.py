"""mosaic_torch —— EP 通信的 drop-in 库(L4,Imp_Plan/05 §2.2)。

给训练框架的三件东西:

  mt.init(backend="mosaic"|"fast"|"direct"|"nccl", ...)
      从 torch.distributed 的 world 建 mosaicComm(nccl 后端不建,原生路由)。
  mt.all_to_all_single(output, input, output_split_sizes, input_split_sizes,
                       group=None, async_op=False)
      与 torch.distributed.all_to_all_single **同签名**(dim0 行切分;
      splits=None = 均匀)。backend=nccl 时直接转发给 torch.distributed ——
      三后端在训练脚本里只差 init 的一个参数,公平性由此成立。
  mt.all_to_all_autograd(input, out_splits, in_splits)
      autograd 版:backward = splits 转置后的同一 alltoallv(与 Megatron
      _AllToAll 逐语义对应)。
  mt.patch_megatron()
      单点替换 megatron.core.tensor_parallel.mappings 的 all_to_all 原语
      (符号断言 fail-fast,E4xxx;DP allreduce 等全部留在 NCCL,不受污染)。

流语义(File/log_raven.md §6 D11):默认(MOSAIC_ASYNC_SUBMIT=1)提交时在
torch 当前流上录一个"输入就绪"事件交给引擎,引擎的拷贝流 WaitEvent 它 ——
CPU 协议(demand 交换/注水/通道 B)与调用方流上的尾部计算重叠,无 host 同步。
MOSAIC_ASYNC_SUBMIT=0 退回旧保守口径(提交前 torch.cuda.synchronize)。
调用整体仍是阻塞语义(返回 = 输出已就绪),MoE dispatcher 语义等价。
"""
import os

import torch
import torch.distributed as dist

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/ 目录
import mosaic_py as mp  # noqa: E402

_COMM = None
_BACKEND = "nccl"


def _infer_hosts(world, g):
    """MOSAIC_HOSTS 由启动脚本给(与 bench 同约定);缺省单机。"""
    return os.environ.get("MOSAIC_HOSTS", "127.0.0.1")


def init(backend="mosaic", gpus_per_host=None, **cfg_kw):
    """从 torch.distributed 的 world 建 mosaicComm。必须在 dist.init_process_group
    之后调用。backend="nccl" 时不建 comm(a2a 原生路由),其余建对应调度器的
    comm。gpus_per_host 缺省 = world_size 分给 MOSAIC_NHOSTS(默认 2)台。"""
    global _COMM, _BACKEND
    _BACKEND = backend
    if backend == "nccl":
        return None
    assert dist.is_initialized(), "先 dist.init_process_group 再 mt.init"
    rank, world = dist.get_rank(), dist.get_world_size()
    if gpus_per_host is None:
        nhosts = int(os.environ.get("MOSAIC_NHOSTS", "2"))
        assert world % nhosts == 0, f"world={world} 不能整除 host 数 {nhosts}"
        gpus_per_host = world // nhosts
    _COMM = mp.Comm(rank=rank, nranks=world, gpus_per_host=gpus_per_host,
                    hosts=_infer_hosts(world, gpus_per_host), sched=backend, **cfg_kw)
    return _COMM


def comm():
    assert _COMM is not None, "先 mt.init(backend=...)"
    return _COMM


def backend():
    return _BACKEND


def destroy():
    global _COMM
    if _COMM is not None:
        if _inflight:
            _COMM.wait()          # S3:在途收尾后再放引用
            _inflight.clear()
        _COMM.close()
        _COMM = None


def _splits_bytes(t, splits, world):
    """dim0 行数 splits → 字节 counts/displs(None = 均匀)。"""
    rows = t.shape[0]
    if splits is None:
        assert rows % world == 0, f"rows={rows} 非均匀且未给 splits"
        splits = [rows // world] * world
    row_bytes = t[0].numel() * t.element_size() if rows else \
        t.numel() // max(rows, 1) * t.element_size()
    counts = [int(s) * row_bytes for s in splits]
    displs, off = [], 0
    for c in counts:
        displs.append(off)
        off += c
    return counts, displs


# ── 2026-08-15 优化批旋钮(与 libmosaic 侧同名 env;File/log_raven.md §6)
# O1 异步提交:默认开。提交时在 torch 当前流上录事件交给引擎(device 侧依赖),
# 不再 torch.cuda.synchronize() —— CPU 协议与调用方流上的尾部计算重叠。
_ASYNC = os.environ.get("MOSAIC_ASYNC_SUBMIT", "1") != "0"
# O2 demand hint:默认 auto。相邻两次调用的 token 矩阵结构相同(probs)或互为
# 转置(combine / autograd backward)时,引擎从上次全局矩阵恒等推导,免 allgather。
_HINT = int(os.environ.get("MOSAIC_DEMAND_HINT", "1") or "1")
_prev_call = {"sc": None, "rc": None, "rb": 0}
# S3 异步提交(MOSAIC_TORCH_ASYNC=1,默认关=旧阻塞语义):Begin 立即返回,
# 数据就绪由调用方流上的设备侧等待兜住;两侧张量引用持到下一次调用
# (引擎一次一笔,下一次 Begin 内部隐式收尾上一笔 ⇒ 届时引用可放)。
_ASYNC_A2A = os.environ.get("MOSAIC_TORCH_ASYNC", "0") == "1"
_inflight = []   # [(input, output)] 在途张量保活


def _row_bytes(t):
    rows = t.shape[0]
    if rows:
        return t[0].numel() * t.element_size()
    return t.numel() // max(rows, 1) * t.element_size()


def _hint_mode(sc, rc, rb):
    """与上一次调用的 token 矩阵比对:SAME(仅 row_bytes 不同,MoE probs)→1;
    TRANSPOSE(dispatch↔combine / forward↔backward)→2;其余→0。
    只看本 rank 的行/列 —— 引擎侧再以行+列校验兜底,失配 = E4005 硬错。"""
    p = _prev_call
    if not _HINT or rb <= 0 or p["rb"] <= 0 or p["sc"] is None:
        return 0
    pr = p["rb"]

    def toks(v, r):
        return None if any(x % r for x in v) else [x // r for x in v]

    a, b = toks(sc, rb), toks(rc, rb)
    pa, pb = toks(p["sc"], pr), toks(p["rc"], pr)
    if None in (a, b, pa, pb):
        return 0
    if a == pa and b == pb:
        return 1
    if a == pb and b == pa:
        return 2
    return 0


def all_to_all_single(output, input, output_split_sizes=None,
                      input_split_sizes=None, group=None, async_op=False):
    """签名对齐 torch.distributed.all_to_all_single(async_op 恒同步完成,
    返回 None —— MoE dispatcher 随即要用结果,语义等价)。"""
    if _BACKEND == "nccl":
        return dist.all_to_all_single(output, input, output_split_sizes,
                                      input_split_sizes, group=group)
    c = comm()
    assert input.is_contiguous() and output.is_contiguous(), "E4001 需要连续张量"
    world = c.n
    sc, sd = _splits_bytes(input, input_split_sizes, world)
    rc, rd = _splits_bytes(output, output_split_sizes, world)
    rb = _row_bytes(input)
    mode = _hint_mode(sc, rc, rb)
    _prev_call.update(sc=sc, rc=rc, rb=rb)
    stream = 0
    if input.is_cuda:
        if _ASYNC:
            stream = torch.cuda.current_stream().cuda_stream   # O1:事件交引擎
            # 默认流的句柄是 0,与 C API"未提供流"的哨兵撞了 —— 撞上时传
            # cudaStreamLegacy(0x1,CUDA 原生的默认流显式句柄)。不处理的话
            # 引擎既不录事件也无人同步,input 未就绪就被读走(实测:训练 loss
            # 发散而 drop-in 单测因自带 synchronize 侥幸全绿)。
            if stream == 0:
                stream = 1
        else:
            torch.cuda.synchronize()   # 旧语义(保守口径,见文件头)
    use_begin = _ASYNC_A2A and stream != 0
    c.alltoallv(input.data_ptr(), sc, sd, output.data_ptr(), rc, rd,
                stream=stream, demand_mode=mode, row_bytes=rb, begin=use_begin)
    if use_begin:
        # 上一笔已被引擎隐式收尾(一次一笔)⇒ 旧引用可放;换成本笔的
        _inflight.clear()
        _inflight.append((input, output))
    return None


class _MosaicAllToAll(torch.autograd.Function):
    """backward = splits 转置后的同一 alltoallv(Megatron _AllToAll 同语义)。"""

    @staticmethod
    def forward(ctx, input, out_splits, in_splits, out_rows):
        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        ctx.in_rows = input.shape[0]
        out = input.new_empty((out_rows,) + tuple(input.shape[1:]))
        all_to_all_single(out, input.contiguous(), out_splits, in_splits)
        return out

    @staticmethod
    def backward(ctx, grad):
        g = grad.contiguous()
        back = g.new_empty((ctx.in_rows,) + tuple(g.shape[1:]))
        all_to_all_single(back, g, ctx.in_splits, ctx.out_splits)
        return back, None, None, None


def all_to_all_autograd(input, out_splits=None, in_splits=None, out_rows=None):
    if out_rows is None:
        out_rows = int(sum(out_splits)) if out_splits is not None else input.shape[0]
    return _MosaicAllToAll.apply(input, out_splits, in_splits, out_rows)


def patch_megatron():
    """单点替换 Megatron-Core 的 a2a 原语(Imp_Plan/05 §2.1 方案 b)。
    nccl 后端 = 不打 patch(原生路径)。符号断言 fail-fast(E4002)。"""
    if _BACKEND == "nccl":
        return
    import megatron.core.tensor_parallel.mappings as mappings
    assert hasattr(mappings, "all_to_all"), \
        "E4002 megatron mappings.all_to_all 不存在(版本符号搬移,核对 pin 的 commit)"

    def mosaic_all_to_all(group, input_, output_split_sizes=None,
                          input_split_sizes=None):
        world = dist.get_world_size(group=group)
        if world == 1:
            return input_
        if output_split_sizes is None:
            out_rows = input_.shape[0]
        else:
            out_rows = int(sum(int(x) for x in output_split_sizes))
        return _MosaicAllToAll.apply(input_.contiguous(), output_split_sizes,
                                     input_split_sizes, out_rows)

    mappings.all_to_all = mosaic_all_to_all
    # dispatcher 可能 from-import 了符号,把已知引用点一并替换(fail-soft)
    try:
        import megatron.core.transformer.moe.token_dispatcher as td
        if hasattr(td, "all_to_all"):
            td.all_to_all = mosaic_all_to_all
    except Exception:
        pass
    return mosaic_all_to_all
