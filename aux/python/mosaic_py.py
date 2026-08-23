"""mosaic_py —— libmosaic 的 Python 绑定(ctypes)。

为什么用 ctypes 而不是 pybind11/torch.utils.cpp_extension:
  库的公开 API 是纯 C ABI、且数据面只收**指针 + 字节数**(torch 张量的
  `data_ptr()` 正好给这个),不需要任何 C++ 类型转换。用 ctypes 就完全绕开了
  本机编 CUDA 扩展的两个坑(glibc 2.41 与 CUDA 12.8 头文件的 cospi 冲突、
  torch 的 -D__CUDA_NO_HALF_OPERATORS__),而且换 torch 版本不用重编。

用法:
    import mosaic_py as mp
    comm = mp.Comm(rank=0, nranks=8, gpus_per_host=4,
                   hosts="192.168.100.2,192.168.100.1", sched="mosaic")
    comm.alltoallv_t(send_t, scounts_rows, recv_t, rcounts_rows, row_bytes)
"""
import ctypes
import os
from pathlib import Path

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        p = os.environ.get("MOSAIC_LIB") or str(Path(__file__).resolve().parents[1] / "build" / "libmosaic.so")
        _LIB = ctypes.CDLL(p)
        _LIB.mosaicConfigDefault.argtypes = [ctypes.c_void_p]
        _LIB.mosaicCommInit.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.mosaicCommDestroy.argtypes = [ctypes.c_void_p]
        _LIB.mosaicAlltoallv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
        # 扩展提交:caller_stream(O1 异步提交)、demand_mode/row_bytes(O2 hint)、
        # phase(任务二相位:0 未知 / 1 dispatch / 2 combine,见 mosaic.h)
        _LIB.mosaicAlltoallvEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
                                           ctypes.c_uint64, ctypes.c_int]
        _LIB.mosaicBarrier.argtypes = [ctypes.c_void_p]
        _LIB.mosaicAllgather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _LIB.mosaicGetStats.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.mosaicStatsSync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _LIB.mosaicGetLastError.argtypes = [ctypes.c_void_p]
        _LIB.mosaicGetLastError.restype = ctypes.c_char_p
    return _LIB


class Config(ctypes.Structure):
    """必须与 include/mosaic/mosaic.h 的 mosaicConfig_t 逐字段对齐。
    2026-08-14:补 pair_color/cold_pace/detector/spineagg(T3),错位 = 静默
    内存踩踏,改 mosaic.h 必须同步改这里(两处都有指回对方的注释)。"""
    _fields_ = [
        ("nranks", ctypes.c_int), ("rank", ctypes.c_int), ("gpus_per_host", ctypes.c_int),
        ("corridors", ctypes.c_int), ("waves", ctypes.c_int), ("sched", ctypes.c_int),
        ("nv_gbps", ctypes.c_double), ("nic_gbps", ctypes.c_double), ("spine_gbps", ctypes.c_double),
        ("slice_bytes", ctypes.c_int), ("sync_us", ctypes.c_int), ("stagger_us", ctypes.c_int),
        ("stagger_slots", ctypes.c_int), ("ledger", ctypes.c_int), ("bcast", ctypes.c_int),
        ("pair_color", ctypes.c_int), ("cold_pace", ctypes.c_int),
        ("detector", ctypes.c_int), ("spineagg", ctypes.c_int),
        ("use_gpu", ctypes.c_int),
    ]


class Stats(ctypes.Structure):
    _fields_ = [
        ("completion_us", ctypes.c_double), ("makespan_us", ctypes.c_double),
        ("bytes_sent", ctypes.c_uint64), ("bytes_recv", ctypes.c_uint64), ("bytes_relayed", ctypes.c_uint64),
        ("corridor_bytes", ctypes.c_uint64 * 8), ("ingress_corr", ctypes.c_uint64 * 8),
        ("chA_msgs", ctypes.c_uint64), ("chA_bytes", ctypes.c_uint64),
        ("chB_msgs", ctypes.c_uint64), ("chB_bytes", ctypes.c_uint64),
        ("plan_us_total", ctypes.c_double), ("plan_calls", ctypes.c_uint64), ("plan_us_max", ctypes.c_double),
        ("stagger_us_total", ctypes.c_double), ("waves_emitted", ctypes.c_uint64),
        ("detector_events", ctypes.c_uint64), ("res_passed", ctypes.c_uint64 * 5),
        ("sched_plan_us", ctypes.c_double),
        ("plan_us_p50", ctypes.c_double), ("plan_us_p99", ctypes.c_double),
        ("tick_jitter_p99_us", ctypes.c_double),
        ("chA_drops", ctypes.c_uint64), ("det_rate_gbps", ctypes.c_double * 2),
    ]


SCHED = {"mosaic": 0, "direct": 1, "fast": 2, "fast_struct": 3}


class Comm:
    def __init__(self, rank, nranks, gpus_per_host, hosts=None, sched="mosaic",
                 job=None, **kw):
        # 这些必须在 CommInit 之前进环境,库侧从 env 读默认值
        if hosts:
            os.environ["MOSAIC_HOSTS"] = hosts
        if job is not None:
            os.environ["MOSAIC_JOB"] = str(job)
        cfg = Config()
        _lib().mosaicConfigDefault(ctypes.byref(cfg))
        cfg.rank, cfg.nranks, cfg.gpus_per_host = rank, nranks, gpus_per_host
        cfg.sched = SCHED[sched]
        for k, v in kw.items():
            setattr(cfg, k, v)
        self.cfg = cfg
        self.n = nranks
        self.rank = rank
        self._h = ctypes.c_void_p()
        rc = _lib().mosaicCommInit(ctypes.byref(self._h), ctypes.byref(cfg))
        if rc != 0:
            raise RuntimeError(f"mosaicCommInit 失败 rc={rc}")

    # ---- 低层:按字节 ----
    # stream:调用方 CUDA 流句柄(O1 异步提交;0 = 旧语义,调用方自行同步)。
    # demand_mode/row_bytes:O2 demand hint(0 = 正常 allgather;1 = SAME 缩放;
    # 2 = TRANSPOSE 缩放;见 mosaic.h mosaicAlltoallvEx 注释)。
    def alltoallv(self, sendptr, scounts, sdispls, recvptr, rcounts, rdispls,
                  stream=0, demand_mode=0, row_bytes=0, phase=0):
        """阻塞语义:返回 = 输出已就绪(每轮 A2A 结束必有同步,硬不变量)。"""
        A = ctypes.c_size_t * self.n
        rc = _lib().mosaicAlltoallvEx(self._h, ctypes.c_void_p(sendptr),
               A(*scounts), A(*sdispls),
               ctypes.c_void_p(recvptr), A(*rcounts), A(*rdispls),
               ctypes.c_uint64(stream), ctypes.c_int(demand_mode),
               ctypes.c_uint64(row_bytes), ctypes.c_int(phase))
        if rc != 0:
            raise RuntimeError(f"alltoallv 失败 rc={rc}: {_lib().mosaicGetLastError(self._h).decode()}")

    # ---- 便利层:按"行"(MoE 的 token 数)----
    def alltoallv_rows(self, send_t, send_rows, recv_t, recv_rows, row_bytes,
                       phase=0):
        """send_t/recv_t 是行主序的 2D 张量(或等价的连续缓冲)。
        send_rows[j] = 发给 rank j 的行数;recv_rows[i] = 从 rank i 收的行数。
        phase:1 = dispatch(发送端均匀)/ 2 = combine(接收端均匀),
        调用方按调用位置填(mosaic.h 契约)。"""
        sc = [r * row_bytes for r in send_rows]
        rc = [r * row_bytes for r in recv_rows]
        sd, rd, a, b = [], [], 0, 0
        for x in sc:
            sd.append(a); a += x
        for x in rc:
            rd.append(b); b += x
        self.alltoallv(send_t.data_ptr(), sc, sd, recv_t.data_ptr(), rc, rd,
                       phase=phase)

    def allgather_i64(self, vals):
        """每 rank 贡献一个 int64 向量,返回 [nranks][len] 的列表(控制面小消息)。"""
        import ctypes as C
        k = len(vals)
        send = (C.c_int64 * k)(*vals)
        recv = (C.c_int64 * (k * self.n))()
        rc = _lib().mosaicAllgather(self._h, send, C.c_size_t(8 * k), recv)
        if rc != 0:
            raise RuntimeError(f"allgather 失败 rc={rc}")
        return [[recv[i * k + j] for j in range(k)] for i in range(self.n)]

    def barrier(self):
        _lib().mosaicBarrier(self._h)

    def stats(self, sync=False):
        st = Stats()
        f = _lib().mosaicStatsSync if sync else _lib().mosaicGetStats
        f(self._h, ctypes.byref(st))
        return st

    def close(self):
        if self._h:
            _lib().mosaicCommDestroy(self._h)
            self._h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
