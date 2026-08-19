/* mosaic.h —— libmosaic 公开 C API。
 *
 * 语义 = MPI_Alltoallv / ncclGroupStart+Send/Recv 的非均匀 all-to-all,
 * 但内部执行的是 MOSAIC-U2b 调度(账本核 + 双通道广播 + 决策错峰,
 * 见 Moasic/Note/report_mosaic_eps.md §11)。
 *
 * 分层(Imp_Plan/01):本头文件 = L3;调用者不需要知道 corridor / wave /
 * ledger 的存在,也不需要知道底下是 SHM 还是 TCP 还是 NVSHMEM。
 */
#ifndef MOSAIC_H
#define MOSAIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mosaicComm* mosaicComm_t;

typedef enum {
  mosaicSuccess = 0,
  mosaicErrConfig = 1,      /* E4xxx: 配置/参数不合法 */
  mosaicErrTransport = 2,   /* E1xxx: 建链/收发失败 */
  mosaicErrProtocol = 3,    /* E2xxx: 协议状态机异常(卡死/账本不平) */
  mosaicErrVerify = 4,      /* E3xxx: 数据校验失败 */
  mosaicErrPlatform = 5,    /* E5xxx: CUDA/系统调用失败 */
  mosaicErrTimeout = 6,
} mosaicResult_t;

/* 调度器:同一套 transport 上可换的三种"发什么"的策略。
 * 公平性的技术落点(Imp_Plan/04 §3):三者共用完全相同的执行底座。 */
typedef enum {
  MOSAIC_SCHED_MOSAIC = 0,  /* MOSAIC-U2b:账本 + 走廊 + 波 + 双通道 */
  MOSAIC_SCHED_DIRECT = 1,  /* 每 pair 全量走 DIRECT,无波无账本 = static ECMP 位 */
  MOSAIC_SCHED_FAST   = 2,  /* FAST:官方 host 调度器(官方 fast_alltoall 源码
                               未改动地编译链接,Birkhoff/balance/restore 全在)
                               驱动的 adapter,见 src/fast_native.cc 头注释;
                               MOSAIC_FAST_NATIVE=0 退回 structural 近似(消融) */
  MOSAIC_SCHED_FAST_STRUCT = 3,  /* structural FAST:按论文结构重写的近似版,
                                    不是官方代码 —— 只作消融/对照位 */
} mosaicSched_t;

/* 运行时配置。全部字段都有环境变量镜像(MOSAIC_*),见 File/log_raven.md §5 旋钮表。
 * 改字段布局必须同步 python/mosaic_py.py 的 ctypes Config/Stats(逐字段对齐)。 */
typedef struct {
  int nranks;
  int rank;
  int gpus_per_host;        /* G:每台"模拟 server"上的 vGPU 数 */
  int corridors;            /* C,1..6(嵌套:DIRECT,RX,TX,SPREAD lane...) */
  int waves;                /* W */
  int sched;                /* mosaicSched_t */
  double nv_gbps;           /* 每 vGPU 的 NV 管额定(单向) */
  double nic_gbps;          /* 每 vGPU 的 vNIC 额定(单向) */
  double spine_gbps;        /* 跨 host 聚合(taper);<=0 表示不设限 */
  int slice_bytes;
  int sync_us;              /* 通道 A 遥测周期 */
  int stagger_us;           /* 决策错峰槽宽 */
  int stagger_slots;
  int ledger;               /* 1 = 账本核;0 = 退化(仅 prior 注水) */
  int bcast;                /* 1 = 通道 B 策略广播 */
  int pair_color;           /* PC 门:-1 = auto(diag CoV<0.1 → K=n,否则 0),
                               0 = off,K>0 = 固定 K(报告 §11.2 三重门) */
  int cold_pace;            /* PACE 门:k(冷 dst 最多领先 hot k-1 波);0 = off */
  int detector;             /* 1 = NIC 域异常检测器(0.45×/3 折/quorum≤2) */
  int spineagg;             /* 1 = fabric 账聚合档(纯 ECMP fabric 的兜底,
                               emu 与 Raven 都是无 spine 选择的 fabric → 恒 1;
                               旋钮保留是为与模拟器 flag 对表) */
  int verify;               /* 1 = 逐 slice CRC 校验 */
  int use_gpu;              /* 默认自动探测(有 CUDA 设备即 1);0 = 纯 host 内存跑
                               协议(离线决策回放用;有机内 peer 的形态不支持)。
                               2026-08-19 起无 env 镜像,只能程序内设。 */
  /* (原 nv_ipc 已删,2026-08-19:机内 payload 恒走 CUDA IPC 设备内存;
     POSIX 共享内存 payload 属模拟遗留路径。) */
} mosaicConfig_t;

/* 用默认值 + 环境变量填充 cfg;rank/nranks 也可由 MOSAIC_RANK/MOSAIC_NRANKS 给。 */
mosaicResult_t mosaicConfigDefault(mosaicConfig_t* cfg);

mosaicResult_t mosaicCommInit(mosaicComm_t* comm, const mosaicConfig_t* cfg);
mosaicResult_t mosaicCommDestroy(mosaicComm_t comm);

/* 阻塞式 alltoallv。counts/displs 单位 = 字节。
 * sendbuf/recvbuf:cfg.use_gpu 时为 device 指针,否则 host 指针。
 * 本调用内部含一次隐式的 demand 矩阵 allgather(协议需要全局矩阵)。 */
mosaicResult_t mosaicAlltoallv(mosaicComm_t comm,
                               const void* sendbuf, const size_t scounts[], const size_t sdispls[],
                               void* recvbuf, const size_t rcounts[], const size_t rdispls[]);

/* 扩展提交(2026-08-15 优化批;全部行为可用环境变量关回旧语义):
 *   caller_stream:调用方 CUDA 流句柄(cudaStream_t 转 uint64)。非 0 时引擎
 *     在该流上录一个"输入就绪"事件并让内部拷贝流 cudaStreamWaitEvent 它 ——
 *     调用方不必先 synchronize,CPU 协议(demand 交换/注水/通道 B)与调用方流上
 *     未完成的计算重叠(O1;MOSAIC_ASYNC_SUBMIT=0 时调用方应传 0 并自行同步)。
 *   demand_mode:0 = 正常(demand 矩阵 allgather);
 *     1 = SAME:全局矩阵 = 上次矩阵按 row_bytes 等比缩放(MoE probs a2a);
 *     2 = TRANSPOSE:上次矩阵转置后缩放(MoE combine / autograd backward)。
 *     模式 1/2 由调用方保证数学恒等(引擎校验本 rank 行+列,失配 = E4005 硬错,
 *     绝不带病继续);MOSAIC_DEMAND_HINT=0 全局禁用,=2 为 verify 档(照常
 *     allgather 并对拍推导矩阵,只验不省)。FAST 调度器需要全局一致矩阵,
 *     引擎自动把 1/2 降为 verify 档(它的官方实现本就要求全量交换)。
 *   row_bytes:本次每行字节数(demand_mode!=0 时用于缩放;其余传 0)。 */
mosaicResult_t mosaicAlltoallvEx(mosaicComm_t comm,
                                 const void* sendbuf, const size_t scounts[], const size_t sdispls[],
                                 void* recvbuf, const size_t rcounts[], const size_t rdispls[],
                                 uint64_t caller_stream, int demand_mode, uint64_t row_bytes);

/* (原 mosaicAlltoallvBegin/End 异步提交 API 已删,2026-08-19:每轮 A2A
 *  结束必有同步是硬不变量,异步提交语义连同实现一起移除 ——
 *  见 File/debug-819/local-0819-transport-cleanup-plan.md。) */

/* 小数据 allgather(host 内存,走控制面)。MoE 集成需要先交换各 rank 的
   token 计数才能定 buffer 大小 —— 这类元数据交换不该占数据管。
   recvbuf 需 nranks*bytes 大小,按 rank 顺序拼接。 */
mosaicResult_t mosaicAllgather(mosaicComm_t comm, const void* sendbuf, size_t bytes, void* recvbuf);

/* 全体同步(控制面实现,不走数据管);基准测试用来对齐起点。 */
mosaicResult_t mosaicBarrier(mosaicComm_t comm);

/* 上一次 collective 的统计。用于结果表与 06 的性能日志。 */
typedef struct {
  double completion_us;      /* 本 rank 从 start 到收满 */
  double makespan_us;        /* 全局 max(需 mosaicStatsSync 后才有效) */
  uint64_t bytes_sent, bytes_recv, bytes_relayed;
  uint64_t corridor_bytes[8];   // 出向:本 rank 实际发出的字节按走廊分
  uint64_t ingress_corr[8];     // 入向:本 rank 作为 dst 控制环给自己安排的进流分配
  uint64_t chA_msgs, chA_bytes, chB_msgs, chB_bytes;
  double plan_us_total; uint64_t plan_calls; double plan_us_max;
  double stagger_us_total;
  uint64_t waves_emitted;
  uint64_t detector_events;
  uint64_t res_passed[5];       /* 各资源实际通过字节:NICup,NICdn,NVup,NVdn,SPINEup */
  double sched_plan_us;         /* 静态调度器(FAST native)一次 plan 的耗时;
                                   报表报「含/不含 plan」两口径(Imp_Plan/07 §2.2) */
  double plan_us_p50, plan_us_p99;   /* 注水耗时分位(03 §7 效率红线) */
  double tick_jitter_p99_us;         /* 通道 A 100µs 节拍的实际抖动 p99 */
  uint64_t chA_drops;                /* 注入丢弃的通道 A 报文数(MOSAIC_DROP_A) */
  double det_rate_gbps[2];           /* 检测器当前生效速率(NIC↑,NIC↓);=额定表示未锁定 */
} mosaicStats_t;

mosaicResult_t mosaicGetStats(mosaicComm_t comm, mosaicStats_t* out);
/* 把各 rank 的 completion_us 归约成 makespan(控制面 allgather)。 */
mosaicResult_t mosaicStatsSync(mosaicComm_t comm, mosaicStats_t* out);

const char* mosaicGetLastError(mosaicComm_t comm);
const char* mosaicSchedName(int sched);

#ifdef __cplusplus
}
#endif
#endif /* MOSAIC_H */
