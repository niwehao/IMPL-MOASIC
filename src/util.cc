/* util.cc —— 时钟、日志、CRC、trace ring、配置(环境变量)。 */
#include "internal.h"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

namespace mosaic {

int g_log_level = 1;
int g_rank_for_log = -1;

// ── 二进制 trace ring(06 §5.2)。单写者(event 线程),mmap 即持久。
bool g_trace_on = false;
bool g_trace_loop = false;
static TraceRec* g_trace = nullptr;
static uint64_t* g_trace_cur = nullptr;    // 文件头 8B = 写游标
static uint32_t g_trace_cap = 0;

void mtrace_init(int rank) {
  const char* e = getenv("MOSAIC_TRACE");
  if (!e || atoi(e) == 0) return;
  g_trace_loop = atoi(e) >= 2;
  const char* dir = getenv("MOSAIC_TRACE_DIR");
  char path[512];
  snprintf(path, sizeof(path), "%s/mosaic_rank%d.trace", dir && *dir ? dir : "/tmp", rank);
  const uint32_t cap = getenv("MOSAIC_TRACE_CAP") ? atoi(getenv("MOSAIC_TRACE_CAP")) : (1u << 20);
  size_t bytes = 64 + (size_t)cap * sizeof(TraceRec);
  int fd = open(path, O_CREAT | O_RDWR | O_TRUNC, 0644);
  if (fd < 0 || ftruncate(fd, bytes) != 0) { MLOG_WARN("trace 文件失败:%s", path); return; }
  void* base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) { MLOG_WARN("trace mmap 失败"); return; }
  g_trace_cur = (uint64_t*)base;
  *g_trace_cur = 0;
  ((uint32_t*)base)[2] = cap;              // 头里带容量,trace2csv 用
  g_trace = (TraceRec*)((uint8_t*)base + 64);
  g_trace_cap = cap;
  g_trace_on = true;
  MLOG_INFO("trace ring 启用:%s(%u 条 × 32B)", path, cap);
}

void mtrace(uint16_t ev, uint32_t a, uint64_t b, uint64_t c) {
  if (!g_trace) return;
  uint64_t i = (*g_trace_cur)++;
  TraceRec& r = g_trace[i % g_trace_cap];
  r.t_ns = now_ns(); r.ev = ev; r.rank = (uint16_t)g_rank_for_log;
  r.a = a; r.b = b; r.c = c;
}

uint64_t now_ns() {
  // CLOCK_MONOTONIC_RAW:不受 NTP 调整影响,100µs 节拍与陈旧度测量都靠它
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void mlog(int level, const char* fmt, ...) {
  if (level > g_log_level) return;
  static const char* tag[] = {"ERR ", "WARN", "INFO", "DBG "};
  static uint64_t t0 = now_ns();
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fprintf(stderr, "[mosaic r%-2d %8.3fms %s] %s\n", g_rank_for_log,
          (now_ns() - t0) / 1e6, tag[level < 4 ? level : 3], buf);
  fflush(stderr);
}

// CRC 用 SSE4.2 硬件 crc32c(x86 上 ~20GB/s)。原先的字节表版只有 ~400MB/s,
// 而 CRC 在**单事件线程的热路径**上逐 slice 算 —— verify=1 时把 RDMA 数据面
// 从 88Gbps 拖到 1.8Gbps(48×!),而且旧 TCP 结果在 NIC 侧也被它悄悄压过
// (见 log.md 坑 #14)。校验语义不变:两端用同一函数,只比对一致性。
uint32_t crc32_buf(const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  uint64_t c = 0xFFFFFFFFu;
  size_t i = 0;
#if defined(__SSE4_2__)
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    memcpy(&v, b + i, 8);
    c = __builtin_ia32_crc32di(c, v);
  }
#endif
  for (; i < n; i++)
#if defined(__SSE4_2__)
    c = __builtin_ia32_crc32qi((uint32_t)c, b[i]);
#else
  {  // 无 SSE4.2 的兜底(慢,但仅影响 verify 模式)
    uint32_t x = ((uint32_t)c ^ b[i]) & 0xFF;
    for (int k = 0; k < 8; k++) x = (x & 1) ? 0x82F63B78u ^ (x >> 1) : x >> 1;
    c = ((uint32_t)c >> 8) ^ x;
  }
#endif
  return (uint32_t)c ^ 0xFFFFFFFFu;
}

static int envi(const char* k, int d) {
  const char* v = getenv(k);
  return v && *v ? atoi(v) : d;
}
static double envd(const char* k, double d) {
  const char* v = getenv(k);
  return v && *v ? atof(v) : d;
}

}  // namespace mosaic

using namespace mosaic;

extern "C" mosaicResult_t mosaicConfigDefault(mosaicConfig_t* cfg) {
  if (!cfg) return mosaicErrConfig;
  memset(cfg, 0, sizeof(*cfg));
  cfg->nranks = envi("MOSAIC_NRANKS", 8);
  cfg->rank = envi("MOSAIC_RANK", 0);
  cfg->gpus_per_host = envi("MOSAIC_G", 4);
  cfg->corridors = envi("MOSAIC_CORRIDORS", 6);
  cfg->waves = envi("MOSAIC_WAVES", 6);
  cfg->sched = envi("MOSAIC_SCHED", MOSAIC_SCHED_MOSAIC);
  // 默认额定值:本机模拟的基准档。NV:NIC = 13.3:1(对应报告主线的强 scale-up 端),
  // 数值取小是为了让 PCIe(Gen4 x16 ≈ 22GB/s 单向)不会先于模拟带宽成为瓶颈
  // —— 标定与取值依据见 log.md「为什么 vNIC 默认 2.5Gbps」。
  // 忠实模拟包线(scripts/calibrate 实测):NV 管(IPC+1MB)≈41.7GB/s 聚合、
  // vNIC(TCP)≈9.2Gbps/rank。额定值取到上限的 1/3~1/2,保证**限速器是绑定约束**。
  cfg->nic_gbps = envd("MOSAIC_NIC_GBPS", 2.0);
  cfg->nv_gbps = envd("MOSAIC_NV_GBPS", 8.0);
  cfg->spine_gbps = envd("MOSAIC_SPINE_GBPS", 0);
  // 1MB 与模拟器同口径。实测:slice 越大每字节的 CUDA API 开销越低 ——
  // 64KB 时 NV 管上限 12 GB/s(API 绑定),1MB 时 41.7 GB/s(接近 HBM 侧真实带宽)。
  // 规划粒度已与 slice 解耦(见 emit_wave 的 GRAN_FLOOR),所以放大 slice 不损失分流精度。
  cfg->slice_bytes = envi("MOSAIC_SLICE_BYTES", 1048576);
  cfg->sync_us = envi("MOSAIC_SYNC_US", 100);
  cfg->stagger_us = envi("MOSAIC_STAGGER_US", 3);
  cfg->stagger_slots = envi("MOSAIC_STAGGER_SLOTS", 8);
  cfg->ledger = envi("MOSAIC_LEDGER", 1);
  cfg->bcast = envi("MOSAIC_BCAST", 1);
  // 报告 §11.2 复现配置:-mosaic_pair_color_auto -mosaic_cold_pace 1 + 检测器
  {  // MOSAIC_PAIR_COLOR = auto | 0 | K
    const char* v = getenv("MOSAIC_PAIR_COLOR");
    cfg->pair_color = (!v || !*v || !strcmp(v, "auto")) ? -1 : atoi(v);
  }
  cfg->cold_pace = envi("MOSAIC_COLD_PACE", 1);
  cfg->detector = envi("MOSAIC_DETECTOR", 1);
  cfg->spineagg = envi("MOSAIC_SPINEAGG", 1);
  cfg->verify = envi("MOSAIC_VERIFY", 0);
  // use_gpu 自动探测(2026-08-19:MOSAIC_USE_GPU env 删除)。无 CUDA 设备
  // (Raven 登录节点跑离线回放)自动走纯 host;测试可程序内显式置 0。
  {
    int ndev = 0;
    cfg->use_gpu = (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0) ? 1 : 0;
  }
  g_log_level = envi("MOSAIC_LOG", 1);
  return mosaicSuccess;
}

extern "C" const char* mosaicSchedName(int s) {
  switch (s) {
    case MOSAIC_SCHED_MOSAIC: return "mosaic";
    case MOSAIC_SCHED_DIRECT: return "direct";
    case MOSAIC_SCHED_FAST: return "fast";
    case MOSAIC_SCHED_FAST_STRUCT: return "fast_struct";
    default: return "?";
  }
}
