/* engine.cc —— L1 执行引擎 + L2 协议核。
 *
 * 单线程事件循环(event 线程)负责全部:CUDA 异步拷贝的提交与轮询、网络轮询、
 * IPC 环轮询、账本注水、波装配、通道 A/B。单线程 ⇒ 无锁 ⇒
 * 决策序列可复现(这是能对拍、能离线回放的前提)。
 *
 * slice 状态机(2026-08-19 GDR 重构:payload 全程在设备内存):
 *   READY --(D2D 到 GPU staging)--> WIRE --(IPC 环 publish / verbs SEND(GDR)
 *                    verbs SEND)--> 对端
 *   对端 --(收齐)--> (终跳) D2D 到 recvbuf → 计交付
 *                  \(中转) D2D 到 relay 池 → 变回 READY 发下一跳
 * hop k+1 只在 hop k 的字节**完整落到中转 GPU 之后**才发(chunk availability)。
 */
#include "internal.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

// P2 线格式前提:60B 前缀 = TelemMsg 去掉 fabric 数组(改 TelemMsg 布局必须同步)
static_assert(offsetof(mosaic::TelemMsg, fab_up) == mosaic::TELEM_NOFAB_BYTES,
              "TELEM_NOFAB_BYTES 必须等于 TelemMsg 的 fab_up 偏移");

namespace mosaic {

// ─────────────────────────────────────────────────────────── CUDA 薄封装
#define CUCHK(call, what)                                                    \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) { MLOG_ERR("E5001 %s: %s", what, cudaGetErrorString(_e)); }        \
  } while (0)

// ── 2026-08-15 优化批旋钮(一次解析;语义与默认值见 File/log_raven.md §6)
static int env_flag(const char* name, int dflt) {
  const char* e = getenv(name);
  return e && *e ? atoi(e) : dflt;
}
// O4 波折叠:所有 (src,me) pair 走廊数 ≤1(无任何路径选择)时单波发完。
// 分配输出与逐波恒等(唯一走廊,注水无自由度),省掉 W-1 轮触发/错峰/广播延迟。
// 对拍/归因跑设 0(模拟器语义 = 固定 W 波)。
static int opt_wave_collapse() { static int v = env_flag("MOSAIC_WAVE_COLLAPSE", 1); return v; }
// O2 demand hint:0 = off;1 = trust(命中即免 allgather 等待);2 = verify
// (照常 allgather,推导矩阵只对拍不省 —— bring-up/Raven 阶梯用)
static int opt_demand_hint() { static int v = env_flag("MOSAIC_DEMAND_HINT", 1); return v; }
// O0 相位日志:每 collective 一行 demand/首发/总时长分解
static int opt_phase_log() { static int v = env_flag("MOSAIC_PHASE", 0); return v; }
// P1 SSP demand(算法级,用户授权):dispatch 不等 allgather —— 注水只读本地列
// (rcounts 调用方就有),PACE/PC 用上次矩阵(有界陈旧,与通道 A 同哲学),
// demand 行降级为后台广播。仅 mosaic 调度器;首个 collective 回退等待。
static int opt_demand_ssp() { static int v = env_flag("MOSAIC_DEMAND_SSP", 1); return v; }
// P2 定向遥测:通道 A 按 dst 在 CT_DEMAND 里捎带的读集位图裁剪(意图未到前
// 全量发,det 位对全员恒发 —— quorum 语义)。稳态下决策不变;collective 边界
// 新增字段有 ≤1 拍额外陈旧(deviations D17)。
static int opt_telem_targeted() { static int v = env_flag("MOSAIC_TELEM_TARGETED", 1); return v; }
// S2 自动选档(sota_plan;默认关):coll_begin 按 demand 形态选本次 W/C
static int opt_auto_tier() { static int v = env_flag("MOSAIC_AUTO_TIER", 0); return v; }
// 任务一 全局账本视野(File/design-global-view-src-driven-0823.md §1):
// 1(默认)= 通道 B 的完整分配广播给**所有** rank(中转节点也能把"被借了
// 多少"记进账本,分布式全局视野在协议层成立);0 = 旧行为(只发给本波的
// src,账本对中转占用残缺 —— A/B 退路)。开销差见 chB_bytes 计量。
static int opt_policy_gbcast() { static int v = env_flag("MOSAIC_POLICY_GBCAST", 1); return v; }
// 任务二 按阶段反转决策者:1(默认)= dispatch 相(coll_phase=1)由 src 驱动
// 发波(决策权给"总量相等"的那一侧);0 = 全部 collective 保持 dst 驱动
//(A/B 退路)。combine/未知相恒 dst 驱动,不受本旋钮影响。
static int opt_src_driven() { static int v = env_flag("MOSAIC_SRC_DRIVEN", 1); return v; }
// 波推进阈值(dst 环 = 上一波已交付比例;src 环 = 上一波跨机字节已 ACK 比例)。
// 默认 0.75 = 模拟器 _threshold 的既有标定(报告 §11.2);调大 = 波间流水更浅
// 但重规划用的账本更新;调小 = 更激进的流水但下一波决策基于更早的态。
static double wave_thresh() {
  static double v = -1;
  if (v < 0) {
    const char* e = getenv("MOSAIC_WAVE_THRESH");
    v = e && *e ? atof(e) : 0.75;
    if (v <= 0 || v > 1) v = 0.75;
  }
  return v;
}
// 目标碎片大小(KB):>0 时按"每对字节 ÷ (目标碎片 × 走廊数)"反推波数。
// 0 = 关(旧行为)。动机与实测见 coll_begin 里的注释。
static int opt_wave_frag_kb() { static int v = env_flag("MOSAIC_WAVE_FRAG_KB", 0); return v; }
static inline int eff_W(Comm* c) { return c->eff_waves > 0 ? c->eff_waves : c->cfg.waves; }
static inline int eff_C(Comm* c) { return c->eff_corridors > 0 ? c->eff_corridors : c->cfg.corridors; }

// 一律用 cudaMemcpyDefault:UVA 下 CUDA 自己按指针判方向,调用方不必记
// "这个 payload 到底在 host 还是 device"(IPC 模式下 NV 管的 payload 在设备内存)。
static void copy_async(Comm* c, void* dst, const void* src, size_t n, cudaMemcpyKind, void* st) {
  if (!c->cfg.use_gpu) { memcpy(dst, src, n); return; }
  CUCHK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, (cudaStream_t)st), "memcpyAsync");
}
static void* ev_take(Comm* c) {
  if (!c->cfg.use_gpu) return nullptr;
  if (c->ev_pool.empty()) {
    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    return (void*)e;
  }
  void* e = c->ev_pool.back(); c->ev_pool.pop_back(); return e;
}
static void ev_record(Comm* c, void* e, void* st) {
  if (!c->cfg.use_gpu) return;
  CUCHK(cudaEventRecord((cudaEvent_t)e, (cudaStream_t)st), "eventRecord");
}
static bool ev_ready(Comm* c, void* e) {
  if (!c->cfg.use_gpu) return true;
  return cudaEventQuery((cudaEvent_t)e) == cudaSuccess;
}
static void ev_give(Comm* c, void* e) { if (c->cfg.use_gpu && e) c->ev_pool.push_back(e); }

// ───────────────────────────────────── CUDA driver API 薄封装(PTX JIT 加载)
// 用 cuModuleLoadDataEx JIT 加载嵌入 PTX,免 nvcc(本机 glibc/CUDA 头冲突,
// Makefile 特意无 nvcc);libcuda 经 dlopen,CPU-only 构建不受影响。
typedef int (*cu_mload_fn)(void** mod, const void* image, unsigned int n,
                           void* opts, void** vals);
typedef int (*cu_mfunc_fn)(void** fn, void* mod, const char* name);
typedef int (*cu_launch_fn)(void* f, unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz,
                            unsigned shmem, void* stream, void** params, void** extra);
typedef int (*cu_addrrange_fn)(void** base, size_t* size, unsigned long long ptr);
struct CuDrv {
  cu_mload_fn mload = nullptr;
  cu_mfunc_fn mfunc = nullptr;
  cu_launch_fn launch = nullptr;
  cu_addrrange_fn addrrange = nullptr;   // cuMemGetAddressRange(dev-pipe2 求分配基址)
  bool ok = false;
};
static CuDrv& cu_drv() {
  static CuDrv d = []() {
    CuDrv r;
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (!h) return r;
    r.mload = (cu_mload_fn)dlsym(h, "cuModuleLoadDataEx");
    r.mfunc = (cu_mfunc_fn)dlsym(h, "cuModuleGetFunction");
    r.launch = (cu_launch_fn)dlsym(h, "cuLaunchKernel");
    r.addrrange = (cu_addrrange_fn)dlsym(h, "cuMemGetAddressRange_v2");
    if (!r.addrrange) r.addrrange = (cu_addrrange_fn)dlsym(h, "cuMemGetAddressRange");
    r.ok = r.mload && r.mfunc && r.launch;
    return r;
  }();
  return d;
}

// ─────────────── 零流数据面:GPU 消费者内核(流序窗口的搬运工)。
// 交付/staging 拷贝由消费者核自己做,host 在完成路径上只写 pinned 内存
// (推描述符 / 写 stop)。契约与内核见 consumer_kernel.cu。
#include "consumer_kernel_ptx.h"
// 消费者核并行度:blocks = nch × bpg(consumer_kernel.cu 的 channel 模型)。
// MOSAIC_KCONS="blocks,nch,threads":容量参数(不改数据面走法)。
// 默认 4,4,512 = 4×4 Raven MoE e2e 标定(log_raven.md 2026-08-21 R2 消融:
//   4 blk iter 198.4ms < 6 blk 206.0 < 8 blk 218.0 << 16 blk 333.3)——
//   更多 block 提纯 A2A 但训练重叠期抢 SM(16 块时 bwd +80%),勿盲目加大;
//   更少(2,2)拷贝宽度不足,fwd a2a 反升。调参必须跑 MoE e2e,不许用
//   纯 A2A 定档(CLAUDE.md 补充 H)。
static void kcons_cfg(unsigned* b, unsigned* n, unsigned* t) {
  static unsigned vb = 0, vn = 0, vt = 0;
  if (!vb) {
    vb = 4; vn = 4; vt = 512;
    const char* e = getenv("MOSAIC_KCONS");
    if (e && *e) sscanf(e, "%u,%u,%u", &vb, &vn, &vt);
    if (vt < 32) vt = 32;
    if (vt > 1024) vt = 1024;
    if (vn < 1) vn = 1;
    if (vn > vb) vn = vb;   // blocks<channels ⇒ 每 channel 0 block = 非法访存(29425220-002012)
  }
  *b = vb; *n = vn; *t = vt;
}
static unsigned kcons_blocks() { unsigned b,n,t; kcons_cfg(&b,&n,&t); return b; }
static unsigned kcons_nch()    { unsigned b,n,t; kcons_cfg(&b,&n,&t); return n; }
static unsigned kcons_threads(){ unsigned b,n,t; kcons_cfg(&b,&n,&t); return t; }
static bool kcons_init(Comm* c) {
  CuDrv& d = cu_drv();
  if (!d.ok) return false;
  void* mod = nullptr;
  if (d.mload(&mod, kConsumerPtx, 0, nullptr, nullptr) != 0) return false;
  void* fn = nullptr;
  if (d.mfunc(&fn, mod, "mosaic_consume") != 0) return false;
  void* ff = nullptr;
  if (d.mfunc(&ff, mod, "mosaic_flagset") != 0) return false;
  c->kcons_mod = mod;
  c->kcons_fn = fn;
  c->kflag_fn = ff;
  return true;
}
// S3 二期:input 就绪标志 —— caller 流上排 flagset 核(input 事件点之后),
// 流到此 = 输入张量已产出;engine 纯读 pinned 标志(kdesc_ctr+160)判 hop0
// 可发。异步窗口 event 线程一次 CUDA 查询都不能有(app 在 torch sync 里持锁
// 时查询冻死,29349040-112136 实测)。
bool async_mark_input(Comm* c, void* stream) {
  if (!c->kdesc_ctr || !stream || !c->kflag_fn) return false;
  __atomic_store_n((uint64_t*)(c->kdesc_ctr + 160), 0, __ATOMIC_RELEASE);
  // flagset 小核排在 caller 流上(输入事件点之后):设备直写 pinned 标志,
  // 免 cudaLaunchHostFunc 的回调派发延迟(实测 10-100µs/次,24 次/iter 就是
  // 0.5-2ms)。流到此 = 输入张量已产出且 caller 流先前工作全部完成。
  unsigned long long addr = c->kdesc_ctr_dev + 160, val = 1;
  void* params[2] = {&addr, &val};
  return cu_drv().launch(c->kflag_fn, 1, 1, 1, 1, 1, 1, 0, stream,
                         params, nullptr) == 0;
}
// ── 流序完成的门核(2026-08-19):caller 流上的有界自旋核,等 done_flag ≥
// 本 coll id 即退。**有界**(每 coll 一个,引擎完成必写 done_flag)⇒ 不踩
// 坑 #36 的常驻核死锁;门未决窗口内引擎零 CUDA 调用 ⇒ 不踩 gate_probe P2
// 的 cudaFree 楔死。sm_70+ 通用。
static const char* kGatePtx =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .entry mosaic_gate(.param .u64 p_addr, .param .u32 p_want){\n"
    " .reg .pred %p1;\n .reg .b32 %r1,%r2;\n .reg .b64 %rd1;\n"
    " ld.param.u64 %rd1,[p_addr];\n ld.param.u32 %r2,[p_want];\n"
    "$L_loop:\n ld.volatile.global.u32 %r1,[%rd1];\n"
    " setp.lt.u32 %p1,%r1,%r2;\n @%p1 nanosleep.u32 500;\n @%p1 bra $L_loop;\n ret;\n}\n";
static bool gate_init(Comm* c) {
  CuDrv& d = cu_drv();
  if (!d.ok) return false;
  void* mod = nullptr;
  if (d.mload(&mod, kGatePtx, 0, nullptr, nullptr) != 0) return false;
  void* fn = nullptr;
  if (d.mfunc(&fn, mod, "mosaic_gate") != 0) return false;
  c->gate_mod = mod;
  c->gate_fn = fn;
  return true;
}
// caller 流上排门(app 线程,提交路径内)。false = 排失败(退回阻塞语义)。
bool gate_arm(Comm* c, void* stream, uint32_t want) {
  if (!c->gate_fn || !c->done_flag_dev) return false;
  unsigned long long addr = (unsigned long long)c->done_flag_dev;
  unsigned int w = want;
  void* gp[2] = {&addr, &w};
  return cu_drv().launch(c->gate_fn, 1, 1, 1, 1, 1, 1, 0, stream, gp, nullptr) == 0;
}

// dev-pipe2(app 线程,提交路径 = 门外唯一安全的 CUDA 调用点):
// ①为本 coll 的 recvbuf 求分配基址并导出 IPC 句柄(接收方窗口);
// ②补 open 发送侧缓存 miss 的句柄(engine 在窗口内只查表,不 open)。
void devpipe_submit_prep(Comm* c, bool want_rwin) {
  {
    std::vector<std::array<uint8_t, 64>> todo;
    {
      std::lock_guard<std::mutex> lk(c->ipc_mu);
      todo.swap(c->ipc_pending);
    }
    for (auto& h : todo) {
      void* base = nullptr;
      cudaIpcMemHandle_t ih;
      memcpy(&ih, h.data(), sizeof(ih));
      if (cudaIpcOpenMemHandle(&base, ih, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess) {
        std::lock_guard<std::mutex> lk(c->ipc_mu);
        c->ipc_map[h] = base;
      } else {
        MLOG_WARN("E5032 IPC open 失败(直写降级为环路径)");
        std::lock_guard<std::mutex> lk(c->ipc_mu);
        c->ipc_map[h] = nullptr;              // 记失败,避免反复排队
      }
    }
  }
  c->my_rwin_valid = false;
  if (!want_rwin || !c->recvbuf) return;
  void* base = nullptr;
  size_t sz = 0;
  if (cu_drv().addrrange &&
      cu_drv().addrrange(&base, &sz, (unsigned long long)(uintptr_t)c->recvbuf) == 0 && base) {
    cudaIpcMemHandle_t ih;
    if (cudaIpcGetMemHandle(&ih, base) == cudaSuccess) {
      memcpy(c->my_rwin_handle, &ih, sizeof(ih));
      c->my_rwin_base_off = (uint64_t)((uint8_t*)c->recvbuf - (uint8_t*)base);
      c->my_rwin_valid = true;
    }
  }
}

// 供 api.cc 调(kdesc 数据面挂载):启动消费者核。核退出 = coll 收尾写 stop
// (kdesc_release_gate)。完成语义:流序档由 caller 流上的门核提供,阻塞档由
// wait_coll 提供。
bool zerostream_arm(Comm* c) {
  if (!c->zs_cap || !c->stream_cons || !c->kcons_fn) return false;
  CuDrv& d = cu_drv();
  // **每 collective 起核,coll 收尾即退**(坑 #36:常驻核 × torch 的
  // cudaFree 设备级隐式同步 = 永久死锁,test_persist_sync 实测)。换代竞态
  // (坑 #32)由代际号根治:先 gen++ 杀掉任何残留老核,再复位 stop,再启动
  // 带新 gen 的新核 —— 老核见 gen 变即退,且此刻新描述符尚未推入。
  c->kdesc_gen++;
  __atomic_store_n((uint64_t*)(c->kdesc_ctr + 192), c->kdesc_gen, __ATOMIC_RELEASE);
  __atomic_store_n((uint64_t*)(c->kdesc_ctr + 128), ~0ull, __ATOMIC_RELEASE);
  // chgo 广播区跨代清零(排在 stream_cons 上,老核退出后、新核启动前;
  // 不清的话上一代的退出广播(MSB)会让新核组员即刻误退)
  cudaMemsetAsync((void*)(uintptr_t)(c->kdesc_prog_dev + 64 * sizeof(uint64_t)), 0,
                  16 * sizeof(uint64_t), (cudaStream_t)c->stream_cons);
  {
    unsigned long long ring = c->kdesc_ring_dev, ctr = c->kdesc_ctr_dev,
                       prog = c->kdesc_prog_dev, start = c->kdesc_pushed,
                       gen = c->kdesc_gen;
    unsigned int cap = c->kdesc_cap, nch = kcons_nch();
    void* params[7] = {&ring, &ctr, &prog, &start, &cap, &gen, &nch};
    if (d.launch(c->kcons_fn, kcons_blocks(), 1, 1, kcons_threads(), 1, 1, 0,
                 c->stream_cons, params, nullptr) != 0)
      return false;                     // 没起来,无需叫停
  }
  return true;
}
static void maybe_open_gate_early(Comm* c);
// 描述符推入。false = 环满(调用方退回旧拷贝路径 —— credit 前缀游标容忍
// 乱序标记(R15b),两路混用安全)
static bool kdesc_push(Comm* c, uint64_t src_dev, void* dst_dev, uint64_t len) {
  if (c->kdesc_pushed - c->kdesc_consumed >= c->kdesc_cap) return false;
  Comm::KDesc& d = c->kdesc_ring[c->kdesc_pushed & (c->kdesc_cap - 1)];
  d.src = src_dev;
  d.dst = (uint64_t)(uintptr_t)dst_dev;
  d.len = len;
  __atomic_store_n((uint64_t*)(c->kdesc_ctr + 0), c->kdesc_pushed + 1,
                   __ATOMIC_RELEASE);          // 描述符字段先落,head 后发布
  c->kdesc_pushed++;
  return true;
}
// input_ready:caller 流已越过提交时刻的输入事件点(flagset 核直写 pinned
// 标志)。**消费者核在设备侧统一等这道门**(kernel 入口自旋 inflag),所以
// kdesc 描述符可以立即推;本函数只服务仍由 host 直接发起的读 sendbuf 路径
// 门的语义:生产者已写完 sendbuf,且 caller 流先前工作(含 caching allocator
// 复用块上的旧 kernel)全部完成 —— 读 sendbuf 与写 recvbuf 都以它为界,
// 否则出现"交付先落、旧 kernel 后写"的静默损坏(训练 loss 慢漂)。
static bool kdesc_input_ready(Comm* c) {
  if (c->input_ready) return true;
  if (c->kdesc_ctr &&
      __atomic_load_n((uint64_t*)(c->kdesc_ctr + 160), __ATOMIC_ACQUIRE)) {
    c->input_ready = true;
    return true;
  }
  return false;
}

// 解析消费者核可读的 src 指针:IPC 环槽在设备内存,verbs 接收槽在 host
// pinned(UVA 直读)。
static uint64_t kdesc_resolve_src(Comm* c, const void* payload, int shm_ring,
                                  int verbs_token, bool via_nic) {
  (void)c; (void)via_nic;
  if (shm_ring >= 0 || verbs_token >= 0) return (uint64_t)(uintptr_t)payload;
  return 0;
}

// ─────────────────────────────────────────────────────────── 池
static uint8_t* relay_ptr(Comm* c, int slot) {
  return (uint8_t*)c->relay_pool + (size_t)slot * (size_t)c->cfg.slice_bytes;
}
// verbs staging:payload 槽在 host pinned 池(消费者核 D2H 组装,NIC 从
// host MR 读 = 两平台皆满线速;见 internal.h stage_pay_pool 注);头由
// verbs_post_slice 写进 host 头池,双 SGE 同一 WR 发出
static uint8_t* stage_pay(Comm* c, int slot) {
  return (uint8_t*)c->stage_pay_pool + (size_t)slot * (size_t)c->cfg.slice_bytes;
}
static int relay_alloc(Comm* c) {
  if (c->relay_free.empty()) return -1;
  int s = c->relay_free.back(); c->relay_free.pop_back(); return s;
}
static int stage_alloc2(Comm* c) {
  if (c->stage_free.empty()) return -1;
  int s = c->stage_free.back(); c->stage_free.pop_back(); return s;
}

// ─────────────────────────────────────────────────────────── 待发/在途队列项
struct ReadySlice {
  SliceHdr h;
  void* gpu_src = nullptr;
  int relay_slot = -1;      // >=0:gpu_src 在中转池,D2H 完成后归还
};
struct InflightTx {
  SliceHdr h;
  void* ev = nullptr;
  int peer = -1;
  bool via_nic = false;
  int shm_ring = -1;        // >=0:publish 该环
  int stage_slot = -1;      // >=0:TCP 发送(writev 完成后归还)
  int relay_slot = -1;      // >=0:D2H 完成即可归还
};
struct InflightRx {
  SliceHdr h;
  void* ev = nullptr;
  bool via_nic = false;
  bool final_hop = false;
  int relay_slot = -1;      // 中转落地槽
  int src_shm_ring = -1;    // 来源:SHM 环(完成后 pop)
  int src_stage = -1;       // 来源:staging 槽(完成后 release)
  int src_verbs = -1;       // 来源:RDMA 预投槽(完成后 verbs_repost)
};
struct WaitRelay {          // 到达了但中转池满,先扣着源槽等
  SliceHdr h;
  void* payload;
  int src_shm_ring, src_stage, src_verbs;
  bool via_nic;
};

struct EarlyPolicy {
  int peer; uint16_t wave; uint8_t cid8;
  bool from_src = false;             // CT_POLICY_SRC(纯账本视图,永不执行)
  std::vector<PolicyEnt> ents;
};

// demand pend 环的深度(2 的幂)= **允许的 rank 间 collective 漂移上限**。
// 行是稀疏寄存的(每槽的 vector 懒分配),加深只在真漂移那么多时才占内存。
static constexpr int PEND_SLOTS = 64;

// S3 二期:消费者核完成队列的条目(done 计数按序驱动)。
// kind=0:交付(rx 收编);kind=1:机内 dev 环发送(完成后 host publish)
struct KEntry {
  int8_t kind = 0;   // 0=交付收编 1=dev环发送 2=纯拷贝 4=verbs staging 转发
  InflightRx r;
  InflightTx t;
};

// 这些队列不放进 internal.h(只有 engine 用),用一个隐藏状态挂在 Comm 上
struct EngineState {
  // 待发队列按**管**分开。合成一条 FIFO 会让被限速的 vNIC slice 把后面的
  // NV slice 一起堵住(队头阻塞),那等于凭空把两条独立的管串联起来 ——
  // 模拟就不忠实了。分开之后两条管各自按自己的 token bucket 前进。
  std::deque<ReadySlice> ready_nv, ready_nic;
  std::deque<InflightTx> tx;
  std::deque<InflightRx> rx;
  std::deque<KEntry> kq;             // S3 二期:消费者核完成队列(done 计数驱动)
  std::deque<WaitRelay> wait_relay;
  // 通道 A 基线(每次 collective 归零)
  uint64_t base_passed[RES_KINDS] = {0};
  // demand allgather 握手
  bool demand_sent = false;
  bool coll_started = false;
  int demand_got = 0;                            // 收到的**他人**行数,需 n-1
  // 提前到货的 demand 行按 coll_id 分槽寄存。槽数 = **rank 间允许的最大漂移**:
  // 同步档漂移 ≤1(每个 collective 都有隐式同步),4 槽绰绰有余;异步档
  // (MOSAIC_TORCH_ASYNC)app 可以连着塞 24 个 collective 不等,漂移轻易 ≥4 ⇒
  // 4 槽时 si=cid&3 别名,新代际把**当前 collective 还没收齐的行整槽擦掉**,
  // demand_got 永远停在半路 = 全局死锁(坑 #42,29355321-183004 实测 got=25/31)。
  std::vector<std::vector<uint64_t>> pend_rows;  // [coll_id % PEND_SLOTS][n*n]
  std::vector<std::vector<uint8_t>> pend_seen;   // [coll_id % PEND_SLOTS][n] 该行到了没
  uint8_t pend_cintent[PEND_SLOTS][MAX_RANKS] = {{0}};  // 行捎带的走廊承诺位,随行寄存
  std::vector<int> pend_cnt;
  // pend 槽属于哪个 coll_id:hint(O2)允许 rank 间跨 collective 的偏斜变大,
  // 槽位仅按 cid&3 复用会把 4 个 collective 前的陈旧行当成本次的 —— 记录归属,
  // 收编与写入都先对 cid(旧行为里偏斜 ≤1,巧合安全;现在显式化)。
  std::vector<int> pend_cid;                     // [PEND_SLOTS]
  // policy 累积(通道 B 的 charges 由本地累加,见 log.md)
  std::vector<std::vector<uint64_t>> peer_acc;
  std::vector<EarlyPolicy> early_policy;         // 比本 rank 的 coll_begin 还早到的
  int fin_got = 0;
  bool fin_sent = false;                         // 本 collective 的 CT_FIN 已广播
  // 早到的 CT_FIN(hint 下对端可能在我 coll_begin 前就收满 —— 空行 pair 场景)
  // 与 pend_rows 同一条纪律(坑 #42):异步档漂移可 ≥4,槽数必须按漂移上限给
  int fin_early_cid[PEND_SLOTS];
  uint8_t fin_early_who[PEND_SLOTS][MAX_RANKS] = {{0}};
};
static EngineState* ES(Comm* c);
static std::atomic<int> g_ev_phase{0};   // event 线程当前阶段(调试:冻结定位)
static std::vector<std::pair<Comm*, EngineState*>> g_es;
static std::mutex g_es_mu;
static EngineState* ES(Comm* c) {
  std::lock_guard<std::mutex> lk(g_es_mu);
  for (auto& p : g_es) if (p.first == c) return p.second;
  auto* e = new EngineState();
  g_es.push_back({c, e});
  return e;
}
static void ES_drop(Comm* c) {
  std::lock_guard<std::mutex> lk(g_es_mu);
  for (size_t i = 0; i < g_es.size(); i++)
    if (g_es[i].first == c) { delete g_es[i].second; g_es.erase(g_es.begin() + i); return; }
}

// ─────────────────────────────────────────────────────────── 限速器
static ResKind up_kind(bool via_nic) { return via_nic ? RES_NIC_UP : RES_NV_UP; }
static ResKind dn_kind(bool via_nic) { return via_nic ? RES_NIC_DN : RES_NV_DN; }


// 本 rank 自己的资源计数直接写进账本(比等自己的遥测新鲜)
static void sync_own_discharge(Comm* c) {
  auto* es = ES(c);
  int me = c->cfg.rank;
  for (int k = 0; k < RES_KINDS; k++)
    c->ledger.discharge[c->topo.res(me, k)] = c->buck[k].passed - es->base_passed[k];
  // fabric 聚合账 = 全体 rank 发布贡献之和 + 自己的实时贡献
  const int G = c->cfg.gpus_per_host;
  for (int j = 0; j < G && j < 8; j++) {
    uint64_t up = c->fab_own[0][j], dn = c->fab_own[1][j];
    for (int p = 0; p < c->cfg.nranks; p++) {
      if (p == me || c->peer_fab.empty()) continue;
      up += c->peer_fab[((size_t)p * 2 + 0) * 8 + j];
      dn += c->peer_fab[((size_t)p * 2 + 1) * 8 + j];
    }
    c->ledger.discharge[c->topo.res_fab_up(j)] = up;
    c->ledger.discharge[c->topo.res_fab_dn(j)] = dn;
  }
}

// ─────────────────────────────────────────────────────────── NIC 域异常检测器
// 逻辑逐行对照模拟器 mosaic_schedule.h:1961-2016(见 internal.h Detector 注释)。
// 平台适配(注明的常数差):模拟器用资源积压(_lg_arr)判"忙",真机 runtime 用
// (a) 在途/待发计数(UP)与 (b) 事件间隔 ≤4×额定 slice 服务时间(DN)近似 ——
// 语义同为"只统计忙时吞吐",不引入新自由度。
static int det_locked_global(Comm* c) {
  int n = c->det[0].locked + c->det[1].locked;
  for (int p = 0; p < c->cfg.nranks; p++) {
    if (p == c->cfg.rank) continue;
    uint8_t s = c->peer_det[p];
    n += (s & 1) + ((s >> 1) & 1);
  }
  return n;
}

void det_feed(Comm* c, int which, uint64_t bytes) {
  if (!c->cfg.detector) return;
  Detector& d = c->det[which];
  if (d.nom_gbps <= 0) return;
  uint64_t now = now_ns();
  // busy 判定对应模拟器 _lg_arr(**端口级**队列,不是链级 outstanding —— 后者
  // 把"字节还堵在上游 relay"也计成本口忙时,会系统性低估速率、过度改道,
  // 实测 hot×4 上比不开检测器还慢 33%,见 log.md 坑 #17):
  //   UP:上一事件后本地发送队列/在途 >0(精确,发送队列就在本进程手里);
  //   DN:上一完成事件被处理时 CQ 里还压着后续完成(det_dn_hint,verbs_poll 置)。
  bool busy_since_prev = d.busy_after && d.prev_ns > 0;
  if (which == 1) busy_since_prev = c->det_dn_hint && d.prev_ns > 0;
  if (d.prev_ns > 0 && now > d.prev_ns && busy_since_prev) {
    d.bacc += (double)bytes;
    d.tacc_ns += (double)(now - d.prev_ns);
    if (d.tacc_ns >= 500e3) {                       // 500µs 忙时折算一次
      double inst = d.bacc * 8.0 / d.tacc_ns;       // Gbps
      if (inst > d.nom_gbps) inst = d.nom_gbps;
      if (inst < 0.45 * d.nom_gbps) {
        if (d.streak < 250) d.streak++;
        // quorum≤2 门:全域已锁 ≥2 时(且自己未降级)不再新锁 —— 拥塞态冻结
        if (d.streak >= 3 && (d.locked || det_locked_global(c) < 2)) {
          double est = d.locked ? 0.5 * d.rate_gbps + 0.5 * inst : inst;
          if (est < d.nom_gbps / 16.0) est = d.nom_gbps / 16.0;
          if (!d.locked) {
            d.locked = true; c->st.detector_events++;
            MLOG_WARN("检测器锁定:NIC%s 实测 %.2fG < 0.45×%.2fG(账本额定改实测)",
                      which == 0 ? "↑" : "↓", est, d.nom_gbps);
          }
          MTRACE(TE_DET, (uint32_t)which, 1, (uint64_t)(est * 1000));
          d.rate_gbps = est;
        }
      } else {
        d.streak = 0;
        if (d.locked) { d.locked = false; c->st.detector_events++; }
        d.rate_gbps = d.nom_gbps;                   // 恢复即回额定
      }
      // 账本额定改实测(属主本地立即生效;他人经通道 A)
      int res = c->topo.res(c->cfg.rank, which == 0 ? RES_NIC_UP : RES_NIC_DN);
      c->ledger.cap_Bps[res] = d.rate_gbps * 1e9 / 8.0;
      c->st.det_rate_gbps[which] = d.rate_gbps;
      d.bacc = 0; d.tacc_ns = 0;
    }
  }
  if (which == 0) {
    auto* es2 = ES(c);
    d.busy_after = !es2->ready_nic.empty() || c->nic_tx_outstanding > 0;
  }
  d.prev_ns = now;
}

static uint64_t rng_next(Comm* c) {
  uint64_t& s = c->rng;
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return s;
}

// ─────────────────────────────────────────────────────────── 账本注水(L2 核心)
// 代价 = 这批字节落进该 corridor 后,其资源签名里"最忙的口"的投影完成时间。
// 与模拟器的 24 步贪心注水同语义(报告 §11.2「落入后最大投影完成最小」)。
// S1(sota_plan,handoff #2):MOSAIC_HOP_COST_US = 每多一跳的固定开销(µs),
// 计入投影 —— 小报文时多跳走廊的固定成本(中转 H2D/D2H、二次排队、slice 头)
// 不再被带宽项掩盖,"过度分流"由代价函数自己抑制。默认 0 = 旧行为。
static double opt_hop_cost_s() {
  static double v = -1;
  if (v < 0) {
    const char* e = getenv("MOSAIC_HOP_COST_US");
    v = (e && *e) ? atof(e) * 1e-6 : 0;
  }
  return v;
}
static double corridor_cost(Comm* c, const Chain& ch, uint64_t extra) {
  // S2v2:tier_blind(auto-tier 判为非热点小报文)= 本 collective 决策盲档,
  // 只计每跳固定成本(直发必胜)—— 与"BNV=0 盲档"champion 行为对齐,
  // 但无需按 cell 换全局配置。
  if (c->tier_blind) return opt_hop_cost_s() * (ch.nhops - 1);
  int res[16];
  int n = c->topo.chain_res(ch, res);
  double worst = 0;
  for (int i = 0; i < n; i++) {
    double cap = c->ledger.cap_Bps[res[i]];
    if (cap <= 0) continue;
    double v = (double)(c->ledger.outstanding(res[i]) + extra) / cap;
    if (v > worst) worst = v;
  }
  return worst + opt_hop_cost_s() * (ch.nhops - 1);
}

static void charge_chain_local(Comm* c, const Chain& ch, uint64_t bytes) {
  int res[16];
  int n = c->topo.chain_res(ch, res);
  c->ledger.charge_chain(res, n, bytes);
}

// ─────────────────────────────────────────────────────────── 发射一个 hop
// PC(pair_color)门:模拟器把 (wave,hop) 相同的发送按 (dst-src) mod K 的对角线
// 子槽排序(min-epoch 是偏好不是闸,mosaic_schedule.h:1663-1671)。runtime 的
// 对应物 = 待发队列按同一 key 有序插入:key = (wave*4+hop)*K + sub。
// 等 key 保持 FIFO(稳定),仍然 work-conserving —— 语义一致。
static uint32_t pc_key(Comm* c, const SliceHdr& h) {
  uint32_t base = ((uint32_t)h.wave * 4u + h.hop);
  if (c->pc_k <= 0) return base << 8;
  uint32_t K = (uint32_t)c->pc_k;
  uint32_t sub = ((h.dst % K) + K - (h.src % K)) % K;
  return base * K + sub;
}
// 发送序:(wave,hop[,PC 色])结构序(PC 是决策语义)。结构序内不再做任何
// 热度排序:NIC 排干队列本身即空泡遮蔽,发送序不影响 makespan
// (raven-0819-verbs-plane,24 格 A/B 全 <0.5% 噪声)。true = a 应排在 b 之前。
static inline bool order_before(Comm* c, const SliceHdr& a, const SliceHdr& b) {
  return pc_key(c, a) < pc_key(c, b);
}
static void enqueue_hop(Comm* c, const SliceHdr& h, void* gpu_src, int relay_slot) {
  ReadySlice r;
  r.h = h; r.gpu_src = gpu_src; r.relay_slot = relay_slot;
  const int peer = h.path[h.hop + 1];
  auto* es = ES(c);
  auto& q = c->topo.same_host(c->cfg.rank, peer) ? es->ready_nv : es->ready_nic;
  if (c->pc_k > 0) {
    auto it = q.end();
    while (it != q.begin()) {          // 从尾部找插入点(新 slice 通常排尾)
      auto prev = it; --prev;
      if (!order_before(c, h, prev->h)) break;
      it = prev;
    }
    q.insert(it, r);
  } else {
    q.push_back(r);
  }
}

// 按显式链发射:把 (src→dst) 的一段字节按 slice_bytes 切开,挂上给定 hop 链。
// corr 标签只用于统计归类,按链形状推断(DIRECT/RX/TX/SPREAD)。
static void emit_chain(Comm* c, int dst, const Chain& ch, uint64_t off, uint64_t len, int wave) {
  int corr;
  if (ch.nhops == 1) corr = CORR_DIRECT;
  else if (ch.nhops == 2)
    corr = c->topo.same_host(c->cfg.rank, ch.path[1]) ? CORR_TX : CORR_RX;
  else corr = CORR_SPREAD0;
  const uint64_t S = (uint64_t)c->cfg.slice_bytes;
  for (uint64_t p = 0; p < len; p += S) {
    SliceHdr h{};
    h.magic = SLICE_MAGIC;
    h.src = (uint16_t)c->cfg.rank; h.dst = (uint16_t)dst;
    h.off = off + p; h.len = (uint32_t)std::min(S, len - p);
    h.wave = (uint16_t)wave; h.corr = (uint8_t)corr;
    h.coll = (uint16_t)(c->coll_id & 0xFFFF);   // 坑 #48:代际随数据走
    h.hop = 0; h.nhops = ch.nhops;
    for (int i = 0; i <= ch.nhops; i++) h.path[i] = ch.path[i];
    void* gsrc = (uint8_t*)const_cast<void*>(c->sendbuf) + c->sdispls[dst] + h.off;
    enqueue_hop(c, h, gsrc, -1);
  }
}

// 把 (src→dst) 的一段字节挂上指定 corridor 的链
static void emit_range(Comm* c, int dst, int corr, uint64_t off, uint64_t len, int wave) {
  Chain ch;
  if (!c->topo.chain_of(c->cfg.rank, dst, corr, &ch)) {
    MLOG_ERR("E2003 corridor %d 对 (%d→%d) 不可用", corr, c->cfg.rank, dst);
    return;
  }
  emit_chain(c, dst, ch, off, len, wave);
}

// 一个 (s→d) pair 的配额注水:按账本代价把 quota 切进可用走廊(per_corr),
// charges 记入本地账本。dst 驱动(d=me)与 src 驱动(s=me)共用 ——
// 注水算法的唯一来源,两相的决策可与模拟器 lg_split 逐字对拍。
// 返回可用走廊数(0 = 该 pair 无路可走,调用方跳过)。
static int waterfill_pair(Comm* c, int s, int d, uint64_t quota, int c_limit,
                          uint64_t per_corr[NCORR]) {
  int cl[NCORR], ncl = c->topo.corridors_of(s, d, c_limit, cl);
  if (ncl == 0) return 0;
  for (int i = 0; i < NCORR; i++) per_corr[i] = 0;
  if (!c->cfg.ledger || ncl == 1) {
    per_corr[cl[0]] = quota;                   // 退化档:全走第一条(= DIRECT/NV)
    // **必须照样记账**:机内 pair 只有一条走廊(NV 单跳),但它占用的正是
    // 目的地的 NV 下行管。不记的话,后面规划跨机流量时账本会认为 NV 管空闲,
    // 于是把大量跨机流量改道到早已饱和的 NV 管上 —— 那恰好推翻 §4.2
    // 「same_d 与 cross_d 在两管间注水」的整个前提。
    // 实测代价:0.5:1 点上 69% 的跨机流量被错误改道,慢 2 倍。
    if (c->cfg.ledger) {
      Chain ch0;
      if (c->topo.chain_of(s, d, cl[0], &ch0)) charge_chain_local(c, ch0, quota);
    }
    return ncl;
  }
  // 24 步贪心注水。粒度**不能**用 slice_bytes 兜底:那是传输层的分片大小,
  // 与规划粒度是两码事。步长公式与模拟器 lg_split 逐字一致:
  // step = ((quota/24)/1024)*1024,地板 1024。
  uint64_t gran = ((quota / 24) / 1024) * 1024;
  if (gran < 1024) gran = 1024;
  uint64_t done = 0;
  while (done < quota) {
    uint64_t g = std::min(gran, quota - done);
    // tie-break 与模拟器逐字一致:best_lvl 是**运行最小值**(只在更低时更新),
    // 5% 级带内取短链。
    int best = -1; double best_lvl = 0; int best_hops = 99;
    for (int i = 0; i < ncl; i++) {
      Chain ch;
      if (!c->topo.chain_of(s, d, cl[i], &ch)) continue;
      double v = corridor_cost(c, ch, g);
      if (best < 0) {
        best = i; best_lvl = v; best_hops = ch.nhops;
      } else if (v < best_lvl * 0.95 ||
                 (v < best_lvl * 1.05 && ch.nhops < best_hops)) {
        best = i; best_hops = ch.nhops;
        if (v < best_lvl) best_lvl = v;
      }
    }
    if (best < 0) break;
    Chain ch;
    c->topo.chain_of(s, d, cl[best], &ch);
    charge_chain_local(c, ch, g);
    per_corr[cl[best]] += g;
    done += g;
  }
  return ncl;
}

// ─────────────────────────────────────────────────────────── 波装配(dst 控制环)
static void emit_wave(Comm* c) {
  DstCtrl& d = c->dst;
  const int n = c->cfg.nranks, me = c->cfg.rank;
  const uint64_t t0 = now_ns();

  int waves_left = eff_W(c) - d.wave_idx;
  if (waves_left < 1) waves_left = 1;

  std::vector<PolicyEnt> ents;
  ents.reserve(n * NCORR);
  uint64_t wave_total = 0;

  for (int s = 0; s < n; s++) {
    if (s == me || d.remaining[s] == 0) continue;
    uint64_t quota = (waves_left <= 1) ? d.remaining[s]
                                       : (d.remaining[s] + waves_left - 1) / waves_left;
    // 承诺兑现:corridor_intent_bit=0 的 coll 只许直发 —— 中转义务门凭承诺位
    // 放行别人收尾,违背 = 对端已拆核,中转片静默丢失
    uint64_t per_corr[NCORR];
    if (!waterfill_pair(c, s, me, quota, c->my_corr_intent ? eff_C(c) : 1, per_corr))
      continue;
    // 比例定了之后再切**连续**区间:每 (src,corridor) 一条 entry
    // (线上报文因此是 O(n·C) 而不是 O(n·24),与论文的 0.3-2.6KB 量级一致)
    uint64_t cur = d.cursor[s];
    for (int cc = 0; cc < NCORR; cc++) {
      if (!per_corr[cc]) continue;
      PolicyEnt e{};
      e.src = (uint16_t)s; e.corr = (uint8_t)cc;
      e.off = cur; e.len = (uint32_t)per_corr[cc];
      ents.push_back(e);
      c->st.ingress_corr[cc & 7] += per_corr[cc];   // 入向分配(热点 dst 的分流实况)
      cur += per_corr[cc];
      wave_total += per_corr[cc];
    }
    d.cursor[s] = cur;
    d.remaining[s] -= std::min(d.remaining[s], quota);
  }

  d.wave_bytes = wave_total;
  d.wave_delivered = 0;
  d.wave_start_ns = now_ns();
  // 波截止:75% 触发是"波差不多完了"的信号,但真机上少量 3 跳中转 slice 的
  // 延迟远大于直发(0.5:1 点实测 20ms+ vs 4ms),尾部会把下一波卡住 ——
  // 实测瓶颈管利用率因此只有 52%。加一个按两管注水速率估的兜底期限,
  // 到期就发下一波(在途仍 ≤2 波,不破坏 wave = implicit flow control 的性质)。
  {
    double cap_Bps = (c->cfg.nv_gbps + c->cfg.nic_gbps) * 1e9 / 8.0;
    double est_us = cap_Bps > 0 ? (double)wave_total / cap_Bps * 1e6 : 0;
    // MOSAIC_WAVE_DEADLINE:0 = 关兜底(纯 75% 触发,= 模拟器语义,对拍/归因用);
    // 默认 1.5×估计(D6:真机中转尾延迟会饿死流水线,见上方注释)
    static double dl_mult = -1;
    if (dl_mult < 0) {
      const char* e = getenv("MOSAIC_WAVE_DEADLINE");
      dl_mult = e && *e ? atof(e) : 1.5;
    }
    d.wave_deadline_ns = dl_mult > 0
        ? d.wave_start_ns + (uint64_t)(est_us * dl_mult * 1000.0) : 0;
  }

  double plan_us = (now_ns() - t0) / 1000.0;
  c->st.plan_us_total += plan_us; c->st.plan_calls++;
  if (plan_us > c->st.plan_us_max) c->st.plan_us_max = plan_us;
  c->plan_hist[c->plan_hist_n++ & 255] = plan_us;   // 环形样本 → p50/p99(GetStats)
  c->st.waves_emitted++;

  // 通道 B:把这一波怎么分广播出去。它同时是**执行指令**,所以走可靠通道。
  // aux 高 8 位 = coll_id(mod 256):hint(O2)下对端可能仍在上一个 collective,
  // 收端凭它把"来自我下一个 collective 的策略"寄存而不是误执行(数据安全线)。
  if (!ents.empty() && c->cfg.bcast) {
    const uint16_t aux = (uint16_t)(((c->coll_id & 0xFF) << 8) | (d.wave_idx & 0xFF));
    if (opt_policy_gbcast()) {
      // 任务一(全局账本视野):**完整分配发给所有 rank** —— 一条链的占用含
      // 中转跳(RX/TX/SPREAD 借道第三方的 NV/NIC),中转节点必须知道自己被借
      // 了多少,否则它自己决策时会高估可用带宽。收方 apply_policy 对全部条目
      // 累加 charges(peer_acc[发布者] 因此是完整视图),只执行 src==自己的
      // 条目。归属按**决策者**(单写者,与模拟器 _lg_chg_by/_lg_pubsum 同型,
      // 无重复计账);条目列表就是 charges 的紧凑编码,免 O(nres) 全向量。
      // 代价 = 每 rank 收到全量条目而非自己那几条,进 chB_bytes 计量。
      for (int p = 0; p < n; p++) {
        if (p == me) continue;
        ctrl_send(c, p, CT_POLICY, aux, ents.data(),
                  (uint32_t)(ents.size() * sizeof(PolicyEnt)));
        c->st.chB_msgs++;
        c->st.chB_bytes += sizeof(CtrlHdr) + ents.size() * sizeof(PolicyEnt);
      }
    } else {
      // MOSAIC_POLICY_GBCAST=0(A/B 退路):只发给本波的 src —— 非 src 的
      // rank 看不到这份策略,它对中转占用的账本视图残缺
      for (int p = 0; p < n; p++) {
        if (p == me) continue;
        std::vector<PolicyEnt> mine;
        for (auto& e : ents) if (e.src == p) mine.push_back(e);
        if (mine.empty()) continue;
        ctrl_send(c, p, CT_POLICY, aux, mine.data(),
                  (uint32_t)(mine.size() * sizeof(PolicyEnt)));
        c->st.chB_msgs++;
        c->st.chB_bytes += sizeof(CtrlHdr) + mine.size() * sizeof(PolicyEnt);
      }
    }
  }
  // 本 rank 自己作为 src 的部分直接执行
  for (auto& e : ents)
    if (e.src == me) emit_range(c, me, e.corr, e.off, e.len, d.wave_idx);

  MLOG_DBG("wave %d 装配:%llu 字节 %zu 条 plan=%.1fus", d.wave_idx,
           (unsigned long long)wave_total, ents.size(), plan_us);
  c->dbg_last_ents = ents;                    // 对拍/调试:留一份完整分配
  MTRACE(TE_WAVE_EMIT, (uint32_t)d.wave_idx, wave_total, (uint64_t)plan_us);
  d.wave_idx++;
}

static uint8_t corridor_intent_bit(Comm* c);
void engine_test_emit_wave(Comm* c) {
  // 回放 harness 不走 demand 块:走廊承诺位就地按同一谓词补算
  // (rcounts 从 remaining 同步 —— 真路径里 remaining 正是从 rcounts 来的)
  for (int s = 0; s < c->cfg.nranks; s++)
    if (s != c->cfg.rank) c->rcounts[s] = c->dst.remaining[s];
  c->src_mode = false;
  c->my_corr_intent = corridor_intent_bit(c);
  emit_wave(c);
}

// ─────────────────────────────────────────────────────── 波装配(src 控制环)
// dispatch 相的发波:src 对自己的行注水并**自己执行**(数据从本 rank 流出,
// 无策略往返);通道 B 只承担账本视图(CT_POLICY_SRC,收方永不执行)。
// 结构与 emit_wave 镜像,注水共用 waterfill_pair —— 两相决策同源可对拍。
static void emit_wave_src(Comm* c) {
  SrcCtrl& s = c->srcc;
  const int n = c->cfg.nranks, me = c->cfg.rank;
  const uint64_t t0 = now_ns();

  int waves_left = eff_W(c) - s.wave_idx;
  if (waves_left < 1) waves_left = 1;

  std::vector<PolicyEnt> ents;
  ents.reserve(n * NCORR);
  uint64_t wave_total = 0, wave_nic = 0;

  for (int d2 = 0; d2 < n; d2++) {
    if (d2 == me || s.remaining[d2] == 0) continue;
    // 行门(不变量替代握手,internal.h peer_row_seen):对端本 coll 的行未到
    // = 无法证明它的 recv 缓冲已切换,本波跳过它;行到后由 pump_protocol 的
    // 补发路径追齐。离线回放(peer_row_seen 空)视为全开。
    if (!c->peer_row_seen.empty() && !c->peer_row_seen[d2]) continue;
    uint64_t quota = (waves_left <= 1) ? s.remaining[d2]
                                       : (s.remaining[d2] + waves_left - 1) / waves_left;
    uint64_t per_corr[NCORR];
    if (!waterfill_pair(c, me, d2, quota, c->my_corr_intent ? eff_C(c) : 1, per_corr))
      continue;
    uint64_t cur = s.cursor[d2];
    for (int cc = 0; cc < NCORR; cc++) {
      if (!per_corr[cc]) continue;
      PolicyEnt e{};
      e.src = (uint16_t)d2;                 // CT_POLICY_SRC 语义:端点 = dst
      e.corr = (uint8_t)cc;
      e.off = cur; e.len = (uint32_t)per_corr[cc];
      ents.push_back(e);
      c->st.ingress_corr[cc & 7] += per_corr[cc];   // src 相 = 出流分配(mosaic.h 注)
      Chain ch;
      if (c->topo.chain_of(me, d2, cc, &ch) && !c->topo.same_host(me, ch.path[1]))
        wave_nic += per_corr[cc];           // hop0 跨机才有 RC ACK,进 ack 门分母
      cur += per_corr[cc];
      wave_total += per_corr[cc];
    }
    s.cursor[d2] = cur;
    s.remaining[d2] -= std::min(s.remaining[d2], quota);
  }

  s.wave_nic_bytes = wave_nic;
  s.wave_acked = 0;
  s.wave_start_ns = now_ns();
  // 兜底期限:src 的波从本 rank 的出向管(NV↑+NIC↑)离开,估计式 =
  // 波字节 ÷ 出向聚合额定 ×(与 dst 环同一个 MOSAIC_WAVE_DEADLINE 倍数)。
  // dst 环的公式对的是"进向聚合"(nv+nic 同数值),src 侧语义换成出向。
  {
    double cap_Bps = (c->cfg.nv_gbps + c->cfg.nic_gbps) * 1e9 / 8.0;
    double est_us = cap_Bps > 0 ? (double)wave_total / cap_Bps * 1e6 : 0;
    static double dl_mult = -1;
    if (dl_mult < 0) {
      const char* e = getenv("MOSAIC_WAVE_DEADLINE");
      dl_mult = e && *e ? atof(e) : 1.5;
    }
    s.wave_deadline_ns = dl_mult > 0
        ? s.wave_start_ns + (uint64_t)(est_us * dl_mult * 1000.0) : 0;
  }

  double plan_us = (now_ns() - t0) / 1000.0;
  c->st.plan_us_total += plan_us; c->st.plan_calls++;
  if (plan_us > c->st.plan_us_max) c->st.plan_us_max = plan_us;
  c->plan_hist[c->plan_hist_n++ & 255] = plan_us;
  c->st.waves_emitted++;

  // 通道 B:src 波的完整分配广播给**所有** rank(纯账本视图;任务一的全局
  // 视野在 src 相的对应物 —— 中转节点从这里知道自己被借了多少)。
  if (!ents.empty() && c->cfg.bcast) {
    const uint16_t aux = (uint16_t)(((c->coll_id & 0xFF) << 8) | (s.wave_idx & 0xFF));
    for (int p = 0; p < n; p++) {
      if (p == me) continue;
      ctrl_send(c, p, CT_POLICY_SRC, aux, ents.data(),
                (uint32_t)(ents.size() * sizeof(PolicyEnt)));
      c->st.chB_msgs++;
      c->st.chB_bytes += sizeof(CtrlHdr) + ents.size() * sizeof(PolicyEnt);
    }
  }
  // 自己执行:条目端点是 dst,数据从 sendbuf + sdispls[dst] + off 切片发射
  for (auto& e : ents) emit_range(c, e.src, e.corr, e.off, e.len, s.wave_idx);

  MLOG_DBG("src wave %d 装配:%llu 字节(nic %llu)%zu 条 plan=%.1fus", s.wave_idx,
           (unsigned long long)wave_total, (unsigned long long)wave_nic,
           ents.size(), plan_us);
  c->dbg_last_ents = ents;
  MTRACE(TE_WAVE_EMIT, (uint32_t)s.wave_idx, wave_total, (uint64_t)plan_us);
  s.wave_idx++;
}

void engine_test_emit_wave_src(Comm* c) {
  for (int d = 0; d < c->cfg.nranks; d++)
    if (d != c->cfg.rank) c->scounts[d] = c->srcc.remaining[d];
  c->src_mode = true;
  c->my_corr_intent = corridor_intent_bit(c);
  emit_wave_src(c);
}

// src 波的 ACK 计数(verbs_poll 数据 WR send CQ 成功时调):RC send 完成 =
// 对端已收。只计 hop0 且 h.src==me 的片(= 本 rank 自己的行字节;中转转发的
// 片 hop>0,不属于本 rank 的波)。
void src_wave_ack(Comm* c, const SliceHdr& h) {
  if (!c->src_mode || !c->srcc.active) return;
  if (h.hop != 0 || (int)h.src != c->cfg.rank) return;
  if ((int)h.wave == c->srcc.wave_idx - 1) c->srcc.wave_acked += h.len;
}

// DIRECT:不需要 dst 协调,每个 src 用全局矩阵自己算自己的行(诊断位)
static void apply_policy(Comm* c, int peer, uint16_t wave, const PolicyEnt* e, int ne);
static void apply_policy_src(Comm* c, int peer, const PolicyEnt* e, int ne);

static void plan_static(Comm* c) {
  const int n = c->cfg.nranks, me = c->cfg.rank;
  for (int d = 0; d < n; d++) {
    if (d == me || c->scounts[d] == 0) continue;
    int corr = CORR_DIRECT;
    // 网络虚拟化 rail 档:跨 rail 没有直连通道 ⇒ 静态规划器也必须换轨。
    // 取第一条合法走廊(corridors_of 已按 chain_of 过滤,rail 档下天然排除 DIRECT)。
    // origin 档下 DIRECT 恒合法,这段等价于旧行为(逐字节不变)。
    if (c->topo.net_rail) {
      Chain probe;
      if (!c->topo.chain_of(me, d, CORR_DIRECT, &probe)) {
        int cl[NCORR], ncl = c->topo.corridors_of(me, d, c->cfg.corridors, cl);
        if (ncl == 0) { MLOG_ERR("E2004 rail 档下 (%d→%d) 无任何可用走廊", me, d); continue; }
        corr = cl[0];
      }
    }
    emit_range(c, d, corr, 0, c->scounts[d], 0);
  }
}

// ─────────────────────────────────────────────────────────── P2 读集意图
// dst 视角:本 collective 我会读哪些 (owner,kind) 的账 —— 由本地列(rcounts)
// × 候选走廊全集(保守超集,覆盖逐波改道)推出,每 owner 一个字节的位图,
// 捎带在 CT_DEMAND 里广播。cap<=0 的资源从不被 corridor_cost 读(投影跳过),
// 从意图中剪掉 —— 不限速跑法下意图自然为空,遥测缩到 det-only。
static void compute_intent(Comm* c, std::vector<uint8_t>& out) {
  const int n = c->cfg.nranks, me = c->cfg.rank;
  if (c->cfg.sched != MOSAIC_SCHED_MOSAIC || !opt_telem_targeted()) {
    out.assign(n, INTENT_FULL);
    return;
  }
  out.assign(n, 0);
  bool need_fab = false;
  int cl[NCORR], res[16];
  for (int s = 0; s < n; s++) {
    if (s == me || c->rcounts[s] == 0) continue;
    int ncl = c->topo.corridors_of(s, me, c->cfg.corridors, cl);
    for (int i = 0; i < ncl; i++) {
      Chain ch;
      if (!c->topo.chain_of(s, me, cl[i], &ch)) continue;
      int nr = c->topo.chain_res(ch, res);
      for (int k = 0; k < nr; k++) {
        if (c->ledger.cap_Bps[res[k]] <= 0) continue;
        if (res[k] < n * RES_KINDS)
          out[res[k] / RES_KINDS] |= (uint8_t)(1u << (res[k] % RES_KINDS));
        else
          need_fab = true;    // fabric 是全员贡献的聚合账 → 要所有 owner 的份额
      }
    }
  }
  if (need_fab)
    for (int r = 0; r < n; r++) out[r] |= INTENT_FAB;
}

// ─────────────────────────────────────────────────────────── 走廊承诺位
// dst 视角:本 coll 我是否可能给任何 src 计划非直发路径(=可能产生中转义务)。
// 判据 = 存在某条非直发链,其 NIC 资源(**按物理单元**,topo.nic_leader)与
// 直发不同 —— 即中转真能把 NIC 负载挪到另一根物理管子。共享 HCA 拓扑
// (Raven 每节点 1×HDR200)下 RX/TX 中转与直发落同一单元 ⇒ 恒 0 ⇒
// dst 承诺全程直发,coll_try_finish 的中转义务门对它免除。
// 多 rail / 每 GPU 私有 NIC 的拓扑自动变 1(方法论:拓扑推导,零集群硬编码)。
// 位=0 时 emit_wave 强制 c_limit=1,承诺必须兑现(否则义务门失效 = 静默丢中转片)。
static uint8_t corridor_intent_bit(Comm* c) {
  if (c->cfg.sched != MOSAIC_SCHED_MOSAIC || c->cfg.corridors <= 1) return 0;
  if (c->topo.net_rail) return 1;    // rail 档跨轨无直连,结构性依赖中转
  const int n = c->cfg.nranks, me = c->cfg.rank;
  int cl[NCORR], res_d[16], res_c[16];
  for (int p = 0; p < n; p++) {
    // 谓词按**决策方向**取 pair:dst 相看入向列(s=p→me,量 = rcounts),
    // src 相看出向行(me→d=p,量 = scounts)—— 承诺的都是"我做的计划
    // 会不会借道第三方",只是计划的方向随决策者反转。
    const int s = c->src_mode ? me : p, d = c->src_mode ? p : me;
    const uint64_t cnt = c->src_mode ? c->scounts[p] : c->rcounts[p];
    if (p == me || cnt == 0 || c->topo.same_host(p, me)) continue;
    Chain dch;
    if (!c->topo.chain_of(s, d, CORR_DIRECT, &dch)) continue;
    int nd = c->topo.chain_res(dch, res_d);
    int ncl = c->topo.corridors_of(s, d, c->cfg.corridors, cl);
    for (int i = 0; i < ncl; i++) {
      if (cl[i] == CORR_DIRECT) continue;
      Chain ch;
      if (!c->topo.chain_of(s, d, cl[i], &ch)) continue;
      int nc = c->topo.chain_res(ch, res_c);
      for (int k = 0; k < nc; k++) {
        if (res_c[k] >= n * RES_KINDS) continue;            // fabric 聚合账不算
        int kind = res_c[k] % RES_KINDS;
        if (kind != RES_NIC_UP && kind != RES_NIC_DN) continue;
        bool in_direct = false;
        for (int q = 0; q < nd; q++)
          if (res_d[q] == res_c[k]) { in_direct = true; break; }
        if (!in_direct) return 1;
      }
    }
  }
  return 0;
}

// ─────────────────────────────────────────────────────────── collective 起止
// MOSAIC_CREDIT_HOT_W0:热目的列的每 src 初始突发额度(slice;容量类,不改
// 数据面走法)。默认 2:4×4 形态 12 跨机 src × 2 × 1MB slice = 24MB 初始
// 站队,与 node_inflight 帽和 NCCL 每对 ~4MB 槽环同量级(Raven α2 悬崖
// 标定,File/design-credit-backpressure-0821.md §3.4)。调大 = 热列首波
// 流水更满但初始突发更逼近排水悬崖;调小 = 保护更强、深斜首波更慢。
// 稳态吞吐与它无关(稳态窗随 credit 回流自增长)。
static uint64_t credit_hot_w0() {
  static uint64_t v = 0;
  if (!v) {
    const char* e = getenv("MOSAIC_CREDIT_HOT_W0");
    long w = e && *e ? atol(e) : 2;
    if (w < 1) w = 1;
    v = (uint64_t)w;
  }
  return v;
}

static void coll_begin(Comm* c) {
  g_ev_phase = 16;
  c->ph_submit_snap = c->ph_submit_ns;
  c->ph_notice_snap = c->ph_notice_ns;
  c->ph_bcast_snap = c->ph_bcast_ns;
  auto* es = ES(c);
  const int n = c->cfg.nranks, me = c->cfg.rank;
  c->ph_begin_ns = now_ns();

  // O1 的 WaitEvent 在提交路径(api.cc)挂,不在这里 —— src 驱动调度器的对端
  // 数据可早于本函数执行,必须保证任何本 collective 的 H2D 都排在输入事件之后。

  // O2 verify 档:allgather 收齐后,把推导矩阵与真实矩阵对拍(只验不省)
  if (es->coll_started == false && c->demand_check_on) {
    c->demand_check_on = false;
    int bad = 0;
    for (size_t i = 0; i < c->demand.size(); i++)
      if (c->demand[i] != c->demand_check[i]) bad++;
    if (bad)
      MLOG_ERR("E4006 demand hint verify:推导矩阵与实际 %d 项不符(coll=%d dm=%d)"
               " —— 该负载不满足恒等前提,勿开 trust 档", bad, c->coll_id, c->demand_mode);
    else
      MLOG_DBG("demand hint verify OK(coll=%d dm=%d)", c->coll_id, c->demand_mode);
  }

  // (O2/P1 的 demand_prev 快照在 coll_try_finish 做 —— SSP 下 begin 时矩阵
  //  还残缺,结束时后台行大概率已齐,快照质量最好)

  // dev-pipe2:本 coll 直写账基线(DLV 计数跨 coll 单调,起点快照取差)
  c->rwin_published = false;
  for (int p = 0; p < n; p++) {
    c->dlv_folded[p] = 0;
    c->dlv_base[p] = (p != me && c->rx_ring[p]) ? shm_dlv_read(c->rx_ring[p]) : 0;
  }
  c->ledger.clear_epoch();
  for (int k = 0; k < RES_KINDS; k++) es->base_passed[k] = c->buck[k].passed;
  memset(c->fab_own, 0, sizeof(c->fab_own));               // fabric 账随 epoch 清零
  std::fill(c->peer_fab.begin(), c->peer_fab.end(), 0);
  es->peer_acc.assign(n, std::vector<uint64_t>(c->topo.nres(), 0));
  es->fin_got = 0;
  es->fin_sent = false;

  // 信用反压:本 coll 突发钳基线快照 + W0 复位为全窗(热目的列在下方 W
  // 自适应块内按 demand 列覆盖;File/design-credit-backpressure-0821.md §3.4)
  {
    const uint64_t cr_nrx = (uint64_t)rdma_rx_slots(c->cfg.slice_bytes);
    for (int p = 0; p < n; p++) {
      c->cr_base_sent[p] = c->cr_sent[p];
      c->cr_base_limit[p] = c->cr_limit[p];
      c->cr_w0[p] = cr_nrx;
    }
  }

  DstCtrl& d = c->dst;
  // **不能整体重置**:对端可能在我完成 demand allgather 之前就开始发本次
  // collective 的数据了(DIRECT/FAST 由 src 自行决策,不等我的策略广播)。
  // 那些字节已经正确落进 recvbuf 并计过数,这里一清就成了"永远收不满"——
  // 症状是间歇性挂死、数据却是对的。交付计数改在**上一次 collective 结束时**清。
  const uint64_t keep_recv = d.total_recv;
  uint64_t keep_by_src[MAX_RANKS];
  memcpy(keep_by_src, d.recv_by_src, sizeof(keep_by_src));
  d = DstCtrl();
  d.total_recv = keep_recv;
  memcpy(d.recv_by_src, keep_by_src, sizeof(keep_by_src));
  d.active = true;
  // 收期望必须取**调用方给的 rcounts**,不能取 demand 矩阵(坑 #43)。
  // 两个理由,后者是死锁根因:
  //  ①对称性:total_send 取的就是本地 scounts(下面 948 行),收侧却取网络来的
  //    矩阵,两侧真理来源不同 —— 任何一侧不新鲜就自相矛盾;
  //  ②`c->demand` **跨 collective 不清零**(只在行到达时逐行覆盖),而 SSP(P1,
  //    默认开)与 hint(O2)都允许不等 allgather 就开跑 ⇒ 尚未到货的行仍是**上一个
  //    collective 的值**。对端按新 scounts 只发 0 字节,我却按旧行等 540672B,
  //    永久挂死(29355321-184351 实测:r20 coll18 src3 want=540672 recv=0,
  //    而 r3 已完成 coll18 进到 19)。反向(旧行更小)更糟:提前判完成、静默丢数据。
  // rcounts 是 alltoallv 的入参,本地已知且权威 —— 收期望本就不该绕网络一圈。
  for (int s = 0; s < n; s++) {
    if (s == me) continue;
    uint64_t b = c->rcounts[s];
    d.remaining[s] = b;
    d.total_expect += b;
  }
  c->total_recv_expect = d.total_expect;
  // 零收量 rank 也必须广播 FIN:中转义务门(coll_try_finish)等全体 FIN,
  // 不发会把别人钉死;语义上它本就已"收齐"。
  if (d.total_expect == 0) {
    es->fin_sent = true;
    ctrl_bcast(c, CT_FIN, (uint16_t)(c->coll_id & 0xFFFF), nullptr, 0);
  }
  c->total_send = 0;
  for (int t = 0; t < n; t++) if (t != me) c->total_send += c->scounts[t];
  c->posted_send = 0;

  // src 驱动 dispatch(任务二):发波控制环换到 src 侧 —— remaining 是我的
  // **行**(scounts,本地权威,零通信)。dst 环仍负责收账/FIN/完成判据,
  // 只是不再发波(pump_protocol 的 dst 触发块按 src_mode 旁路)。
  c->srcc = SrcCtrl();
  if (c->src_mode) {
    c->srcc.active = true;
    for (int t = 0; t < n; t++)
      if (t != me) c->srcc.remaining[t] = c->scounts[t];
    static bool logged = false;   // 一次性:确认相位真的从 L4 传到了引擎
    if (!logged) {
      logged = true;
      MLOG_INFO("dispatch 相 src 驱动生效(首个 coll=%d,MOSAIC_SRC_DRIVEN=0 可关)", c->coll_id);
    }
  }

  // 机内自拷贝(s == d)直接做掉,不进调度。
  g_ev_phase = 17;
  // **坑 #51**:自拷贝读的是 **sendbuf**,必须等调用方的生产者 kernel 写完。
  // 阻塞档走 stream_h2d(提交时挂了 `WaitEvent(input_ev)`,顺序由构造保证);
  // 流序档(gated)禁止流操作 ⇒ 交消费者核,但要先等 input_ready 标志
  // (cudaLaunchHostFunc 回调写 pinned,= 生产者已完成),由 try_self_copy 补推。
  c->self_copy_pending = false;
  c->self_kq_pending = 0;                 // 防御复位(正常路径 coll 末即为 0)
  if (c->scounts[me]) {
    if (c->kdesc_active) {
      c->self_copy_pending = true;      // try_self_copy 立即/稍后推(输入门在核内)
      // 下面直接试推一次,环满才留到事件循环重试
    } else {
      copy_async(c, (uint8_t*)c->recvbuf + c->rdispls[me],
                 (const uint8_t*)c->sendbuf + c->sdispls[me], c->scounts[me],
                 cudaMemcpyDeviceToDevice, c->stream_h2d);
    }
  }

  g_ev_phase = 18;
  // ── PC(pair_color)三重门:auto = demand 对角线 CoV<0.1 → K=n,否则 0
  // (模拟器 set_pair_color_auto 逐行,mosaic_schedule.h:348-367)
  // P1 SSP:形状门(PC/PACE)读 D —— 正常 = 本次矩阵;SSP = 上次矩阵(有界陈旧)
  const uint64_t* D = (c->pace_use_prev && c->demand_prev.size() == c->demand.size())
                          ? c->demand_prev.data() : c->demand.data();
  // ── S2 自动选档(MOSAIC_AUTO_TIER=1,默认关):全局矩阵在手 → 形态特征
  // → 本 collective 生效 W/C。映射来自 T5 冠军表(32 GPU 实测):
  //   热点(t_max≥2×median)→ C=3(RX 摊布);热点且中等报文 → 再 W=3;
  //   非热点且小报文(每对均值 ≤1MB)→ W=2。只向下收(≤cfg),
  //   compute_intent 的读集超集性质不破。分布式视图+动态调整的正统延伸。
  c->eff_waves = 0;
  c->eff_corridors = 0;
  c->tier_blind = false;
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && opt_auto_tier()) {
    uint64_t total = 0, cmax = 0;
    std::vector<uint64_t> col(n, 0);
    for (int s2 = 0; s2 < n; s2++)
      for (int d2 = 0; d2 < n; d2++) {
        if (s2 == d2) continue;
        uint64_t v = D[(size_t)s2 * n + d2];
        col[d2] += v;
        total += v;
      }
    std::vector<uint64_t> sc = col;
    std::sort(sc.begin(), sc.end());
    uint64_t cmed = sc[n / 2];
    for (uint64_t v : col) cmax = std::max(cmax, v);
    uint64_t mean_pair = total / (uint64_t)(n * (n - 1));
    const bool hot = cmed > 0 && cmax >= 2 * cmed;
    if (hot) {
      c->eff_corridors = std::min(3, c->cfg.corridors);
      // W3 判据用**每对**字节(T5 冠军:hotdst2M=W3,hotdst8M=W 默认)
      if (cmax / (uint64_t)(n - 1) < (uint64_t)4 << 20)
        c->eff_waves = std::min(3, c->cfg.waves);
    } else {
      // 非热点:决策转盲档(32 rank 实测知情档对 uniform/persrc 反噬 2×+,
      // T5 首轮同款结论);小报文再压波数
      c->tier_blind = true;
      if (mean_pair <= (uint64_t)1 << 20) c->eff_waves = std::min(2, c->cfg.waves);
    }
    if (c->eff_waves || c->eff_corridors)
      MLOG_DBG("auto_tier coll=%d hot=%d mean_pair=%llu → W=%d C=%d", c->coll_id,
               (int)hot, (unsigned long long)mean_pair, eff_W(c), eff_C(c));
  }

  // ── 波数按「目标碎片大小」反推(MOSAIC_WAVE_FRAG_KB>0 生效,默认 0 = 旧行为)。
  //
  // **实测动机**(本机 2 机 × 4 vGPU = 8 rank,不限速,mosaic 调度器,2026-08-17):
  //   msg/pair    W=1        W=2         W=3
  //   256 KB    1.014ms   1.729(+70%)  2.053(+100%)
  //     1 MB    2.171ms   2.242(+3.3%) 2.353(+8.4%)
  //     8 MB   14.352ms  14.434(+0.6%) 14.458(+0.7%)
  // 多分一波的代价随报文增大而急剧衰减 ⇒ **决定成本的不是波数本身,而是
  // 分完之后每个碎片还剩多大**:碎片 = 每对字节 ÷ (波数 × 走廊数),碎片一小,
  // 每片的固定开销(发起 + 发现 + 交付 + credit)就摊不薄。
  //
  // **规则(凹函数,不是线性)**:
  //     W = clamp( round( sqrt( 每对字节 ÷ (目标碎片 × 生效走廊数) ) ), 1, cfg.waves )
  // 为什么开方而不是正比(用户 2026-08-17 指出"越大越难增加波数",与模型一致):
  // 分波的**收益**是流水化多跳链,时间 ≈ T·(1 + (depth−1)/W) —— 边际收益按 1/W² 衰减;
  // 分波的**成本**是每波一次的同步/触发/广播,随 W **线性**增长(≈ W·c)。
  // 两者相加求极值得 **W* = sqrt(T·(depth−1)/c)** ⇒ 最优波数随字节数**开方**增长。
  //
  // **⚠ 调参必读(2026-08-17 审阅纠错,照字面理解会把旋钮调错约 5 倍)**:
  // 1. `MOSAIC_WAVE_FRAG_KB` **不是"目标碎片大小"**,它是**尺度常数 f**。
  //    开方律下实际碎片 = 每对字节/(W·C) = **sqrt(P·f/C)**,并不恒等于 f。
  // 2. **风险是不对称的:少分波远比多分波贵。** T5 实测(log_raven,hotdst8M C3+知情)
  //    `W2=143.7 / W6=95.5 / W9=96.2 / W12=101.1` —— 分少了(W2)比最优差 **50%**,
  //    分多了(W12)只差 **6%**。⇒ **宁高勿低**,拿不准就把 f 往小取(f 小 ⇒ W 大)。
  // 3. 要让本律命中 T5 的 W6-9 平台区,f 需取 **≈50-60KB**
  //    (f = P/(W²·C) = 8MB/(49·3) ≈ 56KB);该值在本机端也自洽(256KB,C=6 → W=1)。
  //    **推荐初值 48-64KB**,不是 256KB。
  // 4. **`cfg.waves` 会静默封顶**(下面的 clamp):默认 6,要够到 W9 必须**同时**抬
  //    `MOSAIC_WAVES`,否则旋钮看起来"不起作用"。
  // 5. f 不能跨档直搬:`eff_C` 在 AUTO_TIER 关时 = `cfg.corridors`(默认 6),
  //    而每对**实际**可用走廊是 1-4(Raven C3 档是 3)⇒ 同一个 f 在不同 C 档下
  //    差 sqrt 倍。**本机标的 f 必须在 Raven 重标。**
  // 6. 与 `MOSAIC_AUTO_TIER` 同开时:**C 归 AUTO_TIER、W 归本规则**(本块在其后,
  //    无条件覆盖 eff_waves);AUTO_TIER 的 W 判据被丢弃,但它设的 tier_blind 仍生效。
  //    一次只动一个变量的话,建议两者不要同开。
  // 驱动量用**最热目的地的每对字节**(cmax/(n-1)),因为 makespan 由最忙的那条边定;
  // uniform 时它退化成均值,hotdst 时自动放大 —— 与 T5 冠军表方向一致
  // (Raven 32GPU:hotdst8M 最优 W6-9、hotdst2M 最优 W3,对应碎片都落在几百 KB)。
  //
  // **口径边界(如实,2026-08-17 二次修正)**:本机数据只能定**小报文端**(W=1),
  // 大报文端的 f 必须在 Raven 标定。**但原因不是我最初写的"本机没有中转深度"** ——
  // 开了 rail 档(75% 跨机对被逼换轨、depth=2)后重测,W=1 **依然在所有尺度最优**:
  //   rail 档 256KB/pair:W1 1469.6 / W2 2359.3(+61%)/ W3 2774.1(+89%)/ W4 3343.7(+128%)
  //   rail 档   4MB/pair:W1 7711.4 / W2 7726.8(+0.2%)/ W3 7818.8(+1.4%)/ W4 7781.9(+0.9%)
  // ⇒ **"分波 = 流水化多跳链"这个模型对本实现不成立**:切片本身已经在流水,
  //   波不是流水单元,而是**重规划单元**(每波一次注水/策略广播/同步)。
  //   所以在本机这种"负载稳定、无需重规划"的场景里,多分波纯粹是净成本。
  // 那 T5 在 Raven 上测到的 W6-9 最优(hotdst8M)又是怎么来的?**目前无定论**,
  // 合理猜测是 ①波级重规划让账本能跟上热点漂移(= MOSAIC 的立项点本身)
  // ②中转池水位在深中转下需要分批。**这条必须在 Raven 上单独查清**,
  // 在查清之前,开方律的**形状**只有小报文端有实证支撑,大报文端属外推。
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && opt_wave_frag_kb() > 0) {
    uint64_t cmax2 = 0;
    for (int d2 = 0; d2 < n; d2++) {
      uint64_t colb = 0;
      for (int s2 = 0; s2 < n; s2++)
        if (s2 != d2) colb += D[(size_t)s2 * n + d2];
      cmax2 = std::max(cmax2, colb);
    }
    const uint64_t per_pair = n > 1 ? cmax2 / (uint64_t)(n - 1) : cmax2;
    const uint64_t frag = (uint64_t)opt_wave_frag_kb() << 10;
    const int cc = eff_C(c) > 0 ? eff_C(c) : 1;
    const double ratio = (double)per_pair / (double)(frag * (uint64_t)cc);
    long w = (long)(std::sqrt(ratio > 0 ? ratio : 0) + 0.5);   // 开方后四舍五入
    if (w < 1) w = 1;
    if (w > (long)c->cfg.waves) w = (long)c->cfg.waves;
    c->eff_waves = (int)w;
    MLOG_DBG("wave_frag coll=%d per_pair=%llu frag=%lluKB C=%d ratio=%.2f → W=%d",
             c->coll_id, (unsigned long long)per_pair,
             (unsigned long long)(frag >> 10), cc, ratio, eff_W(c));
  }
  // ── W 形态自适应 v3(2026-08-21;AUTO_TIER/WAVE_FRAG 显式设时优先):
  // 非热点列 → W=1(多波在直发形态是纯税:8M uniform W1 打平 NCCL vs W6 5×;
  // 小报文 W 是 RTT 叠加)。热点列 → **在线自适应**(设计 File/design-
  // decision-layer-0821.md §1):W 地形非单调且有共振(α1:W2=20.0ms、W3=56、
  // W4=29、W6=108,NCCL 37.2),静态公式不可靠;MoE 同形状每 iter 重复 24 次,
  // 在线样本白送。按形状签名记 EMA,从 cfg.waves 向下探测,劣化(>5%)档记入
  // bad_mask 永久绕开;探索下界 = 1(QP 悬崖由传输层信用反压负责,见下方
  // warm_wsafe 注释)。全部输入为运行时量 + 本地观测(G.2 合规,零集群常数)。
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && c->eff_waves == 0) {
    uint64_t wmax = 0, wmax_x = 0;
    std::vector<uint64_t> wcol(n, 0), wcolx(n, 0);
    for (int d2 = 0; d2 < n; d2++) {
      uint64_t col = 0, colx = 0;
      for (int s2 = 0; s2 < n; s2++) {
        if (s2 == d2) continue;
        uint64_t v = D[(size_t)s2 * n + d2];
        col += v;
        if (!c->topo.same_host(s2, d2)) colx += v;
      }
      wcol[d2] = col;
      wcolx[d2] = colx;
      wmax = std::max(wmax, col);
      wmax_x = std::max(wmax_x, colx);
    }
    // 信用反压 W0 覆盖(设计 §3.4):热判据 = 单列热 **或** 所在节点聚合热。
    // 单列:跨机列和 ≥ 2× 跨机列中位;节点:成员列跨机和 ≥ 2× 节点中位 ——
    // 排水瓶颈是节点级(HCA/host 通路共享),热节点上的小列(如 α1 的
    // d13=44MB)不钳就是拥塞树的漏洞:29438267-2202 实测 α1 W1 边缘随机死,
    // 四节点 port_xmit_wait 全暴涨(拥塞树扩散),热节点仅 d13 未钳。
    // 输入全部来自已交换的 demand 矩阵(SSP 残缺时列和偏小 → 判不出热 →
    // 退化全窗,只剩 retire 闭环保护,如实标注于设计 §3.4)。
    {
      std::vector<uint64_t> sx = wcolx;
      std::sort(sx.begin(), sx.end());
      const uint64_t medx = sx[n / 2];
      const int G = c->cfg.gpus_per_host, H = n / G;
      std::vector<uint64_t> ncx(H, 0);
      for (int d2 = 0; d2 < n; d2++) ncx[c->topo.host(d2)] += wcolx[d2];
      std::vector<uint64_t> ns = ncx;
      std::sort(ns.begin(), ns.end());
      const uint64_t nmed = ns[H / 2];
      if (medx > 0)
        for (int d2 = 0; d2 < n; d2++)
          if (d2 != me && !c->topo.same_host(d2, me) &&
              (wcolx[d2] >= 2 * medx ||
               (nmed > 0 && ncx[c->topo.host(d2)] >= 2 * nmed)))
            c->cr_w0[d2] = credit_hot_w0();
    }
    std::vector<uint64_t> ws = wcol;
    std::sort(ws.begin(), ws.end());
    uint64_t wmed = ws[n / 2];
    c->warm_cur = nullptr;
    if (!(wmed > 0 && wmax >= 2 * wmed)) {
      c->eff_waves = 1;                          // 非热点:直发单波,不进搜索
    } else {
      // 探索下界 = 1:QP 悬崖(E1025 级联死)由传输层信用反压负责 ——
      // credit 只在接收端 rx_retire 铸造 + demand 感知 W0 突发钳(节点聚合
      // 热判据),进水速率被排水速率闭环钳住,悬崖构造性不发生
      // (File/design-credit-backpressure-0821.md;标定 = 29438267 实测
      // α2 W=1 直发 59.2ms / α1 W=1 三连稳 39.6-39.8ms,零 E1025 全校验)。
      // 波数只有性能语义,由在线 arm 全域探索。
      c->warm_wsafe = 1;
      // 形状签名:列和 log2 半档分桶 + 排序 hash(容忍 gating 抖动)
      std::vector<uint8_t> bk(n);
      for (int d2 = 0; d2 < n; d2++) {
        uint64_t v = wcol[d2] >> 16;             // 64KB 粒度
        int b = 0;
        while (v) { b++; v >>= 1; }
        bk[d2] = (uint8_t)b;
      }
      std::sort(bk.begin(), bk.end());
      uint64_t key = 1469598103934665603ull ^ (uint64_t)n;
      for (uint8_t b : bk) { key ^= b; key *= 1099511628211ull; }
      // arm 命中 / LRU 换出
      Comm::WaveArm* arm = nullptr;
      for (auto& a : c->warm) if (a.key == key) { arm = &a; break; }
      if (!arm) {
        arm = &c->warm[0];
        for (auto& a : c->warm) if (a.last_ns < arm->last_ns) arm = &a;
        *arm = Comm::WaveArm();
        arm->key = key;
        arm->cur_w = c->cfg.waves;               // 安全起步
        arm->best_w = c->cfg.waves;
      }
      arm->last_ns = now_ns();
      c->eff_waves = arm->cur_w;
      c->warm_cur = arm;
    }
  }
  c->pc_k = 0;
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && c->cfg.pair_color != 0) {
    if (c->cfg.pair_color > 0) c->pc_k = c->cfg.pair_color;
    else {
      double mean = 0;
      std::vector<double> diag(n, 0.0);
      for (int cc = 1; cc < n; cc++) {
        for (int s = 0; s < n; s++) diag[cc] += (double)D[(size_t)s * n + (s + cc) % n];
        mean += diag[cc];
      }
      mean /= (n - 1);
      double var = 0;
      for (int cc = 1; cc < n; cc++) var += (diag[cc] - mean) * (diag[cc] - mean);
      double cov = mean > 0 ? std::sqrt(var / (n - 1)) / mean : 0;
      c->pc_k = (cov < 0.1) ? n : 0;
      MLOG_DBG("pair_color_auto diag_cov %.3f K %d", cov, c->pc_k);
    }
  }
  // ── PACE(cold_pace)三门:k>0、taper 门(per-rank spine 份额 < nic → off)、
  // skew 门(t_max ≥ 2×median);冷 dst 最多领先 hot k-1 波
  // (模拟器 set_cold_pace / ingress_bounds / skew_gate 逐行)
  c->pace_hot = -1;
  c->pace_cold.assign(n, 0);
  c->peer_wave.assign(n, 0);
  c->peer_fin.assign(n, 0);
  // (src 相不开 PACE:PACE 是 dst 发波节奏的协调;src 相的公平性由
  //  "行全等 ⇒ 决策者规模相同"结构性成立,不需要节奏协调)
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && !c->src_mode && c->cfg.cold_pace > 0 &&
      !c->tier_blind && c->cfg.nv_gbps > 0 && c->cfg.nic_gbps > 0) {
    double spine_share = c->cfg.spine_gbps > 0 ? c->cfg.spine_gbps / c->cfg.gpus_per_host : 0;
    if (!(spine_share > 0 && spine_share < c->cfg.nic_gbps)) {
      std::vector<double> tt(n, 0);
      double t_max = 0;
      for (int dd = 0; dd < n; dd++) {
        double same = 0, cross = 0;
        for (int s = 0; s < n; s++) {
          if (s == dd) continue;
          if (c->topo.same_host(s, dd)) same += (double)D[(size_t)s * n + dd];
          else cross += (double)D[(size_t)s * n + dd];
        }
        tt[dd] = std::max(same / c->cfg.nv_gbps,
                          (same + cross) / (c->cfg.nv_gbps + c->cfg.nic_gbps));
        t_max = std::max(t_max, tt[dd]);
      }
      std::vector<double> st_t = tt;
      std::sort(st_t.begin(), st_t.end());
      double median = st_t[n / 2];
      if (t_max > 0 && median > 0 && t_max >= 2.0 * median) {
        int ncold = 0;
        for (int dd = 0; dd < n; dd++) {
          if (tt[dd] >= t_max) c->pace_hot = dd;
          c->pace_cold[dd] = (tt[dd] * 2.0 <= t_max) ? 1 : 0;
          ncold += c->pace_cold[dd];
        }
        MLOG_DBG("cold_pace on k %d hot %d cold_dsts %d", c->cfg.cold_pace, c->pace_hot, ncold);
      }
    }
  }

  c->coll_start_ns = now_ns();
  MTRACE(TE_COLL_BEGIN, (uint32_t)c->coll_id, c->total_send, c->total_recv_expect);
  es->coll_started = true;

  // 早到的 CT_FIN 收编(hint 下对端可能在我 begin 前就收满;空行 pair 场景)
  {
    int slot = c->coll_id & (PEND_SLOTS - 1);
    if (es->fin_early_cid[slot] != -1 &&
        (es->fin_early_cid[slot] & 0xFFFF) == (c->coll_id & 0xFFFF)) {
      for (int p = 0; p < n; p++)
        if (es->fin_early_who[slot][p]) { es->fin_got++; c->peer_fin[p] = 1; }
    }
    es->fin_early_cid[slot] = -1;
    memset(es->fin_early_who[slot], 0, sizeof(es->fin_early_who[slot]));
  }

  // O4 波折叠:所有 (src,me) pair 都只有 ≤1 条走廊时,波机制没有任何自由度
  // (逐波与单波的分配逐字节恒等),直接单波发完,省 W-1 轮触发/错峰/通道 B 延迟。
  // world=2/G=1 的训练形态命中;n=8 微基准与 Raven 多走廊形态不触发。
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && opt_wave_collapse() && eff_W(c) > 1) {
    bool degen = true;
    int cl[NCORR];
    // 折叠谓词按决策方向取 pair 集:dst 相 = 入向列,src 相 = 出向行
    for (int p = 0; p < n && degen; p++) {
      if (p == me) continue;
      if (c->src_mode ? c->srcc.remaining[p] == 0 : c->dst.remaining[p] == 0) continue;
      const int ss = c->src_mode ? me : p, dd = c->src_mode ? p : me;
      if (c->topo.corridors_of(ss, dd, eff_C(c), cl) > 1) degen = false;
    }
    if (degen && (c->src_mode ? c->total_send > 0 : c->dst.total_expect > 0)) {
      // waves_left=1 → 全配额一波发完
      if (c->src_mode) c->srcc.wave_idx = eff_W(c) - 1;
      else c->dst.wave_idx = eff_W(c) - 1;
      MLOG_DBG("wave collapse:无路径自由度,单波发完(coll=%d)", c->coll_id);
    }
  }

  g_ev_phase = 19;
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC) {
    if (c->src_mode) emit_wave_src(c);
    else emit_wave(c);
  } else plan_static(c);

  // 补放"比我先到"的策略广播(见 on_ctrl_msg 的 CT_POLICY 早到分支)。
  // 只放属于本 collective 的(cid8 匹配);更晚 collective 的继续留着。
  {
    std::vector<EarlyPolicy> keep;
    for (auto& ep : es->early_policy) {
      if (ep.cid8 == (uint8_t)(c->coll_id & 0xFF)) {
        if (ep.from_src) apply_policy_src(c, ep.peer, ep.ents.data(), (int)ep.ents.size());
        else apply_policy(c, ep.peer, ep.wave, ep.ents.data(), (int)ep.ents.size());
      } else
        keep.push_back(std::move(ep));
    }
    es->early_policy.swap(keep);
  }

}

static void coll_try_finish(Comm* c) {
  if (c->coll_state.load() != 1) return;
  auto* es = ES(c);
  if (!es->coll_started) return;
  // 收满 = 本 rank 的 completion(makespan 由 mosaicStatsSync 取全局 max)。
  // 还要求自己的字节都已离开 sendbuf(D2H 完成),调用方才能安全复用发送缓冲。
  bool recv_done = c->dst.total_recv >= c->total_recv_expect;
  bool send_done = c->posted_send >= c->total_send &&
                   es->ready_nv.empty() && es->ready_nic.empty() && es->tx.empty();
  if (!(recv_done && send_done)) return;
  // kdesc 档:交付走消费者核,完成还要求 kq 排干(纯内存判定)+ 自拷贝已推
  // + CE staging 在途队列排干(WR 已发出,sendbuf/中转字节均已离手)。
  if (c->self_copy_pending) return;
  if (c->kdesc_active && c->kdesc_consumed < c->kdesc_pushed) return;
  // 中转义务门(local-0820-gated-bench bug B):走廊可用的 coll,自己的账
  // 清了还不够 —— 我可能仍要为别的 (src,dst) 对中转,而中转片何时到、
  // 是否会来,本 rank 无法预知(路径由各 dst 的波策略动态定)。
  // FIN(d) ⇔ d 已收齐 ⇔ 经我中转给 d 的字节已全部过手,且 d 不会再发
  // 路由经我的波 ⇒ 只需等**承诺位=1 的 dst**(corridor_intent_bit;=0 的
  // dst 已承诺全程直发,不可能给我中转义务)。位随 CT_DEMAND 行到,所以先
  // 等行收齐(demand_got)再判 —— SSP 的快速启动不受影响,只是收尾多等
  // 几条本来就在路上的小报文。承诺全 0(共享 HCA 拓扑 uniform)= 零耦合。
  // 直发调度(direct/fast)、单机形态、C=1 无中转,不受此门。
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && c->cfg.corridors > 1 &&
      c->cfg.nranks > c->cfg.gpus_per_host) {
    if (es->demand_got < c->cfg.nranks - 1) return;
    if (c->src_mode) {
      // src 相:中转义务由各 **src** 的计划产生(承诺位随行捎带、按出向行
      // 计算)。经我中转的片以某 dst 为终点,FIN(d) ⇔ 不再有去 d 的字节 ⇒
      // 只要有任一 src(含我)承诺过非直发,就等**全体** FIN。比 dst 相的
      // 按-dst 精确门保守(dst 相下"谁会借我"可以从该 dst 的承诺位读出,
      // src 相下借道关系分散在 n 个 src 的计划里,端侧无法逐 dst 判定)。
      bool any = c->my_corr_intent != 0;
      if (!any)
        for (int p = 0; p < c->cfg.nranks && !any; p++)
          if (p != c->cfg.rank && !c->peer_corr_intent.empty() &&
              c->peer_corr_intent[p]) any = true;
      if (any)
        for (int p = 0; p < c->cfg.nranks; p++)
          if (p != c->cfg.rank && !c->peer_fin[p]) return;
    } else {
      for (int p = 0; p < c->cfg.nranks; p++)
        if (p != c->cfg.rank && !c->peer_corr_intent.empty() &&
            c->peer_corr_intent[p] && !c->peer_fin[p]) return;
    }
  }
  // **坑 #51**:零流档也不短路 h2d 检查 —— 自拷贝、以及 kdesc_resolve_src
  // 失败时回退的交付拷贝**仍在 h2d 流上**,短路掉就可能在它们落盘前宣布完成,
  // torch 随即读走半成品的 recvbuf(现象:训练 loss 轻微漂且逐次不同)。
  // 流序窗口(gated)零 CUDA 调用:自拷贝与交付全在 kdesc,h2d 流未被使用,
  // 免查询;阻塞档维持 h2d streamQuery 语义(交付/自拷贝在 h2d 流上)。
  const bool h2d_ok = !c->cfg.use_gpu || c->gated ||
                      cudaStreamQuery((cudaStream_t)c->stream_h2d) == cudaSuccess;
  if (h2d_ok) {
    c->coll_end_ns = now_ns();
    MTRACE(TE_COLL_END, (uint32_t)c->coll_id, c->dst.total_recv, 0);
    c->st.completion_us = (c->coll_end_ns - c->coll_start_ns) / 1000.0;
    // W 在线自适应:喂 completion,8 样本一窗;劣化(>5%)档记 bad_mask 永久
    // 绕开(共振档在线消化),否则向下探测;下界 = warm_wsafe(恒 1,悬崖
    // 保护在传输层信用反压)。
    if (c->warm_cur) {
      Comm::WaveArm& a = *c->warm_cur;
      const double ms = c->st.completion_us / 1000.0;
      a.ema_ms = a.samples ? 0.7 * a.ema_ms + 0.3 * ms : ms;
      a.samples++;
      if (a.samples >= 8) {
        if (a.best_ms <= 0 || a.ema_ms < a.best_ms * 0.98) {
          a.best_ms = a.ema_ms;
          a.best_w = a.cur_w;
        }
        const bool worse = a.best_ms > 0 && a.ema_ms > a.best_ms * 1.05;
        if (worse && a.cur_w != a.best_w) {
          if (a.cur_w >= 0 && a.cur_w < 16) a.bad_mask |= (uint16_t)(1u << a.cur_w);
          a.cur_w = a.best_w;
        } else {
          int nxt = a.cur_w - 1;
          while (nxt >= c->warm_wsafe && nxt < 16 && (a.bad_mask & (1u << nxt))) nxt--;
          if (nxt >= c->warm_wsafe) a.cur_w = nxt;
        }
        MLOG_DBG("W-arm key=%08llx ema=%.1fms best=W%d(%.1f) bad=%03x → W%d(safe≥%d)",
                 (unsigned long long)(a.key & 0xFFFFFFFF), a.ema_ms, a.best_w,
                 a.best_ms, a.bad_mask, a.cur_w, c->warm_wsafe);
        a.samples = 0;
        a.ema_ms = 0;
      }
      c->warm_cur = nullptr;
    }
    for (int k = 0; k < RES_KINDS; k++) c->st.res_passed[k] = c->buck[k].passed - es->base_passed[k];
    c->dst.active = false;
    c->srcc.active = false;
    // 为下一次 collective 归零交付计数(而不是在下一次开始时清 —— 见 coll_begin)
    c->dst.total_recv = 0;
    memset(c->dst.recv_by_src, 0, sizeof(c->dst.recv_by_src));
    c->dst.wave_delivered = 0;
    // O2/P1:快照本次矩阵供下次 hint 推导与 SSP 的 PACE/PC(拷 32KB@n=64)。
    // SSP 下行未收齐 → 矩阵残缺:PACE 用还凑合(prev_valid),hint 的恒等推导
    // 不行(prev_exact=false 让 hint 自动降级,防 E4005 误杀)。
    if (opt_demand_hint() || opt_demand_ssp()) {
      const bool complete = !c->ssp_active || es->demand_got >= c->cfg.nranks - 1;
      c->demand_prev = c->demand;
      c->row_bytes_prev = c->row_bytes_cur;
      c->demand_prev_valid = true;
      c->demand_prev_exact = complete && c->row_bytes_cur > 0;
    }
    es->coll_started = false;
    es->demand_sent = false;
    es->demand_got = 0;
    kdesc_release_gate(c);   // kdesc 档:写 stop,消费者核排干即退、镜像核收到退出信号
    // 流序完成:开门(pure pinned 写;门核见 done_flag ≥ coll_id 即退,
    // caller 流后续算子随之放行)
    if (c->done_flag_host)
      __atomic_store_n((uint32_t*)c->done_flag_host, (uint32_t)c->coll_id, __ATOMIC_RELEASE);
    c->coll_state.store(2);
    // O0 相位分解:submit→begin = demand 交换(含 hint 免等的效果),
    // submit→首发 = 协议启动全程,总时长含 app 侧的提交
    if (opt_phase_log() && c->ph_submit_ns) {
      // level 1(warn 档):MOSAIC_PHASE=1 是显式请求,不该再要求调 MOSAIC_LOG
      int evph = 0;                    // O0b:察觉窗口内单次最耗时的泵
      for (int i = 1; i < 16; i++)
        if (c->ev_seg_snap[i] > c->ev_seg_snap[evph]) evph = i;
      mlog(1, "PHASE coll=%d notice_us=%.1f bcast_us=%.1f demand_us=%.1f first_post_us=%.1f total_us=%.1f "
              "waves=%llu plan_us=%.1f dm=%d ssp=%d evmax=%d:%.1f",
           c->coll_id,
           c->ph_notice_snap >= c->ph_submit_snap ? (c->ph_notice_snap - c->ph_submit_snap) / 1e3 : -1.0,
           c->ph_bcast_snap >= c->ph_submit_snap ? (c->ph_bcast_snap - c->ph_submit_snap) / 1e3 : -1.0,
           (c->ph_begin_ns - c->ph_submit_snap) / 1e3,
           c->ph_first_post_ns ? (c->ph_first_post_ns - c->ph_submit_snap) / 1e3 : -1.0,
           (c->coll_end_ns - c->ph_submit_snap) / 1e3,
           (unsigned long long)c->st.waves_emitted, c->st.plan_us_total,
           c->demand_mode, (int)c->pace_use_prev,
           evph, c->ev_seg_snap[evph] / 1e3);
    }
    c->ssp_active = false;          // 在 PHASE 打印之后复位(打印要读 ssp 标志)
    c->pace_use_prev = false;
    if (g_log_level >= 3) {
      char buf[512]; int o = 0;
      for (int s = 0; s < c->cfg.nranks; s++)
        o += snprintf(buf + o, sizeof(buf) - o, "%llu/%llu ",
                      (unsigned long long)c->dst.recv_by_src[s],
                      (unsigned long long)c->demand[(size_t)s * c->cfg.nranks + c->cfg.rank]);
      MLOG_DBG("collective %d 完成 %.1fus 逐src收/应收: %s", c->coll_id, c->st.completion_us, buf);
    }
  }
}

// 应用一份通道 B 策略:①更新账本里该发布者名下的 charges ②执行属于自己的条目
static void apply_policy(Comm* c, int peer, uint16_t wave, const PolicyEnt* e, int ne) {
  auto* es = ES(c);
  for (int i = 0; i < ne; i++) {
    // 账本视图:把这一波的 charges 累加到该发布者名下(可靠通道保证不丢不
    // 乱序,本地累加与"发布者携带累计值"逐字节等价,省掉 O(nres) 报文)。
    // 全局广播档(MOSAIC_POLICY_GBCAST=1)下 ne 覆盖该 dst 的**全部**条目,
    // peer_acc[发布者] 因此是完整视图 —— 含借道本 rank 的中转跳。
    Chain ch;
    if (c->topo.chain_of(e[i].src, peer, e[i].corr, &ch)) {
      int res[16];
      int nr = c->topo.chain_res(ch, res);
      for (int k = 0; k < nr; k++) es->peer_acc[peer][res[k]] += e[i].len;
    }
    if (e[i].src == c->cfg.rank) emit_range(c, peer, e[i].corr, e[i].off, e[i].len, wave);
  }
  c->ledger.set_peer_charge(peer, es->peer_acc[peer].data(), c->topo.nres());
}

// 应用一份 src 波策略(CT_POLICY_SRC):**只记账,永不执行** —— 数据由发布者
// 自己发出。条目端点 = dst,链 = chain_of(发布者, ent.src, ent.corr)。
static void apply_policy_src(Comm* c, int peer, const PolicyEnt* e, int ne) {
  auto* es = ES(c);
  for (int i = 0; i < ne; i++) {
    Chain ch;
    if (c->topo.chain_of(peer, e[i].src, e[i].corr, &ch)) {
      int res[16];
      int nr = c->topo.chain_res(ch, res);
      for (int k = 0; k < nr; k++) es->peer_acc[peer][res[k]] += e[i].len;
    }
  }
  c->ledger.set_peer_charge(peer, es->peer_acc[peer].data(), c->topo.nres());
}

// ─────────────────────────────────────────────────────────── 控制面回调
void on_ctrl_msg(Comm* c, int peer, const CtrlHdr& h, const void* body) {
  auto* es = ES(c);
  const int n = c->cfg.nranks;
  switch (h.type) {
    case CT_BARRIER:
      { int want = (int)h.aux + 1;
        int cur = c->barrier_seen[peer].load();
        while (cur < want && !c->barrier_seen[peer].compare_exchange_weak(cur, want)) {} }
      break;
    case CT_DEMAND: {
      const uint64_t* row = (const uint64_t*)body;
      int cid = (int)h.aux;
      // P2:提取捎带的读集意图(该 dst 对"我"的那个字节;最新胜出,无 cid ——
      // collective 边界的 ≤1 拍陈旧已在 D17 声明)
      if (h.len >= sizeof(CtrlHdr) + sizeof(uint64_t) * n + n &&
          (size_t)c->cfg.rank < (size_t)n && !c->peer_intent.empty())
        c->peer_intent[peer] = ((const uint8_t*)body)[sizeof(uint64_t) * n + c->cfg.rank];
      // 走廊承诺位(行尾 1B;中转义务门读它,见 coll_try_finish)
      const uint8_t cintent =
          h.len >= sizeof(CtrlHdr) + sizeof(uint64_t) * n + n + 1
              ? ((const uint8_t*)body)[sizeof(uint64_t) * n + n] : 0;
      if (cid == c->coll_id && es->demand_sent && !es->coll_started) {
        memcpy(&c->demand[(size_t)peer * n], row, sizeof(uint64_t) * n);
        if (!c->peer_corr_intent.empty()) c->peer_corr_intent[peer] = cintent;
        if (!c->peer_row_seen.empty()) c->peer_row_seen[peer] = 1;
        es->demand_got++;
      } else if (cid == c->coll_id && es->coll_started &&
                 (c->ssp_active || c->src_mode)) {
        // P1 SSP / src 相:collective 已开跑,行到了就后台刷新(供结束时快照
        // → 下次 PACE/hint 用;本次注水只读本地列,不受影响)。src 相下这里
        // 同时是**行门开启点**:pump_protocol 的补发路径据此追齐迟到 dst。
        memcpy(&c->demand[(size_t)peer * n], row, sizeof(uint64_t) * n);
        if (!c->peer_corr_intent.empty()) c->peer_corr_intent[peer] = cintent;
        if (!c->peer_row_seen.empty()) c->peer_row_seen[peer] = 1;
        es->demand_got++;
      } else {                                   // 提前到货,先寄存
        int si = cid & (PEND_SLOTS - 1);
        auto& slot = es->pend_rows[si];
        if (es->pend_cid[si] != cid) {           // 槽换主:清掉陈旧 collective 的行
          // 只有**更旧**的代际才该被顶掉。顶掉 ≥ 当前 coll_id 的槽 = 丢弃仍会被
          // 消费的行 ⇒ 静默死锁。加深槽数已让它极难发生,真发生了必须吼出来。
          if (es->pend_cid[si] >= c->coll_id)
            MLOG_ERR("E5023 demand pend 槽 %d 被顶掉(存 cid=%d 当前=%d 来 cid=%d)"
                     " —— 漂移超过 %d 个 collective,请加大 PEND_SLOTS",
                     si, es->pend_cid[si], c->coll_id, cid, PEND_SLOTS);
          slot.assign((size_t)n * n, 0);
          std::fill(es->pend_seen[si].begin(), es->pend_seen[si].end(), 0);
          memset(es->pend_cintent[si], 0, sizeof(es->pend_cintent[si]));
          es->pend_cnt[si] = 0;
          es->pend_cid[si] = cid;
        }
        if ((int)slot.size() != n * n) slot.assign((size_t)n * n, 0);
        memcpy(&slot[(size_t)peer * n], row, sizeof(uint64_t) * n);
        es->pend_cintent[si][peer] = cintent;
        es->pend_seen[si][peer] = 1;
        es->pend_cnt[si]++;
      }
      break;
    }
    case CT_POLICY:
    case CT_POLICY_SRC: {
      int ne = (h.len - sizeof(CtrlHdr)) / sizeof(PolicyEnt);
      const PolicyEnt* e = (const PolicyEnt*)body;
      const bool from_src = h.type == CT_POLICY_SRC;
      c->st.chB_msgs++;
      const uint8_t pcid = (uint8_t)(h.aux >> 8);       // 发送方的 coll_id mod 256
      const uint16_t wave = (uint16_t)(h.aux & 0xFF);
      const bool cid_match = pcid == (uint8_t)(c->coll_id & 0xFF);
      if (cid_match && (int)wave + 1 > c->peer_wave[peer]) c->peer_wave[peer] = (int)wave + 1;  // PACE 观测
      if (!es->coll_started || !cid_match) {
        // 比本 rank 的 coll_begin 还早到,或属于对端更靠后的 collective(O2 hint
        // 下对端可以领先我一个 collective):此时 sendbuf/位移还没就位,不能执行,
        // 先寄存;coll_begin 时按 cid8 收编。src 波策略虽不执行,charges 也要
        // 等本 coll 的账本 epoch(clear_epoch)之后再入账,同槽寄存。
        EarlyPolicy ep;
        ep.peer = peer; ep.wave = wave; ep.cid8 = pcid; ep.from_src = from_src;
        ep.ents.assign(e, e + ne);
        es->early_policy.push_back(std::move(ep));
        break;
      }
      if (from_src) apply_policy_src(c, peer, e, ne);
      else apply_policy(c, peer, wave, e, ne);
      break;
    }
    case CT_TELEM: {
      // P2:三种形态,按体长区分(TelemDet 20 / 去 fab 前缀 60 / 全量 188)。
      // 未携带的字段保持旧值(有界陈旧;不发的恰是该 dst 声明不读的)。
      const size_t blen = h.len - sizeof(CtrlHdr);
      c->st.chA_msgs++;
      uint64_t seq;
      float up_g, dn_g;
      uint8_t det;
      const TelemMsg* t = nullptr;
      if (blen == sizeof(TelemDet)) {
        const TelemDet* d2 = (const TelemDet*)body;
        seq = d2->seq; up_g = d2->nic_up_gbps; dn_g = d2->nic_dn_gbps; det = d2->det_state;
      } else if (blen >= TELEM_NOFAB_BYTES) {
        t = (const TelemMsg*)body;
        seq = t->seq; up_g = t->nic_up_gbps; dn_g = t->nic_dn_gbps; det = t->det_state;
      } else break;
      if (seq > c->peer_disch_seq[peer]) {        // 单调、最新胜出
        c->peer_disch_seq[peer] = seq;
        if (t)
          for (int k = 0; k < RES_KINDS; k++)
            c->ledger.discharge[c->topo.res(peer, k)] = t->disch[k];
        // fabric 账贡献(per-rail;仅全量形态携带)
        if (t && blen == sizeof(TelemMsg) && !c->peer_fab.empty()) {
          memcpy(&c->peer_fab[((size_t)peer * 2 + 0) * 8], t->fab_up, sizeof(t->fab_up));
          memcpy(&c->peer_fab[((size_t)peer * 2 + 1) * 8], t->fab_dn, sizeof(t->fab_dn));
        }
        // 检测器传播:属主锁定 → 全员把该资源额定改为实测(报告 §11.2)。
        // det 位在所有形态里恒发 —— quorum≤2 需要全局计数。
        c->peer_det[peer] = det;
        int ru = c->topo.res(peer, RES_NIC_UP), rd = c->topo.res(peer, RES_NIC_DN);
        c->ledger.cap_Bps[ru] = (det & 1) ? (double)up_g * 1e9 / 8.0 : c->nominal_Bps[ru];
        c->ledger.cap_Bps[rd] = (det & 2) ? (double)dn_g * 1e9 / 8.0 : c->nominal_Bps[rd];
      }
      break;
    }
    case CT_STATS: {
      double v = 0;
      memcpy(&v, body, sizeof(v));
      int ep = (int)h.aux, i = ep & 3;
      if (c->stats_slot_epoch[i].load() != ep) {      // 该槽第一次见到这个 epoch
        c->stats_max[i] = v;
        c->stats_cnt[i].store(0);
        c->stats_slot_epoch[i].store(ep);
      } else if (v > c->stats_max[i]) c->stats_max[i] = v;
      c->stats_cnt[i].fetch_add(1);
      break;
    }
    case CT_AG: {
      // 与 stats 同款的 epoch 分槽,免疫"对端比我先到"
      int ep = (int)h.aux, i = ep & 3;
      size_t nb = h.len - sizeof(CtrlHdr);
      std::lock_guard<std::mutex> lk(c->ag_mu);
      if (c->ag_slot_epoch[i].load() != ep) {
        c->ag_buf.assign((size_t)c->cfg.nranks * nb, 0);
        c->ag_bytes = nb;
        c->ag_cnt[i].store(0);
        c->ag_slot_epoch[i].store(ep);
      }
      if (c->ag_buf.size() >= (size_t)(peer + 1) * nb)
        memcpy(c->ag_buf.data() + (size_t)peer * nb, body, nb);
      c->ag_cnt[i].fetch_add(1);
      break;
    }
    case CT_FIN: {  // PACE:hot 完成即放行。带 cid:当前的直接记;更晚的寄存;陈旧的丢
      uint16_t fcid = h.aux;
      if (es->coll_started && fcid == (uint16_t)(c->coll_id & 0xFFFF)) {
        if (!c->peer_fin[peer]) { es->fin_got++; c->peer_fin[peer] = 1; }
      } else if (!es->coll_started || fcid != (uint16_t)(c->coll_id & 0xFFFF)) {
        // 可能属于我尚未 begin 的 collective(对端超前);按低 2 位寄存
        int si = fcid & (PEND_SLOTS - 1);
        int want_cid = -1;
        // 只寄存"看起来是未来"的:cid mod 空间无法精确判序,保守收编到槽,
        // coll_begin 时 cid 全匹配才生效(fin_early_cid 记完整低 16 位对不上就弃)
        want_cid = fcid;
        if (es->fin_early_cid[si] != -1 && (es->fin_early_cid[si] & 0xFFFF) != want_cid) {
          memset(es->fin_early_who[si], 0, sizeof(es->fin_early_who[si]));
        }
        es->fin_early_cid[si] = want_cid;
        es->fin_early_who[si][peer] = 1;
      }
      break;
    }
    case CT_CREDIT: {  // 信用回流:绝对累计授权,取 max(乱序/双通路安全)
      if (h.len >= sizeof(CtrlHdr) + 8) {
        uint64_t cum;
        memcpy(&cum, body, 8);
        if (cum > c->cr_limit[peer]) c->cr_limit[peer] = cum;
      }
      break;
    }
    default: break;
  }
}

// ─────────────────────────────────────────────────────────── 数据面回调
void on_slice_arrived(Comm* c, const SliceHdr& h, void* payload, int shm_ring, int stage_slot,
                      int verbs_token) {
  auto* es = ES(c);
  const bool via_nic = shm_ring < 0;             // 来自 SHM 环 = 机内 NV 管
  const int me = c->cfg.rank;
  const bool final_hop = (h.hop + 1 >= h.nhops);

  // 正确性判据 = 端到端 bit-exact 校验(bench --verify)+ 训练 loss 门
  // (payload 全程在设备内存,CPU 无逐 slice 校验点)。
  c->buck[dn_kind(via_nic)].take(h.len);
  if (via_nic) {
    det_feed(c, 1, h.len);                       // 检测器:NIC↓ 到达事件
    // fabric discharge(跨轨 hop 交付点,与模拟器 lg_discharge 同点)
    int hs = h.path[h.hop];
    if (c->topo.rail(hs) != c->topo.rail(me)) {
      c->fab_own[0][c->topo.rail(hs) & 7] += h.len;   // 源 rail 的 leaf 上行段
      c->fab_own[1][c->topo.rail(me) & 7] += h.len;   // 本 rail 的 leaf 下行段
    }
  }

  if (final_hop) {
    if ((int)h.dst != me) { MLOG_ERR("E2001 终跳落错 rank(%d≠%d)", h.dst, me); return; }
    // 坑 #48 的兜底告警:代际闸门开在各收方路径的入口(pump_shm / verbs 收路),
    // 走到这里就该已经是本代。若不是,说明还有一条路径没装闸门 —— 这一片会被
    // 落进**错的一代的 recvbuf** 并记进错的账。宁可吵,也不要静默损坏。
    if ((int16_t)(h.coll - (uint16_t)(c->coll_id & 0xFFFF)) != 0)
      MLOG_ERR("E5024 代际错位交付 (r%d→r%d) 片 coll=%u 本 rank coll=%d —— 该路径缺代际闸门",
               h.src, h.dst, (unsigned)h.coll, c->coll_id);
    InflightRx r;
    r.h = h; r.via_nic = via_nic; r.final_hop = true;
    r.src_shm_ring = shm_ring; r.src_stage = stage_slot; r.src_verbs = verbs_token;
    void* gdst = (uint8_t*)c->recvbuf + c->rdispls[h.src] + h.off;
    // S3 二期:异步 collective 的交付拷贝交给消费者核(host 只写描述符;
    // verbs RX 槽在 host pinned,SM 经 UVA 读 —— NCCL 收侧同款形态)。
    // 描述符环满且来源是 SHM 环 → **defer 而非混用**
    // (shm_ring_pop 必须按序,两条完成队列交错会弹错槽)
    if (c->kdesc_active) {
      uint64_t sdev = kdesc_resolve_src(c, payload, shm_ring, verbs_token, via_nic);
      if (sdev && kdesc_push(c, sdev, gdst, h.len)) {   // 输入门在消费者核内
        MTRACE(TE_KDESC_PUSH, ((uint32_t)h.src << 16) | h.dst, h.off,
               ((uint64_t)h.len << 8) | h.hop);
        KEntry e; e.kind = 0; e.r = r;
        es->kq.push_back(e);
        return;
      }
      // 环满:defer(源槽扣着 = 天然背压;kdesc 窗口禁流回退)
      WaitRelay w{h, payload, shm_ring, stage_slot, verbs_token, via_nic};
      es->wait_relay.push_back(w);
      return;
    }
    copy_async(c, gdst, payload, h.len, cudaMemcpyHostToDevice, c->stream_h2d);
    r.ev = ev_take(c); ev_record(c, r.ev, c->stream_h2d);
    es->rx.push_back(r);
  } else {
    int slot = relay_alloc(c);
    if (slot < 0) {                              // 中转池满 → 扣着源槽等(背压)
      WaitRelay w{h, payload, shm_ring, stage_slot, verbs_token, via_nic};
      es->wait_relay.push_back(w);
      return;
    }
    InflightRx r;
    r.h = h; r.via_nic = via_nic; r.final_hop = false; r.relay_slot = slot;
    r.src_shm_ring = shm_ring; r.src_stage = stage_slot; r.src_verbs = verbs_token;
    c->st.bytes_relayed += h.len;
    if (c->kdesc_active) {
      uint64_t sdev = kdesc_resolve_src(c, payload, shm_ring, verbs_token, via_nic);
      if (sdev && kdesc_push(c, sdev, relay_ptr(c, slot), h.len)) {
        KEntry e; e.kind = 0; e.r = r;
        es->kq.push_back(e);
        return;
      }
      // 环满背压:defer(源槽扣着;流序窗口禁流回退)
      c->relay_free.push_back(slot);
      c->st.bytes_relayed -= h.len;
      WaitRelay w{h, payload, shm_ring, stage_slot, verbs_token, via_nic};
      es->wait_relay.push_back(w);
      return;
    }
    copy_async(c, relay_ptr(c, slot), payload, h.len, cudaMemcpyHostToDevice, c->stream_h2d);
    r.ev = ev_take(c); ev_record(c, r.ev, c->stream_h2d);
    es->rx.push_back(r);
  }
}

// ─────────────────────────────────────────────────────────── 事件循环各阶段
// MOSAIC_NODE_INFLIGHT_MB:发送端对**每个目的节点**的在途字节上限(容量类,
// 不改数据面走法)。默认 24MB:悬崖标定 —— α1 W2(每波对热节点 ~均摊 24MB+/
// sender)多次存活、更高聚合压强触发 E1025 级联死;调大 = 更满的管线但更接近
// RNR 风暴,调小 = 更强的 incast 保护但可能欠吞吐;在 Raven 4×4 α2 上标定。
static uint64_t node_inflight_cap() {
  static uint64_t v = 0;
  if (!v) {
    const char* e = getenv("MOSAIC_NODE_INFLIGHT_MB");
    long mb = e && *e ? atol(e) : 24;
    if (mb < 1) mb = 1;
    v = (uint64_t)mb << 20;
  }
  return v;
}
static void pump_one_queue(Comm* c, std::deque<ReadySlice>& q, bool via_nic) {
  auto* es = ES(c);
  int budget = 32;
  size_t rot = 0;                 // 本轮已轮转的被限流片(防空转)
  while (!q.empty() && budget-- > 0) {
    ReadySlice& r = q.front();
    const int me = c->cfg.rank;
    const int peer = r.h.path[r.h.hop + 1];
    // 每目的节点在途上限:超限的片转到队尾让别的节点先走(热节点限流不
    // HOL 冷节点;发送序已证不影响 makespan,hot-first A/B 全噪声)
    bool gated_out = false;
    if (via_nic) {
      gated_out = c->node_inflight[c->topo.host(peer)] + r.h.len > node_inflight_cap();
      // 信用反压发送约束(File/design-credit-backpressure-0821.md §3.2):
      // ①物理窗:派发累计不许超过对端授权累计(cr_limit 只在对端 retire 后
      //   增长 ⇒ 每个 SEND 必有已补投的 recv 槽,RNR 构造性消失);
      // ②本 coll 突发钳:本 coll 派发 ≤ W0 + 本 coll 内回流(初始突发被
      //   demand 感知的 W0 罩住)。uniform 时 W0=nrx ⇒ ②恒不紧于①,行为不变。
      if (!gated_out && c->use_verbs) {
        const uint64_t grow = c->cr_limit[peer] - c->cr_base_limit[peer];
        const uint64_t used = c->cr_sent[peer] - c->cr_base_sent[peer];
        gated_out = (c->cr_sent[peer] >= c->cr_limit[peer]) ||
                    (used >= c->cr_w0[peer] + grow);
      }
    }
    if (gated_out) {
      if (++rot > q.size()) return;          // 全队都被限 = 等完成/credit 事件
      ReadySlice cp = r;
      q.pop_front();
      q.push_back(cp);
      budget++;
      continue;
    }
    const bool via_spine = via_nic && (c->topo.rail(me) != c->topo.rail(peer));

    InflightTx t;
    t.h = r.h; t.peer = peer; t.via_nic = via_nic; t.relay_slot = r.relay_slot;
    void* hdst = nullptr;
    // TE_SLICE_POST 打在各分支**真正派发**之后 —— 打在背压检查前会把同一片的
    // 重试轮询记录成百上千次(29425220-021058 实测),观测面失真
    // 流序窗口(kdesc):verbs TX 的 staging 拷贝(sendbuf/relay → host pinned,
    // SM 经 UVA 写)交消费者核 —— 引擎零 CUDA 调用。**窗口内引擎不许碰任何
    // CUDA API**:gated 下 app 线程在 torch sync 里持驱动锁,event 线程一句
    // cudaMemcpyAsync 就互锁(29401070-011125 MoE 实测:node1 四 rank event
    // 线程连 watchdog 都冻死,724s 无进展)。SM 出向流写不再有代价:RX 已改
    // host-bounce,不存在会被压制的入向 GDR 写(见 verbs.cc 头注)。
    // 拷完(kq 排干,kind=4)再发 WR。hop0 源 = sendbuf,输入门在消费者核内。
    if (via_nic && c->kdesc_active) {
      int slot = stage_alloc2(c);
      if (slot < 0) return;                     // staging 用完 = 背压
      if (!kdesc_push(c, (uint64_t)(uintptr_t)r.gpu_src,
                      stage_pay(c, slot), r.h.len)) {
        stage_release(c, slot);
        return;                                 // 描述符环满 = 背压
      }
      t.stage_slot = slot;
      c->cr_sent[peer]++;                       // 信用:派发点记账(kdesc 路径)
      MTRACE(TE_SLICE_POST, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
             ((uint64_t)r.h.len << 8) | r.h.hop);
      KEntry e; e.kind = 4; e.t = t;
      es->kq.push_back(e);
      c->buck[up_kind(true)].take(r.h.len);
      if (via_spine) c->buck[RES_SPINE_UP].take(r.h.len);
      if (r.h.hop == 0) c->posted_send += r.h.len;
      if (!c->ph_first_post_ns) c->ph_first_post_ns = now_ns();
      q.pop_front();
      continue;
    }
    if (via_nic) {
      int slot = stage_alloc2(c);
      if (slot < 0) return;                       // staging 用完 = 背压
      t.stage_slot = slot;
      hdst = stage_pay(c, slot);                  // D2H 到 host pinned staging
    } else {
      ShmRing* ring = c->tx_ring[peer];
      // S3 二期:dev 环(CUDA IPC)发送交消费者核 —— 跨 GPU cudaMemcpyAsync
      // 在 caller(legacy)流挂着门等待时会被驱动 peer-copy 路径隐式序列化
      // (Raven 实测 E5002 集体卡死);SM 直写 NVLink 绕开(NCCL 同款)。
      // publish 必须按 reserve 序 ⇒ kdesc 满时先背压,不 reserve 不混用。
      // dev-pipe2:final-hop 机内片在 kdesc 档下优先直写对端 recvbuf(1-copy)。
      // 对端窗口未发布 = 对端 input 未就绪 ⇒ 背压等(与交付输入门同语义);
      // 窗口哨兵 ~0 = 对端声明不可直写 ⇒ 走环路径;映射 miss ⇒ 本 coll 走环,
      // 句柄排队下一次 submit 补 open。
      // **仅限 hop0(src==me)**:RWIN 发布的偏移是「写入者自己的段」
      // (rdispls[写入者]),中转片 src≠me 直写会落进错误的段(按原始 src
      // 落位只有环路径的交付侧会做,engine.cc pump_shm)——n=8 hotdst 实测
      // 机内源超收 1.5×/跨机源饿死 + 数据腐蚀(local-0820-gated-bench)。
      if (c->kdesc_active && ring && r.h.hop + 1 >= r.h.nhops && r.h.src == me) {
        uint8_t hbuf[64];
        uint64_t doff = 0;
        if (!shm_rwin_read(c->tx_ring[peer], (uint32_t)c->coll_id, hbuf, &doff))
          return;                                    // 窗口未发布 = 背压
        if (doff != ~0ull) {
          std::array<uint8_t, 64> key;
          memcpy(key.data(), hbuf, 64);
          void* base = nullptr;
          bool known = false;
          {
            std::lock_guard<std::mutex> lk(c->ipc_mu);
            auto it = c->ipc_map.find(key);
            if (it != c->ipc_map.end()) { known = true; base = it->second; }
            else if (std::find(c->ipc_pending.begin(), c->ipc_pending.end(), key) ==
                     c->ipc_pending.end())
              c->ipc_pending.push_back(key);         // 下次 submit(门外)补 open
          }
          if (known && base) {
            if (c->kdesc_pushed - c->kdesc_consumed >= c->kdesc_cap) return;  // 背压
            uint8_t* dst = (uint8_t*)base + doff + r.h.off;
            if (!kdesc_push(c, (uint64_t)(uintptr_t)r.gpu_src, dst, r.h.len)) return;
            MTRACE(TE_SLICE_POST, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
                   ((uint64_t)r.h.len << 8) | r.h.hop);
            KEntry e; e.kind = 6; e.t.peer = peer; e.t.h = r.h;
            e.t.relay_slot = r.relay_slot;
            es->kq.push_back(e);
            c->buck[RES_NV_UP].take(r.h.len);
            if (r.h.hop == 0) c->posted_send += r.h.len;
            if (!c->ph_first_post_ns) c->ph_first_post_ns = now_ns();
            q.pop_front();
            continue;
          }
          // known 且 base==nullptr(open 失败)或未知:落环路径(known 失败)
          // 或本 coll 先落环(miss;句柄已排队)
        }
      }
      if (c->kdesc_active && ring) {
        if (c->kdesc_pushed - c->kdesc_consumed >= c->kdesc_cap) return;  // 背压
        SliceHdr* hh = nullptr;
        void* pay = nullptr;
        if (!shm_ring_reserve(ring, &pay, &hh)) return;        // 环满 = 背压
        *hh = r.h;
        t.shm_ring = peer;
        kdesc_push(c, (uint64_t)(uintptr_t)r.gpu_src, pay, r.h.len);   // 必成功
        c->buck[up_kind(false)].take(r.h.len);
        if (r.h.hop == 0) c->posted_send += r.h.len;
        if (!c->ph_first_post_ns) c->ph_first_post_ns = now_ns();
        MTRACE(TE_SLICE_POST, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
               ((uint64_t)r.h.len << 8) | r.h.hop);
        KEntry e; e.kind = 1; e.t = t;
        es->kq.push_back(e);
        q.pop_front();
        continue;
      }
      SliceHdr* hh = nullptr;
      void* pay = nullptr;
      if (!ring || !shm_ring_reserve(ring, &pay, &hh)) return;   // 环满 = 背压
      *hh = r.h;
      t.shm_ring = peer;
      hdst = pay;
    }
    if (via_nic) c->cr_sent[peer]++;            // 信用:派发点记账(CE 路径)
    MTRACE(TE_SLICE_POST, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
           ((uint64_t)r.h.len << 8) | r.h.hop);
    copy_async(c, hdst, r.gpu_src, r.h.len, cudaMemcpyDeviceToHost, c->stream_d2h);
    t.ev = ev_take(c); ev_record(c, t.ev, c->stream_d2h);
    c->buck[up_kind(via_nic)].take(r.h.len);
    if (via_spine) c->buck[RES_SPINE_UP].take(r.h.len);
    if (r.h.hop == 0) c->posted_send += r.h.len;
    if (!c->ph_first_post_ns) c->ph_first_post_ns = now_ns();
    es->tx.push_back(t);
    q.pop_front();
  }
}

static void pump_ready(Comm* c) {
  auto* es = ES(c);
  pump_one_queue(c, es->ready_nv, false);
  pump_one_queue(c, es->ready_nic, true);
}

static void pump_tx(Comm* c) {
  auto* es = ES(c);
  static thread_local uint64_t stuck_since = 0, next_report = 0;
  while (!es->tx.empty()) {
    InflightTx& t = es->tx.front();
    if (!ev_ready(c, t.ev)) {                     // 同一 stream ⇒ 顺序完成
      uint64_t now = now_ns();
      if (!stuck_since) stuck_since = now;
      if (now - stuck_since > 2000000000ull && now > next_report) {
        next_report = now + 2000000000ull;
        int eq = -1, sq = -1, iq = -1;   // 异步窗口内禁查询(诊断不能杀引擎)
        if (!c->kdesc_active && c->cfg.use_gpu) {
          eq = (int)cudaEventQuery((cudaEvent_t)t.ev);
          sq = (int)cudaStreamQuery((cudaStream_t)c->stream_d2h);
          iq = c->input_ev ? (int)cudaEventQuery((cudaEvent_t)c->input_ev) : 0;
        }
        MLOG_ERR("E5002 D2H 卡住:eventQuery=%d streamQuery=%d inputEv=%d len=%u peer=%d nic=%d",
                 eq, sq, iq, t.h.len, t.peer, (int)t.via_nic);
      }
      break;
    }
    stuck_since = 0;
    ev_give(c, t.ev);
    if (t.shm_ring >= 0) {
      // payload 已在 IPC 环里,发布即对端可见
      shm_ring_publish(c->tx_ring[t.shm_ring]);
    } else {
      c->nic_tx_outstanding++;                    // 检测器 busy 判定用
      MTRACE(TE_SLICE_TX, ((uint32_t)t.h.src << 16) | t.h.dst, t.h.off,
             ((uint64_t)t.h.len << 8) | t.h.hop); // D2D 组装完成,这一刻才真正投出
      // RDMA SEND 双 SGE [host 头 | GPU payload],完成事件回收槽。
      // 数据面无回退:发送失败 = 硬错终止本 collective。
      if (verbs_post_slice(c, t.peer, t.h, t.stage_slot) != 0) {
        MLOG_ERR("E1027 verbs 发送失败(r%d)—— 数据面无回退,置错终止本 collective", t.peer);
        c->err = "E1027 verbs 发送失败";
        c->coll_state.store(3);
        stage_release(c, t.stage_slot);
      }
    }
    if (t.relay_slot >= 0) c->relay_free.push_back(t.relay_slot);
    c->st.bytes_sent += t.h.len;
    c->st.corridor_bytes[t.h.corr & 7] += t.h.len;
    es->tx.pop_front();
  }
}

// 交付完成后的统一收编(源槽回收 / credit / 账目 / FIN / 中转下一跳)。
// 旧拷贝路径(event 序)与消费者核路径(done 计数序)共用同一份语义。
static void rx_retire(Comm* c, InflightRx& r) {
  auto* es = ES(c);
  // 源槽位现在可以回收了(字节已经落到 GPU)
  if (r.src_shm_ring >= 0) shm_ring_pop(c->rx_ring[r.src_shm_ring]);
  if (r.src_stage >= 0) stage_release(c, r.src_stage);
  if (r.src_verbs >= 0) {
    verbs_repost(c, r.src_verbs);
    // 信用铸造点(唯一):交付完成 + 槽已补投 = 真实排水一片。回流由
    // pump_credits 合批;稳态下发送端进水率被这里的节拍闭环钳住。
    c->cr_bank[r.src_verbs >> 20]++;
    MTRACE(TE_REPOST, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
           ((uint64_t)r.h.len << 8) | r.h.hop);   // credit 释放时刻
  }

  if (r.final_hop) {
    MTRACE(TE_SLICE_DELIV, ((uint32_t)r.h.src << 16) | r.h.dst, r.h.off,
           ((uint64_t)r.h.len << 8) | r.h.hop);
    c->st.bytes_recv += r.h.len;
    c->dst.total_recv += r.h.len;
    c->dst.recv_by_src[r.h.src] += r.h.len;
    // 波触发计交付(不是到达):A/B 实测(29425220-0200 vs -011534)到达即计
    // 让 α=1/2 斜格劣化 27-31% —— 交付滞后是对热点的**保护性背压**(触发环路
    // 里含接收侧真实排水能力);uniform 由 W 自适应走 W=1,不受此滞后影响。
    if (r.h.wave == c->dst.wave_idx - 1) c->dst.wave_delivered += r.h.len;
    // 本 dst 收满 → 广播 CT_FIN(PACE 的 hot-done 放行信号;控制面小报文)
    if (!es->fin_sent && c->dst.total_recv >= c->total_recv_expect &&
        c->total_recv_expect > 0) {
      es->fin_sent = true;
      ctrl_bcast(c, CT_FIN, (uint16_t)(c->coll_id & 0xFFFF), nullptr, 0);
    }
    maybe_open_gate_early(c);
  } else {
    // hop k 完整落到中转 GPU 之后,才允许发 hop k+1(chunk availability 铁律)
    SliceHdr nh = r.h;
    nh.hop = r.h.hop + 1;
    enqueue_hop(c, nh, relay_ptr(c, r.relay_slot), r.relay_slot);
  }
}

static void pump_rx(Comm* c) {
  auto* es = ES(c);
  // S3 二期:消费者核已完成的描述符(done 为设备写的连续前缀计数,
  // 纯内存读 —— 完成路径零 CUDA 调用)。tx 条目 = dev 环发送,核拷完
  // payload 后由 host publish(publish 序 == reserve 序,推入即按序)
  if (!es->kq.empty()) {
    uint64_t done = __atomic_load_n((uint64_t*)(c->kdesc_ctr + 64), __ATOMIC_ACQUIRE);
    while (!es->kq.empty() && c->kdesc_consumed < done) {
      KEntry& e = es->kq.front();
      if (e.kind == 6) {
        // dev-pipe2 直写完成(消费者核已把字节写进对端 recvbuf):
        // 经 SHM 邮箱给对端记交付账;对端 fold 后计入它的 total_recv/FIN
        shm_dlv_add(c->tx_ring[e.t.peer], e.t.h.len);
        c->st.bytes_sent += e.t.h.len;
        c->st.corridor_bytes[e.t.h.corr & 7] += e.t.h.len;
        if (e.t.relay_slot >= 0) c->relay_free.push_back(e.t.relay_slot);
      } else if (e.kind == 4) {
        // verbs 流序 TX:消费者核已把 payload 拷进 host staging ⇒ 发 WR
        c->nic_tx_outstanding++;
        MTRACE(TE_SLICE_TX, ((uint32_t)e.t.h.src << 16) | e.t.h.dst, e.t.h.off,
               ((uint64_t)e.t.h.len << 8) | e.t.h.hop);
        if (verbs_post_slice(c, e.t.peer, e.t.h, e.t.stage_slot) != 0) {
          MLOG_ERR("E1027 verbs 发送失败(r%d)—— 数据面无回退,置错终止本 collective",
                   e.t.peer);
          c->err = "E1027 verbs 发送失败";
          c->coll_state.store(3);
          stage_release(c, e.t.stage_slot);
        } else {
          c->st.bytes_sent += e.t.h.len;
          c->st.corridor_bytes[e.t.h.corr & 7] += e.t.h.len;
        }
        if (e.t.relay_slot >= 0) c->relay_free.push_back(e.t.relay_slot);
      } else if (e.kind == 2) {
        // 纯拷贝(self-copy):清在途计数,自拷贝可能是开门前最后一块
        if (c->self_kq_pending > 0) c->self_kq_pending--;
        maybe_open_gate_early(c);
      } else if (e.kind == 0) {
        rx_retire(c, e.r);
      } else {
        InflightTx& t = e.t;
        shm_ring_publish(c->tx_ring[t.shm_ring]);
        if (t.relay_slot >= 0) c->relay_free.push_back(t.relay_slot);
        c->st.bytes_sent += t.h.len;
        c->st.corridor_bytes[t.h.corr & 7] += t.h.len;
      }
      es->kq.pop_front();
      c->kdesc_consumed++;
    }
  }
  while (!es->rx.empty()) {
    InflightRx& r = es->rx.front();
    if (!ev_ready(c, r.ev)) break;
    ev_give(c, r.ev);
    rx_retire(c, r);
    es->rx.pop_front();
  }
}

static void pump_shm(Comm* c) {
  for (int p = 0; p < c->cfg.nranks; p++) {
    ShmRing* ring = c->rx_ring[p];
    if (!ring) continue;
    for (int k = 0; k < 8; k++) {
      void* pay = nullptr;
      SliceHdr* hh = nullptr;
      if (!shm_ring_peek(ring, &pay, &hh)) break;
      // 头部要中转但中转池空 → 整条环停住,等池子有位再来(FIFO 不能跳过队头)
      if (hh->hop + 1 < hh->nhops && c->relay_free.empty()) break;
      // 坑 #48:机内路径同理 —— 终跳片属于更后面的一代就整条环停住(不 commit,
      // 槽位留着),等本 rank 进到那一代再收。与上面的中转池背压是同一个形状。
      if (hh->hop + 1 >= hh->nhops &&
          (int16_t)(hh->coll - (uint16_t)(c->coll_id & 0xFFFF)) > 0) break;
      SliceHdr h = *hh;
      shm_ring_commit(ring);
      // 槽位不在这里归还 —— H2D 完成后才 pop(pump_rx),否则生产者会覆盖在途数据
      on_slice_arrived(c, h, pay, p, -1);
    }
  }
}

// 坑 #51:被推迟的自拷贝 —— 输入就绪后交消费者核(kind=2 纯拷贝)。
// 纯内存判定(读 pinned 标志),零 CUDA 调用(流序窗口纪律)。
static void try_self_copy(Comm* c) {
  if (!c->self_copy_pending) return;
  auto* es = ES(c);
  const int me = c->cfg.rank;
  void* sdst = (uint8_t*)c->recvbuf + c->rdispls[me];
  uint64_t ssrc = (uint64_t)(uintptr_t)((const uint8_t*)c->sendbuf + c->sdispls[me]);
  if (!kdesc_push(c, ssrc, sdst, c->scounts[me])) return;   // 环满,下轮再试
  KEntry e; e.kind = 2;
  es->kq.push_back(e);
  c->self_kq_pending++;
  c->self_copy_pending = false;
}

// ── recv 即开门(2026-08-21;每 rank 完成结构优化):torch 语义下门只需保证
// recvbuf 就绪 —— sendbuf 由 python 侧 _inflight 引用持到下一次调用,不会被
// allocator 复用;残余发送/中转在引擎后台排空,下一次 submit 的 wait_coll
// 兜底闭合(一代在飞不破)。skew 下冷 dst 因此不再陪热 dst 等到 makespan
// (29425220-025934 实测:NCCL 冷 rank 也要等 ~makespan,因为它的完成含发送
// 排空 —— 这是对 NCCL 的结构性差异轴)。安全前提:①total_recv 只在拷贝完成
// 后计(rx_retire = kdesc done 前缀;DLV = 对端核完成后加账);②自拷贝
// (不在 expect 里)单独用 self_kq_pending 等清零;③门 flag 是纯 pinned 写。
static void maybe_open_gate_early(Comm* c) {
  if (!c->gated || !c->done_flag_host) return;
  if (c->self_copy_pending || c->self_kq_pending > 0) return;
  if (c->total_recv_expect > 0 && c->dst.total_recv < c->total_recv_expect) return;
  __atomic_store_n((uint32_t*)c->done_flag_host, (uint32_t)c->coll_id, __ATOMIC_RELEASE);
}

// dev-pipe2 泵:①input 就绪后向机内各 src 发布本 coll 的 recv 窗口(SHM 写,
// 零 CUDA);②折叠各机内 src 直写交付的字节进账目(与 rx_retire 的 final-hop
// 语义对齐;wave 归属按当前波近似 —— 机内直写形态默认单波,见 op.md)。
static void pump_devpipe(Comm* c) {
  auto* es = ES(c);
  if (!es->coll_started) return;
  const int n = c->cfg.nranks, me = c->cfg.rank;
  if (!c->rwin_published && c->kdesc_active && kdesc_input_ready(c)) {
    for (int p = 0; p < n; p++) {
      if (p == me || !c->rx_ring[p]) continue;
      // 无效窗口用 ~0 哨兵显式广播:发送方读到即确定走环路径,不再等待
      shm_rwin_publish(c->rx_ring[p], (uint32_t)c->coll_id, c->my_rwin_handle,
                       c->my_rwin_valid ? c->my_rwin_base_off + c->rdispls[p] : ~0ull);
    }
    c->rwin_published = true;
  }
  DstCtrl& d = c->dst;
  for (int p = 0; p < n; p++) {
    if (p == me || !c->rx_ring[p]) continue;
    uint64_t cur = shm_dlv_read(c->rx_ring[p]);
    uint64_t delta = cur - c->dlv_base[p] - c->dlv_folded[p];
    if (!delta) continue;
    c->dlv_folded[p] += delta;
    c->buck[RES_NV_DN].take(delta);
    c->st.bytes_recv += delta;
    d.total_recv += delta;
    d.recv_by_src[p] += delta;
    d.wave_delivered += delta;
    if (!es->fin_sent && d.total_recv >= c->total_recv_expect && c->total_recv_expect > 0) {
      es->fin_sent = true;
      ctrl_bcast(c, CT_FIN, (uint16_t)(c->coll_id & 0xFFFF), nullptr, 0);
    }
    maybe_open_gate_early(c);
  }
}

static void pump_wait_relay(Comm* c) {
  auto* es = ES(c);
  // 只试队头,保持 FIFO(SHM 环的 pop 序、per-peer 交付序都靠它):
  // 重试被再次 defer(押回队尾)时移回队头并停 —— 严格不重排。
  while (!es->wait_relay.empty()) {
    WaitRelay w = es->wait_relay.front();
    es->wait_relay.pop_front();
    size_t before = es->wait_relay.size();
    on_slice_arrived(c, w.h, w.payload, w.src_shm_ring, w.src_stage, w.src_verbs);
    if (es->wait_relay.size() > before) {        // 又被 defer(在队尾)
      WaitRelay w2 = es->wait_relay.back();
      es->wait_relay.pop_back();
      es->wait_relay.push_front(w2);
      break;
    }
  }
}

// 状态转储(06 §5.3):watchdog 首吠与 SIGTERM 都会调;落文件,卡死必留尸检
void dump_state(Comm* c, const char* why) {
  auto* es = ES(c);
  const char* dir = getenv("MOSAIC_DUMP_DIR");
  char path[512];
  snprintf(path, sizeof(path), "%s/mosaic_dump_rank%d.txt",
           dir && *dir ? dir : "/tmp", c->cfg.rank);
  FILE* f = fopen(path, "w");
  if (!f) return;
  fprintf(f, "why=%s t=%llu rank=%d/%d sched=%d coll=%d state=%d evphase=%d\n", why,
          (unsigned long long)now_ns(), c->cfg.rank, c->cfg.nranks, c->cfg.sched,
          c->coll_id, c->coll_state.load(), g_ev_phase.load());
  fprintf(f, "recv %llu/%llu send %llu/%llu wave=%d wb=%llu wd=%llu paced_hot=%d\n",
          (unsigned long long)c->dst.total_recv, (unsigned long long)c->total_recv_expect,
          (unsigned long long)c->posted_send, (unsigned long long)c->total_send,
          c->dst.wave_idx, (unsigned long long)c->dst.wave_bytes,
          (unsigned long long)c->dst.wave_delivered, c->pace_hot);
  if (c->src_mode) {
    uint64_t rem = 0;
    int unseen = 0;
    for (int p = 0; p < c->cfg.nranks; p++) {
      rem += c->srcc.remaining[p];
      if (p != c->cfg.rank && !c->peer_row_seen.empty() && !c->peer_row_seen[p]) unseen++;
    }
    fprintf(f, "srcmode wave=%d nic=%llu acked=%llu remaining=%llu rows_unseen=%d\n",
            c->srcc.wave_idx, (unsigned long long)c->srcc.wave_nic_bytes,
            (unsigned long long)c->srcc.wave_acked, (unsigned long long)rem, unseen);
  }
  fprintf(f, "q nv=%zu nic=%zu tx=%zu rx=%zu waitrelay=%zu stage_free=%zu relay_free=%zu\n",
          es->ready_nv.size(), es->ready_nic.size(), es->tx.size(), es->rx.size(),
          es->wait_relay.size(), c->stage_free.size(), c->relay_free.size());
  for (int s = 0; s < c->cfg.nranks; s++) {
    if (s == c->cfg.rank) continue;
    // want = 权威收期望(调用方 rcounts,坑 #43);dmd = 收到的 demand 矩阵那格
    // (可能陈旧,只用于对拍);scnt = **我**给该 peer 的发送量。跨 rank 对拍时
    // 只需比 R 的 want[S] 与 S 的 scnt[R] —— 不等就是两端对"第几个 collective"
    // 的理解不一致,而不是调度问题。
    fprintf(f, "src %d recv %llu want %llu dmd %llu scnt %llu remaining %llu "
               "peer_wave %d fin %d det %u\n", s,
            (unsigned long long)c->dst.recv_by_src[s],
            (unsigned long long)(s < (int)c->rcounts.size() ? c->rcounts[s] : 0),
            (unsigned long long)(c->demand.empty() ? 0 : c->demand[(size_t)s * c->cfg.nranks + c->cfg.rank]),
            (unsigned long long)(s < (int)c->scounts.size() ? c->scounts[s] : 0),
            (unsigned long long)c->dst.remaining[s], c->peer_wave[s], (int)c->peer_fin[s],
            (unsigned)c->peer_det[s]);
  }
  for (int r = 0; r < c->topo.nres() && r < 512; r++) {
    uint64_t out = c->ledger.outstanding(r);
    if (out) fprintf(f, "res %d outstanding %llu cap %.1f\n", r, (unsigned long long)out,
                     c->ledger.cap_Bps[r]);
  }
  fclose(f);
  MTRACE(TE_DUMP, 0, 0, 0);
  MLOG_ERR("状态转储 → %s(%s)", path, why);
}

// Watchdog(Imp_Plan/06 §5.3):卡死必须留尸检,不能只表现为"超时"。
// Raven 上不能交互调试,这段输出就是唯一的现场。
static void watchdog(Comm* c) {
  auto* es = ES(c);
  static thread_local uint64_t last_prog_ns = 0, last_recv = ~0ull, next_bark = 0;
  uint64_t t = now_ns();
  if (c->coll_state.load() != 1) { last_recv = ~0ull; return; }
  if (c->dst.total_recv != last_recv) { last_recv = c->dst.total_recv; last_prog_ns = t; next_bark = t + 2000000000ull; return; }
  if (last_prog_ns == 0) { last_prog_ns = t; next_bark = t + 2000000000ull; return; }
  if (t < next_bark) return;
  next_bark = t + 2000000000ull;
  dump_state(c, "watchdog");                  // 每次吠都刷新转储(保最新现场)
  MLOG_ERR("E2002 watchdog:%.1fs 无进展 | recv %llu/%llu send %llu/%llu | "
           "started=%d demand_sent=%d demand_got=%d wave=%d wb=%llu wd=%llu | "
           "q: nv=%zu nic=%zu tx=%zu rx=%zu waitrelay=%zu | pool: stage=%zu relay=%zu | "
           "kd: act=%d push=%llu cons=%llu done=%llu q=%zu",
           (t - last_prog_ns) / 1e9,
           (unsigned long long)c->dst.total_recv, (unsigned long long)c->total_recv_expect,
           (unsigned long long)c->posted_send, (unsigned long long)c->total_send,
           (int)es->coll_started, (int)es->demand_sent, es->demand_got,
           c->dst.wave_idx, (unsigned long long)c->dst.wave_bytes,
           (unsigned long long)c->dst.wave_delivered,
           es->ready_nv.size(), es->ready_nic.size(), es->tx.size(), es->rx.size(),
           es->wait_relay.size(), c->stage_free.size(), c->relay_free.size(),
           (int)c->kdesc_active, (unsigned long long)c->kdesc_pushed,
           (unsigned long long)c->kdesc_consumed,
           c->kdesc_ctr ? (unsigned long long)__atomic_load_n((uint64_t*)(c->kdesc_ctr + 64),
                                                              __ATOMIC_ACQUIRE) : 0ull,
           es->kq.size());
  // 队头 slice 的身份:队列非空却不动时定位卡点(池水位在 nvsh 活性行)
  {
    char hb[256]; int ho = 0;
    auto head = [&](const char* tag, std::deque<ReadySlice>& q, bool via_nic) {
      if (q.empty()) return;
      const SliceHdr& h = q.front().h;
      const int pe = h.path[h.hop + 1];
      (void)via_nic;
      ho += snprintf(hb + ho, sizeof(hb) - ho,
                     "%s[->r%d %uB hop%d s%u->d%u corr%u] ", tag, pe,
                     (unsigned)h.len, (int)h.hop, (unsigned)h.src, (unsigned)h.dst,
                     (unsigned)(h.corr & 7));
    };
    head("nvq", es->ready_nv, false);
    head("nicq", es->ready_nic, true);
    if (ho) MLOG_ERR("E2002 队头: %s", hb);
  }
  char rb[512]; int o = 0;
  for (int p = 0; p < c->cfg.nranks && o < 400; p++) {
    uint64_t h1, t1; int i1, p1, h2, t2;
    if (c->rx_ring[p]) { uint64_t hh, tt; int ii, pp; shm_ring_stat(c->rx_ring[p], &hh, &tt, &ii, &pp);
      o += snprintf(rb + o, sizeof(rb) - o, "rx%d[h=%llu t=%llu i=%d] ", p,
                    (unsigned long long)hh, (unsigned long long)tt, ii); }
    if (c->tx_ring[p]) { uint64_t hh, tt; int ii, pp; shm_ring_stat(c->tx_ring[p], &hh, &tt, &ii, &pp);
      o += snprintf(rb + o, sizeof(rb) - o, "tx%d[h=%llu t=%llu p=%d] ", p,
                    (unsigned long long)hh, (unsigned long long)tt, pp); }
  }
  o += snprintf(rb + o, sizeof(rb) - o, "| 逐src收/应收: ");
  for (int p = 0; p < c->cfg.nranks && o < 380; p++) {
    if (p == c->cfg.rank) continue;
    uint64_t want = c->demand[(size_t)p * c->cfg.nranks + c->cfg.rank];
    uint64_t got = c->dst.recv_by_src[p];
    o += snprintf(rb + o, sizeof(rb) - o, "%d:%llu/%llu%s ", p, (unsigned long long)got,
                  (unsigned long long)want, got == want ? "" : "*");
  }
  MLOG_ERR("E2002 watchdog 环状态: %s", rb);
}

// 信用回流(接收端;File/design-credit-backpressure-0821.md §3.3):rx_retire
// 铸造的 credit 合批发回。同一圈内的多次 retire 自然合为一条报文;ctrl 走
// verbs SEND(单程 ~5µs),池干涸自动回退 TCP。发送失败 bank 留到下圈,
// 累计语义保证 credit 不丢不重。
static void pump_credits(Comm* c) {
  if (!c->use_verbs) return;
  for (int p = 0; p < c->cfg.nranks; p++) {
    if (!c->cr_bank[p]) continue;
    uint64_t cum = c->cr_granted[p] + c->cr_bank[p];
    if (ctrl_send(c, p, CT_CREDIT, 0, &cum, 8) == 0) {
      c->cr_granted[p] = cum;
      c->cr_bank[p] = 0;
    }
  }
}

static void pump_protocol(Comm* c) {
  auto* es = ES(c);
  uint64_t t = now_ns();
  sync_own_discharge(c);

  // 通道 A:周期遥测(有界陈旧 SSP,非阻塞)
  if (c->cfg.sched == MOSAIC_SCHED_MOSAIC && t >= c->telem_next_ns) {
    // 节拍抖动测量(03 §7:p99 < 10µs 是效率红线)
    if (c->telem_next_ns > 0)
      c->tick_hist[c->tick_hist_n++ & 255] = (double)(t - c->telem_next_ns) / 1000.0;
    c->telem_next_ns = t + (uint64_t)c->cfg.sync_us * 1000;
    TelemMsg m{};
    m.seq = ++c->telem_seq;
    for (int k = 0; k < RES_KINDS; k++) m.disch[k] = c->buck[k].passed - es->base_passed[k];
    m.nic_up_gbps = (float)c->det[0].rate_gbps;
    m.nic_dn_gbps = (float)c->det[1].rate_gbps;
    m.det_state = (c->det[0].locked ? 1 : 0) | (c->det[1].locked ? 2 : 0);
    memcpy(m.fab_up, c->fab_own[0], sizeof(m.fab_up));
    memcpy(m.fab_dn, c->fab_own[1], sizeof(m.fab_dn));
    for (int p = 0; p < c->cfg.nranks; p++) {
      if (p == c->cfg.rank) continue;
      // 注入:通道 A 人为丢包(SSP 稳健性实验,MOSAIC_DROP_A)
      if (c->drop_a > 0 && (double)(rng_next(c) % 1000000) / 1e6 < c->drop_a) {
        c->st.chA_drops++;
        continue;
      }
      // P2 定向:按该 dst 的读集意图裁形态。det 位对全员恒发(quorum 语义)。
      uint8_t im = opt_telem_targeted() && !c->peer_intent.empty()
                       ? c->peer_intent[p] : INTENT_FULL;
      const void* buf = &m;
      uint32_t blen = sizeof(m);
      TelemDet md{};
      if (im == 0) {
        md.seq = m.seq; md.nic_up_gbps = m.nic_up_gbps;
        md.nic_dn_gbps = m.nic_dn_gbps; md.det_state = m.det_state;
        buf = &md; blen = sizeof(md);
      } else if (!(im & INTENT_FAB)) {
        blen = TELEM_NOFAB_BYTES;               // 去掉 fabric 贡献(报文的 2/3)
      }
      ctrl_send(c, p, CT_TELEM, 0, buf, blen);
      c->st.chA_msgs++;
      c->st.chA_bytes += sizeof(CtrlHdr) + blen;
    }
    MTRACE(TE_CHA_PUB, (uint32_t)m.seq, 0, 0);
  }

  // ── src 相的波触发(任务二):上一波跨机字节的 RC ACK 过阈值 → 错峰槽 →
  // 到点再决策。全 NV 波(wave_nic_bytes=0)无 ACK 事件,即触发(NV 单跳
  // 走廊无重规划自由度,多等无益)。迟到行的补发也在此:全部波额度用完后
  // 才开门的 dst,一次性发它的全部剩余。
  if (c->src_mode && c->srcc.active && c->cfg.sched == MOSAIC_SCHED_MOSAIC) {
    SrcCtrl& s = c->srcc;
    const bool thr_hit = s.wave_nic_bytes == 0 ||
        (double)s.wave_acked >= wave_thresh() * (double)s.wave_nic_bytes;
    const bool deadline_hit = s.wave_deadline_ns && t > s.wave_deadline_ns;
    if (!s.pending_trigger && s.wave_idx < eff_W(c) && (thr_hit || deadline_hit)) {
      bool more = false;
      for (int p = 0; p < c->cfg.nranks; p++)
        if (p != c->cfg.rank && s.remaining[p] &&
            (c->peer_row_seen.empty() || c->peer_row_seen[p])) { more = true; break; }
      if (more) {
        // 决策错峰:与 dst 环同一套槽位(同瞬触发的 n 个 src 互相读到
        // 更早槽位的通道 B 发布再决策)
        s.pending_trigger = true;
        s.trigger_set_ns = t;
        int slot = c->cfg.rank % std::max(1, c->cfg.stagger_slots);
        s.trigger_ns = t + (uint64_t)slot * c->cfg.stagger_us * 1000;
      }
    }
    if (s.pending_trigger && t >= s.trigger_ns) {
      c->st.stagger_us_total += (t - s.trigger_set_ns) / 1000.0;
      MTRACE(TE_STAGGER, (uint32_t)s.wave_idx, (t - s.trigger_set_ns) / 1000, 0);
      s.pending_trigger = false;
      emit_wave_src(c);
    }
    // 迟到行补发:波额度已用完、行门后开的 dst 还压着 remaining →
    // 立即补一波(waves_left=1 ⇒ 全额)。行门未开的 dst 继续等它的行。
    if (s.wave_idx >= eff_W(c) && !s.pending_trigger) {
      for (int p = 0; p < c->cfg.nranks; p++)
        if (p != c->cfg.rank && s.remaining[p] &&
            (c->peer_row_seen.empty() || c->peer_row_seen[p])) {
          emit_wave_src(c);
          break;
        }
    }
  }

  // 波触发(dst 相):交付过阈值且还有波额度 → 进错峰槽 → 到点再决策
  DstCtrl& d = c->dst;
  if (!c->src_mode && d.active && c->cfg.sched == MOSAIC_SCHED_MOSAIC) {
    const bool thr_hit = d.wave_bytes > 0 &&
        (double)d.wave_delivered >= wave_thresh() * (double)d.wave_bytes;
    const bool deadline_hit = d.wave_deadline_ns && t > d.wave_deadline_ns;
    // PACE 门:冷 dst 最多领先 hot k-1 波(hot 完成或推进即放行;
    // deadline×2 兜底防饿死 —— 模拟器 release_paced 的 runtime 对应物)
    bool paced = false;
    if (c->pace_hot >= 0 && c->pace_hot != c->cfg.rank && c->pace_cold[c->cfg.rank] &&
        !c->peer_fin[c->pace_hot] &&
        d.wave_idx > c->peer_wave[c->pace_hot] + c->cfg.cold_pace - 1 &&
        !(d.wave_deadline_ns &&
          t > d.wave_deadline_ns + (d.wave_deadline_ns - d.wave_start_ns))) {
      paced = true;   // 兜底:超过一个额外 deadline 周期仍无 hot 进展则放行
    }
    // 触发归因(TE_STAGGER 复用:b 高位 = 已交付比例‰,低位 = 途径):
    // 每 2ms 打一个"等待中"样本,触发瞬间打途径 —— 波节奏之谜的观测面
    if (d.active && d.wave_bytes > 0 && d.wave_idx < eff_W(c) && g_trace_on) {
      static thread_local uint64_t next_smp = 0;
      if (t > next_smp) {
        next_smp = t + 2000000;
        MTRACE(TE_STAGGER, 0xFFFF, (d.wave_delivered * 1000 / d.wave_bytes) * 4 +
                                        (thr_hit ? 1 : 0) + (deadline_hit ? 2 : 0), 0);
      }
    }
    if (!paced && !d.pending_trigger && d.wave_idx < eff_W(c) &&
        d.wave_bytes > 0 && (thr_hit || deadline_hit)) {
      bool more = false;
      for (int s = 0; s < c->cfg.nranks; s++) if (d.remaining[s]) { more = true; break; }
      if (more) {
        // 决策错峰:同一瞬间触发的多个 dst 会互相看不见对方这一波的 charges,
        // 按 (dst mod slots)×stagger_us 排队,读到最新发布再决策(报告 §11.2)
        d.pending_trigger = true;
        d.trigger_set_ns = t;
        int slot = c->cfg.rank % std::max(1, c->cfg.stagger_slots);
        d.trigger_ns = t + (uint64_t)slot * c->cfg.stagger_us * 1000;
      }
    }
    if (d.pending_trigger && t >= d.trigger_ns) {
      c->st.stagger_us_total += (t - d.trigger_set_ns) / 1000.0;
      MTRACE(TE_STAGGER, (uint32_t)d.wave_idx, (t - d.trigger_set_ns) / 1000, 0);
      d.pending_trigger = false;
      emit_wave(c);
    }
  }
}

// ─────────────────────────────────────────────────────────── 事件线程
static void event_loop(Comm* c) {
  g_rank_for_log = c->cfg.rank;
  // 崩溃 handler 的 alternate stack 是**按线程**记账的:app 线程装过不代表
  // event 线程有。异步档下崩的多半正是这条线程(坑 #44)
  mosaic_install_crash_handlers_this_thread();
  // cudaSetDevice 每线程生效:不设的话本线程落默认 dev0,与 app 线程选的卡
  // 不同 ⇒ ev_take 的 event 和 s_tx/stream_d2h 不同设备,eventRecord 全炸(R13)
  if (c->cfg.use_gpu) CUCHK(cudaSetDevice(c->cuda_dev), "setDevice(ev)");
  auto* es = ES(c);
  const int n = c->cfg.nranks;
  // 调试心跳(MOSAIC_LOG>=3):每秒报一次事件循环活着 + 所在阶段;
  // g_ev_phase 由 dump_state(可能在别的线程)读,定位"冻结在哪个泵"
  uint64_t hb_next = 0;
  uint64_t loop_prev_ns = 0;                      // TE_LOOP:上一圈起点(仅 trace>=2)
  // 控制面兼容档的建链:erpc::Rpc 与创建线程绑定,只能在这里做(不能在
  // engine_start 的 app 线程)。起不来就退回 TCP —— 控制面永远有兜底,
  // 不像数据面那样硬错。
  if (c->ctrl_erpc && erpc_ctrl_setup(c) != 0) {
    c->ctrl_erpc = false;
    MLOG_WARN("E1032 控制面兼容档建链失败(未编入 eRPC 或 DPDK 端口不可用),回退 TCP");
  }
  while (c->running.load()) {
    if (g_trace_loop) {
      uint64_t lt = now_ns();
      mtrace(TE_LOOP, 0, loop_prev_ns ? lt - loop_prev_ns : 0, 0);
      loop_prev_ns = lt;
    }
    if (g_log_level >= 3) {
      uint64_t hb_t = now_ns();
      if (hb_t >= hb_next) {
        hb_next = hb_t + 1000000000ull;
        MLOG_DBG("evloop 心跳 phase=%d state=%d started=%d got=%d coll=%d kd=%llu/%llu",
                 g_ev_phase.load(), c->coll_state.load(), (int)es->coll_started,
                 es->demand_got, c->coll_id,
                 (unsigned long long)c->kdesc_consumed, (unsigned long long)c->kdesc_pushed);
      }
    }
    // O0b:仅 MOSAIC_PHASE=1 时逐泵计时(单次最大耗时),归因察觉延迟。
    // phlog 为 false 时 seg 是纯透传,性能档路径不变。
    const bool phlog = opt_phase_log();
    uint64_t seg_t0 = 0;
    auto seg = [&](int ph) {          // 段边界:收上一段、开下一段(ph=0 只收尾)
      if (!phlog) { g_ev_phase = ph; return; }
      uint64_t t = now_ns();
      int prev = g_ev_phase.load();
      if (seg_t0 && prev < 16 && t - seg_t0 > c->ev_seg_max[prev])
        c->ev_seg_max[prev] = t - seg_t0;
      seg_t0 = t;
      g_ev_phase = ph;
    };
    seg(1);
    net_poll(c);
    // 新 collective:先做 demand 矩阵 allgather(协议要全局矩阵),收齐才开跑
    if (c->coll_state.load() == 1 && !es->coll_started) {
      if (!es->demand_sent) {
        g_ev_phase = 11;
        c->ph_notice_ns = now_ns();   // O0:event 线程察觉提交(notice_us 锚点)
        if (phlog) {                  // O0b:快照+清零 —— 窗口 = 上次察觉→本次察觉
          memcpy(c->ev_seg_snap, c->ev_seg_max, sizeof(c->ev_seg_max));
          memset(c->ev_seg_max, 0, sizeof(c->ev_seg_max));
        }
        MLOG_DBG("demand 块进入(coll=%d)", c->coll_id);
        // 报文 = 我的行(n×8B)+ P2 读集意图位图(n×1B)+ 走廊承诺位(1B,
        // 中转义务门用,见 corridor_intent_bit)—— 全部捎带,零新增条数
        std::vector<uint8_t> body(sizeof(uint64_t) * n + n + 1);
        uint64_t* row = (uint64_t*)body.data();
        for (int t = 0; t < n; t++) row[t] = c->scounts[t];
        memcpy(&c->demand[(size_t)c->cfg.rank * n], row, sizeof(uint64_t) * n);
        {
          std::vector<uint8_t> intent;
          compute_intent(c, intent);
          memcpy(body.data() + sizeof(uint64_t) * n, intent.data(), n);
        }
        // 本 coll 的承诺位窗口从此开:先复位对端位,再挂自己的
        // (此前到的本 cid 行都还在 pend,fold 在下方,顺序安全)
        if ((int)c->peer_corr_intent.size() != n) c->peer_corr_intent.assign(n, 0);
        std::fill(c->peer_corr_intent.begin(), c->peer_corr_intent.end(), 0);
        // 行门复位(src 相的发送门;dst 相不读它,复位无副作用)。
        // 相位判定也在此落定:它只依赖提交参数与旋钮,必须先于承诺位计算
        // (承诺位在 src 相按出向行取谓词)。
        if ((int)c->peer_row_seen.size() != n) c->peer_row_seen.assign(n, 0);
        std::fill(c->peer_row_seen.begin(), c->peer_row_seen.end(), 0);
        c->src_mode = c->cfg.sched == MOSAIC_SCHED_MOSAIC && c->coll_phase == 1 &&
                      opt_src_driven() != 0;
        c->my_corr_intent = corridor_intent_bit(c);
        body[sizeof(uint64_t) * n + n] = c->my_corr_intent;
        // hint(O2)命中也照样广播自己的行:混合模式下(某 rank 局部不匹配退回
        // mode 0)它还等着收满 n-1 行,一条单程小报文换掉一类死锁
        ctrl_bcast(c, CT_DEMAND, (uint16_t)c->coll_id, body.data(), (uint32_t)body.size());
        c->ph_bcast_ns = now_ns();    // O0:n-1 份串行发送段尾(bcast_us 锚点)
        g_ev_phase = 12;
        MLOG_DBG("demand 行已广播(coll=%d)", c->coll_id);
        es->demand_sent = true;
        // 收编"比我先到"的行(对端可能已经进入本次 collective 而我还没)。
        // pend 槽必须属于本 coll_id(pend_cid 归属校验,防陈旧行冒充)
        int si = c->coll_id & (PEND_SLOTS - 1);
        auto& slot = es->pend_rows[si];
        if ((int)slot.size() == n * n && es->pend_cid[si] == c->coll_id) {
          for (int p = 0; p < n; p++) {
            if (p == c->cfg.rank || !es->pend_seen[si][p]) continue;
            memcpy(&c->demand[(size_t)p * n], &slot[(size_t)p * n], sizeof(uint64_t) * n);
            c->peer_corr_intent[p] = es->pend_cintent[si][p];
            c->peer_row_seen[p] = 1;
            es->pend_seen[si][p] = 0;
            es->demand_got++;
          }
          es->pend_cnt[si] = 0;
        }

        // ── O2 demand hint:全局矩阵从上次的恒等推导(SAME 缩放 / TRANSPOSE)。
        // trust 档命中 → 免等 allgather(协议要求"拿到全局矩阵",恒等推导与
        // 重新收集在数学上等价 —— 实现自由,不是改协议)。
        // **trust 免等只对 MOSAIC 调度器安全**:全库的一条隐藏不变量是
        // "收到 peer 的 demand 行 ⇒ peer 已提交 ⇒ peer 的 recv 缓冲已切到本次
        // collective"。src 驱动的调度器(direct/FAST:拿到矩阵就直接发)靠
        // "等满 n-1 行"隐式满足它;免等会把下一次 collective 的数据推给还停在
        // 上一次缓冲的 dst(实测:direct autograd 反向 bit-exact 挂,数据落旧
        // recvbuf 地址)。MOSAIC 是 dst 驱动:数据只在 dst 发出本次策略之后
        // 流动,策略又带 cid 寄存(上方 CT_POLICY),结构上免疫 —— 所以
        // direct/FAST 一律降为 verify 档(FAST 官方运行时本就全量交换 splits)。
        int hint = opt_demand_hint();
        int dm = hint ? c->demand_mode : 0;
        const bool verify_only = (hint == 2) ||
                                 c->cfg.sched != MOSAIC_SCHED_MOSAIC;
        if (dm != 0 && c->demand_prev_exact && c->row_bytes_prev > 0 && c->row_bytes_cur > 0) {
          const uint64_t rp = c->row_bytes_prev, rcur = c->row_bytes_cur;
          std::vector<uint64_t> M((size_t)n * n);
          bool ok = true;
          for (int s = 0; s < n && ok; s++)
            for (int d2 = 0; d2 < n; d2++) {
              uint64_t v = c->demand_prev[dm == 2 ? (size_t)d2 * n + s : (size_t)s * n + d2];
              if (v % rp) { ok = false; break; }
              M[(size_t)s * n + d2] = v / rp * rcur;
            }
          const int me2 = c->cfg.rank;
          if (ok)
            for (int t = 0; t < n; t++) {
              if (M[(size_t)me2 * n + t] != c->scounts[t] ||
                  M[(size_t)t * n + me2] != c->rcounts[t]) { ok = false; break; }
            }
          if (!ok) {
            // trust 档失配 = 硬错:带病继续会按错误矩阵指挥对端发数据(越界读)。
            // verify 档失配走 E4006 警告路径(coll_begin),这里只拦 trust。
            if (!verify_only) {
              MLOG_ERR("E4005 demand hint(dm=%d)推导失配:负载不满足恒等前提"
                       "(token dropping / splits 变形?)。设 MOSAIC_DEMAND_HINT=0 关闭。", dm);
              c->err = "E4005 demand hint 失配(设 MOSAIC_DEMAND_HINT=0)";
              es->demand_sent = false;   // 状态复位,下次 collective 从头来
              es->demand_got = 0;
              c->coll_state.store(3);
              continue;
            }
          } else if (verify_only) {
            c->demand_check = std::move(M);
            c->demand_check_on = true;              // coll_begin 收齐后对拍
          } else {
            c->demand = std::move(M);               // trust:整矩阵采用,免等
            es->demand_got = n - 1;
            MLOG_DBG("demand hint trust dm=%d:免 allgather 等待(coll=%d)", dm, c->coll_id);
          }
        }

        // ── P1 SSP demand(算法级):hint 没接住的调用(dispatch)也不等。
        // 注水的硬需求只有本地列(= rcounts,调用方给了);全矩阵的消费者只有
        // PACE/PC 两个形状门 → 用上次矩阵(有界陈旧)。行广播降级为后台刷新。
        // 仅 mosaic(dst 驱动,安全性与 hint trust 同一条论证);verify 挂起时
        // 不启用(用户在诊断);首个 collective 无历史 → 回退等待。
        if (opt_demand_ssp() && c->cfg.sched == MOSAIC_SCHED_MOSAIC &&
            es->demand_got < n - 1 && !c->demand_check_on && c->demand_prev_valid) {
          for (int s = 0; s < n; s++)
            if (s != c->cfg.rank) c->demand[(size_t)s * n + c->cfg.rank] = c->rcounts[s];
          c->ssp_active = true;
          c->pace_use_prev = true;
          MLOG_DBG("SSP:本地列即开(coll=%d,行 %d/%d 后台补)", c->coll_id,
                   es->demand_got, n - 1);
        }
      }
      g_ev_phase = 15;
      { static thread_local int last_cid = -1;
        if (last_cid != c->coll_id) { last_cid = c->coll_id;
          MLOG_DBG("demand 块尾(got=%d ssp=%d)", es->demand_got, (int)c->ssp_active); } }
      if (es->demand_got >= n - 1 || c->ssp_active) coll_begin(c);
    }
    seg(2); try_self_copy(c); pump_devpipe(c); pump_wait_relay(c);
    seg(3); pump_shm(c);
    seg(4); pump_ready(c);
    seg(5); pump_tx(c);
    seg(6); pump_rx(c);
    seg(7); pump_credits(c); pump_protocol(c);
    // 消费者核 in-vivo 服务曲线采样(200µs;孤立微基准 20+GB/s vs 引擎语境
    // ~2-4GB/s 的落差归因用,local-0821-kcons-rate)
    if (g_trace_on && c->kdesc_active) {
      static thread_local uint64_t next_ks = 0;
      uint64_t tn = now_ns();
      if (tn > next_ks) {
        next_ks = tn + 200000;
        MTRACE(TE_KDESC_SAMP, 0, c->kdesc_pushed,
               __atomic_load_n((uint64_t*)(c->kdesc_ctr + 64), __ATOMIC_ACQUIRE));
      }
    }
    seg(8); coll_try_finish(c);
    seg(9); watchdog(c);
    seg(0);
  }
  if (c->erpc) erpc_ctrl_teardown(c);   // 与 setup 同线程(Rpc 线程绑定)
}

// demand_got 的语义修正:广播计 1,自己那行不算,收齐 = n(自己 1 + 他人 n-1)
// 见 event_loop 里的 max(…,1)。

void engine_start(Comm* c) {
  auto* es = ES(c);
  es->pend_rows.assign(PEND_SLOTS, {});
  es->pend_seen.assign(PEND_SLOTS, std::vector<uint8_t>(c->cfg.nranks, 0));
  es->pend_cnt.assign(PEND_SLOTS, 0);
  es->pend_cid.assign(PEND_SLOTS, -1);
  for (int i = 0; i < PEND_SLOTS; i++) es->fin_early_cid[i] = -1;   // -1 = 空槽
  // 跨代安全性:数据只在两种前提下流动 —— ①direct/FAST:src 等满 n-1 行
  // demand(含对端的)才 coll_begin;②mosaic:dst 发出本次策略后 src 才发。
  // 两条都蕴含"对端已进入本 collective",无需对端在场门。
  // SliceHdr.coll 代际字段与 E5024 告警是这条不变量的哨兵。
  es->peer_acc.assign(c->cfg.nranks, std::vector<uint64_t>(c->cfg.nranks * RES_KINDS, 0));
  c->peer_disch_seq.assign(c->cfg.nranks, 0);
  c->demand.assign((size_t)c->cfg.nranks * c->cfg.nranks, 0);
  // 必须在 event 线程启动前就位:CT_POLICY/CT_TELEM 可能早于第一次 coll_begin
  // 到达(early_policy 路径),届时就要索引这些表(踩过:rank 启动即段错误,
  // 日志 0 字节 —— log.md 坑 #15)
  c->peer_wave.assign(c->cfg.nranks, 0);
  c->peer_fin.assign(c->cfg.nranks, 0);
  c->peer_corr_intent.assign(c->cfg.nranks, 0);   // 走廊承诺位(demand 块复位/填)
  c->peer_row_seen.assign(c->cfg.nranks, 0);      // 行门(src 相发送门,demand 块复位/填)
  c->pace_cold.assign(c->cfg.nranks, 0);
  c->peer_det.assign(c->cfg.nranks, 0);
  c->peer_fab.assign((size_t)c->cfg.nranks * 2 * 8, 0);
  c->peer_intent.assign(c->cfg.nranks, INTENT_FULL);   // P2:意图未到前保守全量

  if (c->cfg.use_gpu) {
    // 选卡 = rank % G(块状放置:同节点 G 个 rank 连号)。Raven 形态是整节点
    // 4 卡全可见 + 代码侧绑卡(官方手册:"user code needs to attach its tasks
    // to the different GPUs based on some code-internal logic";不能用
    // --gpus-per-task=1 —— cgroup 藏掉兄弟卡会弄死机内 cudaIpcOpenMemHandle)。
    // 本机 G=1 时恒为 0,行为不变。MOSAIC_CUDA_DEV 显式覆盖(异形拓扑逃生口)。
    // 卡不够分时回退共享(% ndev):本机 1 卡跑 G=4 的单机烟测(regression
    // 第 3 项)必须保持旧行为 dev=0,不能选出不存在的 device。
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int dev = c->cfg.rank % c->cfg.gpus_per_host;
    if (ndev > 0) dev %= ndev;
    if (const char* e = getenv("MOSAIC_CUDA_DEV")) { if (*e) dev = atoi(e); }
    c->cuda_dev = dev;
    CUCHK(cudaSetDevice(dev), "setDevice");
    if (dev != 0)
      MLOG_INFO("CUDA 选卡 %d(rank %d %% G=%d)", dev, c->cfg.rank, c->cfg.gpus_per_host);
    cudaStream_t s1, s2;
    CUCHK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream d2h");
    CUCHK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), "stream h2d");
    c->stream_d2h = s1; c->stream_h2d = s2;
    // kdesc 数据面资源:消费者内核描述符环 + 计数块 + prog。
    // 任一失败 → zs_cap=false,提交路径退回阻塞 + 流拷贝。
    {
        auto kdesc_setup = [&]() -> bool {
          if (!kcons_init(c)) return false;
          const char* e = getenv("MOSAIC_KDESC_SLOTS");
          uint32_t cap = e && *e ? (uint32_t)atoi(e) : 4096;
          while (cap & (cap - 1)) cap &= cap - 1;          // 取 2 的幂
          if (cap < 64) cap = 64;
          void* ring = nullptr; void* ctr = nullptr;
          void* ring_d = nullptr; void* ctr_d = nullptr; void* prog_d = nullptr;
          if (cudaHostAlloc(&ring, (size_t)cap * sizeof(Comm::KDesc),
                            cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
            return false;
          if (cudaHostAlloc(&ctr, 256, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
            cudaFreeHost(ring); return false;
          }
          if (cudaHostGetDevicePointer(&ring_d, ring, 0) != cudaSuccess ||
              cudaHostGetDevicePointer(&ctr_d, ctr, 0) != cudaSuccess ||
              cudaMalloc(&prog_d, 96 * sizeof(uint64_t)) != cudaSuccess) {   // 64 prog + 16 chgo + 余量
            cudaFreeHost(ring); cudaFreeHost(ctr); return false;
          }
          cudaMemset(prog_d, 0, 96 * sizeof(uint64_t));
          memset(ctr, 0, 256);   // +0 head/+64 done/+128 stop/+160 input/+192 gen
          ((uint64_t*)((uint8_t*)ctr + 128))[0] = ~0ull;   // stop = 进行中
          c->kdesc_ring = (Comm::KDesc*)ring;
          c->kdesc_ring_dev = (uint64_t)(uintptr_t)ring_d;
          c->kdesc_ctr = (uint8_t*)ctr;
          c->kdesc_ctr_dev = (uint64_t)(uintptr_t)ctr_d;
          c->kdesc_prog_dev = (uint64_t)(uintptr_t)prog_d;
          c->kdesc_cap = cap;
          cudaStream_t sc2 = nullptr;
          if (cudaStreamCreateWithFlags(&sc2, cudaStreamNonBlocking) == cudaSuccess)
            c->stream_cons = sc2;
          return true;
        };
        if (kdesc_setup()) {
          c->zs_cap = true;
          MLOG_DBG("零流数据面可用(消费者核 %u 槽)", c->kdesc_cap);
          // 流序完成资源:done_flag(pinned mapped)+ 门核 JIT。任一失败
          // ⇒ done_flag_dev 置空,提交路径退回阻塞语义。
          void* p = nullptr;
          void* dp = nullptr;
          if (cudaHostAlloc(&p, 4, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess &&
              cudaHostGetDevicePointer(&dp, p, 0) == cudaSuccess && gate_init(c)) {
            *(uint32_t*)p = 0;
            c->done_flag_host = p;
            c->done_flag_dev = (uint64_t)(uintptr_t)dp;
          } else if (p) {
            cudaFreeHost(p);
          }
        }
    }
  }
  // staging(pinned host)与中转池(GPU)
  int NSHM = getenv("MOSAIC_SHM_SLOTS") ? atoi(getenv("MOSAIC_SHM_SLOTS")) : 16;
  const int NST = getenv("MOSAIC_STAGE_SLOTS") ? atoi(getenv("MOSAIC_STAGE_SLOTS")) : 128;
  int NRE = getenv("MOSAIC_RELAY_SLOTS") ? atoi(getenv("MOSAIC_RELAY_SLOTS")) : 128;
  // P3 空间预算(MOSAIC_MEM_BUDGET_MB):GPU 侧池(relay + 机内 IPC 环)一键
  // 权衡 —— 深流水 vs 省显存给模型。显式槽数 env 优先;staging 是锁页 host
  // 内存、NVSHMEM 对称堆有自己的 NVSHMEM_SYMMETRIC_SIZE,均不在此账。
  {
    const char* mb = getenv("MOSAIC_MEM_BUDGET_MB");
    if (mb && *mb && c->cfg.use_gpu) {
      const size_t budget = (size_t)atoll(mb) << 20;
      const size_t sb2 = (size_t)c->cfg.slice_bytes;
      const int peers_nv = c->cfg.gpus_per_host - 1;
      if (!getenv("MOSAIC_RELAY_SLOTS"))
        NRE = std::max(8, (int)(budget / 2 / sb2));
      if (!getenv("MOSAIC_SHM_SLOTS") && peers_nv > 0)
        NSHM = std::max(4, (int)(budget / 2 / sb2 / (size_t)peers_nv));
      MLOG_INFO("空间预算 %sMB → relay=%d 槽,IPC 环=%d 槽/peer(slice=%dB)",
                mb, NRE, NSHM, c->cfg.slice_bytes);
    }
  }
  size_t sb = (size_t)c->cfg.slice_bytes;
  c->stage_slots = NST; c->relay_slots = NRE;
  if (c->cfg.use_gpu) {
    // host 头池(64B/槽,CPU 写头 + RDMA SGE0)+ host pinned payload 池
    // (Mapped:消费者核经 UVA 直写,NIC 从 host MR 读;见 internal.h 注)
    CUCHK(cudaHostAlloc((void**)&c->stage_hdr_pool, (size_t)NST * HDR_STRIDE,
                        cudaHostAllocPortable), "hostAlloc stage hdr");
    CUCHK(cudaHostAlloc(&c->stage_pay_pool, sb * NST,
                        cudaHostAllocPortable | cudaHostAllocMapped),
          "hostAlloc stage pay");
    CUCHK(cudaMalloc(&c->relay_pool, sb * NRE), "malloc relay");
    // NV 管的 payload 环:恒在设备内存(CUDA IPC,GPU↔GPU 直达)。
    // 句柄经环头(共享内存)交给对端。
    {
      const size_t ring_bytes = (size_t)NSHM * (size_t)c->cfg.slice_bytes;
      for (int p = 0; p < c->cfg.nranks; p++) {
        if (!c->tx_ring[p]) continue;
        void* dev = nullptr;
        CUCHK(cudaMalloc(&dev, ring_bytes), "malloc nv ring");
        cudaIpcMemHandle_t hnd;
        cudaError_t e = cudaIpcGetMemHandle(&hnd, dev);
        if (e != cudaSuccess) { MLOG_ERR("E5003 cudaIpcGetMemHandle: %s", cudaGetErrorString(e)); return; }
        shm_ring_set_dev(c->tx_ring[p], dev);
        shm_ring_put_ipc(c->tx_ring[p], &hnd, sizeof(hnd));
      }
      for (int p = 0; p < c->cfg.nranks; p++) {
        if (!c->rx_ring[p]) continue;
        cudaIpcMemHandle_t hnd;
        if (!shm_ring_get_ipc(c->rx_ring[p], &hnd, sizeof(hnd))) {
          MLOG_ERR("E5004 等 rank %d 的 IPC 句柄超时", p); return;
        }
        void* dev = nullptr;
        cudaError_t e = cudaIpcOpenMemHandle(&dev, hnd, cudaIpcMemLazyEnablePeerAccess);
        if (e != cudaSuccess) { MLOG_ERR("E5005 cudaIpcOpenMemHandle(r%d): %s", p, cudaGetErrorString(e)); return; }
        shm_ring_set_dev(c->rx_ring[p], dev);
      }
      MLOG_INFO("NV 管 = CUDA IPC 设备环(%zu KB/环)", ring_bytes >> 10);
    }
    // (SHM 段不再钉页:payload 恒在设备内存,共享内存只剩头/序号,CPU 直读写)
  } else {
    // 纯 host 协议档(离线回放/无 GPU):机内环的 payload 需要设备内存,
    // 有机内 peer 的形态不支持 —— fail-fast,别让它跑到 SIGSEGV。
    for (int p = 0; p < c->cfg.nranks; p++)
      if (c->tx_ring[p] || c->rx_ring[p]) {
        MLOG_ERR("E5030 use_gpu=0 但存在机内 peer(G>1):该形态不支持");
        c->err = "E5030 use_gpu=0 不支持机内 peer";
        return;
      }
    c->stage_hdr_pool = (uint8_t*)malloc((size_t)NST * HDR_STRIDE);
    c->relay_pool = malloc(sb * NRE);
  }
  c->stage_free.clear(); c->relay_free.clear();
  for (int i = NST - 1; i >= 0; i--) c->stage_free.push_back(i);
  for (int i = NRE - 1; i >= 0; i--) c->relay_free.push_back(i);

  // ── 跨机数据面 = RDMA verbs(TX/RX 双侧 host-bounce,见 verbs.cc 头注)。
  // 建链失败 = 硬错 fail-fast(无 TCP 回退,不静默降速)。
  //
  // 跨机控制面有两档,正交互斥(见 internal.h 的 ctrl_verbs / ctrl_erpc):
  //   RDMA 档  —— 复用同一次 verbs 建链的独立 ctrl QP,单程 ~5µs;
  //   兼容档   —— eRPC/DPDK 用户态以太网(erpc_ctrl.cc),普通以太网卡即可,
  //               给没有 RDMA 卡的机器兜底。
  // 这里只**选档并置标志**;兼容档的真正建链在 event 线程(erpc::Rpc 与创建
  // 线程绑定),见 event_loop 开头。
  {
    bool has_remote = false;
    for (int p = 0; p < c->cfg.nranks; p++)
      if (p != c->cfg.rank && !c->topo.same_host(c->cfg.rank, p)) { has_remote = true; break; }
    if (has_remote) {
      // MOSAIC_CTRL:跨机控制面选哪一档。**非特殊情况一律用 verbs 档**。
      //   auto(默认)—— 能力探测:verbs 建得起来就用 RDMA 档,否则退兼容档;
      //   verbs      —— 钉死 RDMA 档,建不起来直接硬错(不静默退档);
      //   erpc       —— 钉死兼容档,只在两种情况下用:①机器没有 RDMA 卡;
      //                 ②要在有卡的机器上做两档对比。日常不要设它。
      // 为什么默认偏向 verbs:两档控制面延迟其实同级(本机实测 verbs UD
      // 3.04µs vs eRPC/DPDK 往返 6.06µs,都远优于 kernel TCP 的 p50 35µs,
      // 见 log_raven.md 29348237-ctrl 与 local-0820-erpc-dpdk-ctrl),
      // 但兼容档要 CAP_NET_RAW + hugepages + flow-isolated 独占队列组,
      // 且每机只能有一个进程用(DPDK 单 primary 限制,G>1 会失败回退 TCP)。
      // verbs 档这些代价一个都没有 —— 有卡就没有理由选兼容档。
      const char* ctrl_mode = getenv("MOSAIC_CTRL");
      const bool pin_verbs = ctrl_mode && strcmp(ctrl_mode, "verbs") == 0;
      const bool pin_erpc = ctrl_mode && strcmp(ctrl_mode, "erpc") == 0;
      if (ctrl_mode && !pin_verbs && !pin_erpc && strcmp(ctrl_mode, "auto") != 0) {
        MLOG_ERR("E1033 MOSAIC_CTRL=%s 无效(只能是 auto/verbs/erpc)", ctrl_mode);
        c->err = "E1033 MOSAIC_CTRL 取值无效";
        return;
      }
      if (verbs_setup(c, /*data_plane=*/true) == 0) {
        c->use_verbs = true;
        if (pin_erpc) {
          c->ctrl_erpc = true;
          MLOG_INFO("跨机数据面 = RDMA verbs(host-bounce)+ 控制面兼容档(eRPC/DPDK)");
        } else {
          c->ctrl_verbs = true;
          MLOG_INFO("跨机数据面 = RDMA verbs(host-bounce)+ 控制面 RDMA SEND");
        }
      } else {
        verbs_teardown(c);
        MLOG_ERR("E1028 verbs 建链失败(数据面无 TCP 回退,硬错)");
        c->err = "E1028 verbs 建链失败";
        return;
      }
      // NIC 物理单元账本实例化(log_raven.md 2026-08-21「资源单元」条):
      // 共享 HCA 的节点(Raven:1×HDR200 喂 4 GPU)必须一根管子一本账,
      // 否则注水把同一根管当 4 根独立管错误分流(W6C6 8M uniform 5.4× 灾难
      // 的根因)。cap 语义:**按物理单元口径** —— MOSAIC_NIC_GBPS>0 = 显式
      // 覆盖单元口速率;0(默认)= 用 verbs 探测的物理口速率(HDR200=200)。
      nic_unit_exchange(c);
      {
        c->ledger.disch_agg.assign(c->topo.nres(), {});
        for (int r = 0; r < c->cfg.nranks; r++) {
          int lead = c->topo.nic_leader[r];
          double g = (lead == r)
                         ? (c->cfg.nic_gbps > 0 ? c->cfg.nic_gbps : c->nic_unit_gbps[r])
                         : 0;
          c->ledger.cap_Bps[c->topo.res(r, RES_NIC_UP)] = g * 1e9 / 8.0;
          c->ledger.cap_Bps[c->topo.res(r, RES_NIC_DN)] = g * 1e9 / 8.0;
          if (lead != r) {   // 成员的 discharge 计入 leader 位的在途扣减
            c->ledger.disch_agg[c->topo.res(lead, RES_NIC_UP)]
                .push_back(c->topo.res(r, RES_NIC_UP));
            c->ledger.disch_agg[c->topo.res(lead, RES_NIC_DN)]
                .push_back(c->topo.res(r, RES_NIC_DN));
          }
        }
      }
    }
  }

  uint64_t t = now_ns();
  // 真机为准:nv/nic/spine 额定值只喂决策层(账本 cap、PACE、波 deadline),
  // 数据面永不限速;Bucket 只做 discharge 计数。
  for (int k = 0; k < RES_KINDS; k++) c->buck[k].init();

  // 检测器与注入状态
  c->nominal_Bps = c->ledger.cap_Bps;             // 通告额定(检测器恢复基线)
  c->peer_det.assign(c->cfg.nranks, 0);
  {
    // 检测器额定(单元感知):私有 NIC 单元 = 单元口速率(覆盖值或探测值);
    // **共享单元 = 0(禁用)** —— 检测器测的是本 rank 自己的吞吐,共享 HCA 下
    // 那只是单元的一个份额,拿它对照单元额定必然误锁降速。检测器的共享单元
    // 聚合版留决策层阶段做。nom<=0 时 det_feed 直接返回(engine.cc det_feed 头)。
    double dn = 0;
    if (c->topo.nic_leader[c->cfg.rank] == c->cfg.rank) {
      bool shared = false;
      for (int r = 0; r < c->cfg.nranks; r++)
        if (r != c->cfg.rank && c->topo.nic_leader[r] == c->cfg.rank) { shared = true; break; }
      if (!shared)
        dn = c->cfg.nic_gbps > 0 ? c->cfg.nic_gbps : c->nic_unit_gbps[c->cfg.rank];
    }
    for (int k = 0; k < 2; k++) {
      c->det[k].nom_gbps = dn;
      c->det[k].rate_gbps = dn;
      c->st.det_rate_gbps[k] = dn;
    }
  }
  c->rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)(c->cfg.rank + 1);
  c->drop_a = getenv("MOSAIC_DROP_A") ? atof(getenv("MOSAIC_DROP_A")) : 0.0;
  if (c->cfg.spineagg)
    MLOG_DBG("fabric 账 = spineagg 聚合档(emu/Raven 均为无 spine 选择的 fabric)");

  c->telem_next_ns = t;
  c->running.store(true);
  c->ev_thread = std::thread(event_loop, c);
  // P3 CPU 算力安排:event 线程绑核(与训练 dataloader / SLURM cpu-bind 解耦;
  // 100µs 节拍抖动 p99 是它的效率红线,被抢核的症状就是抖动超标)
  {
    const char* e = getenv("MOSAIC_CPU_CORE");
    if (e && *e && atoi(e) >= 0) {
      cpu_set_t cs;
      CPU_ZERO(&cs);
      CPU_SET(atoi(e), &cs);
      if (pthread_setaffinity_np(c->ev_thread.native_handle(), sizeof(cs), &cs) == 0)
        MLOG_INFO("event 线程绑核 %d(MOSAIC_CPU_CORE)", atoi(e));
      else
        MLOG_WARN("E5020 绑核 %d 失败(核不存在或 cgroup 限制)", atoi(e));
    }
  }
}

void engine_stop(Comm* c) {
  c->running.store(false);
  if (c->ev_thread.joinable()) c->ev_thread.join();
  // 消费者核收摊:写 stop(排干即退)+ gen++(核见变即退),再等流空
  if (c->kdesc_ctr) {
    __atomic_store_n((uint64_t*)(c->kdesc_ctr + 192), ++c->kdesc_gen, __ATOMIC_RELEASE);
    __atomic_store_n((uint64_t*)(c->kdesc_ctr + 128), c->kdesc_pushed, __ATOMIC_RELEASE);
    if (c->stream_cons) cudaStreamSynchronize((cudaStream_t)c->stream_cons);
  }
  if (c->verbs) { verbs_teardown(c); c->use_verbs = false; }
  if (c->cfg.use_gpu) {
    for (void* e : c->ev_pool) cudaEventDestroy((cudaEvent_t)e);
    if (c->done_flag_host) { cudaFreeHost(c->done_flag_host); c->done_flag_host = nullptr; }
    if (c->kdesc_ring) { cudaFreeHost(c->kdesc_ring); c->kdesc_ring = nullptr; }
    if (c->kdesc_ctr) { cudaFreeHost(c->kdesc_ctr); c->kdesc_ctr = nullptr; }
    if (c->kdesc_prog_dev) { cudaFree((void*)(uintptr_t)c->kdesc_prog_dev); c->kdesc_prog_dev = 0; }
    if (c->stream_cons) { cudaStreamDestroy((cudaStream_t)c->stream_cons); c->stream_cons = nullptr; }
    if (c->stage_hdr_pool) cudaFreeHost(c->stage_hdr_pool);
    if (c->stage_pay_pool) cudaFreeHost(c->stage_pay_pool);
    if (c->relay_pool) cudaFree(c->relay_pool);
    if (c->stream_d2h) cudaStreamDestroy((cudaStream_t)c->stream_d2h);
    if (c->stream_h2d) cudaStreamDestroy((cudaStream_t)c->stream_h2d);
  } else {
    free(c->stage_hdr_pool); free(c->relay_pool);
  }
  ES_drop(c);
}

}  // namespace mosaic
