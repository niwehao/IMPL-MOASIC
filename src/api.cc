/* api.cc —— L3 公开 API 的实现。app 线程与 event 线程之间只有极少量交接:
 * 提交(设置缓冲区 + coll_state=1)与等待(自旋到 coll_state=2)。 */
#include "internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <csignal>
#include <execinfo.h>
#include <sched.h>
#include <unistd.h>

using namespace mosaic;

// SIGTERM(SLURM 超时)= 转储后退出(06 §5.3「卡死必留尸检」)。handler 里的
// fprintf 严格说不是 async-signal-safe —— 最后一搏的尸检输出,声明后冒此险。
static Comm* volatile g_term_comm = nullptr;
static void on_sigterm(int) {
  if (g_term_comm) dump_state((Comm*)g_term_comm, "SIGTERM");
  _exit(143);
}
// SIGSEGV 也要留尸检:异步档下崩在哪个线程/哪一帧,靠这段 backtrace 定位
// (Raven 不可交互调试,core dump 也不落盘)。同样不是 async-signal-safe,
// 最后一搏。
//
// **必须走 sigaltstack + SA_ONSTACK**(坑 #44):handler 默认在**出错线程自己的
// 栈**上运行,而 SEGV 最常见的一类原因恰恰是栈溢出(深递归/巨大栈对象)——
// 那种情况下没有栈可用于跑 handler,内核直接二次 SEGV 杀进程,**一个字都不留**。
// 现场表现极具误导性:torchrun 只报 "Signal 11 received by PID …, error_file:
// <N/A>",看起来像 handler 没装(29355321-185139 实测 r15)。另外 sigaction 比
// signal() 明确:SA_NODEFER 关掉才不会在 handler 内二次 SEGV 时无限套娃。
static void on_sigsegv(int sig, siginfo_t* si, void*) {
  void* bt[64];
  int n = backtrace(bt, 64);
  char hdr[160];
  int hl = snprintf(hdr, sizeof(hdr),
                    "[mosaic r%d] **SIG%d**(addr=%p)backtrace(%d 帧):\n",
                    g_rank_for_log, sig, si ? si->si_addr : nullptr, n);
  ssize_t ign = write(2, hdr, (size_t)hl); (void)ign;   // write 才是 signal-safe
  backtrace_symbols_fd(bt, n, 2);
  if (g_term_comm) dump_state((Comm*)g_term_comm, "SIGSEGV");
  _exit(139);
}
// 崩溃类信号统一挂上 alternate stack;每个线程各自 sigaltstack(内核按线程记账)
static void install_crash_handlers() {
  // 新 glibc 的 SIGSTKSZ 是 sysconf() 不是常量,不能做数组维度 —— 给个够大的定值
  static thread_local char altstack[256 * 1024];     // 深 backtrace 也够
  stack_t ss{};
  ss.ss_sp = altstack; ss.ss_size = sizeof(altstack); ss.ss_flags = 0;
  sigaltstack(&ss, nullptr);
  struct sigaction sa {};
  sa.sa_sigaction = on_sigsegv;
  sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESETHAND;   // RESETHAND 防套娃
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
}
void mosaic::mosaic_install_crash_handlers_this_thread() { install_crash_handlers(); }

extern "C" mosaicResult_t mosaicCommInit(mosaicComm_t* out, const mosaicConfig_t* cfg) {
  if (!out || !cfg) return mosaicErrConfig;
  if (cfg->nranks <= 0 || cfg->nranks > MAX_RANKS) return mosaicErrConfig;
  if (cfg->rank < 0 || cfg->rank >= cfg->nranks) return mosaicErrConfig;
  if (cfg->gpus_per_host <= 0) return mosaicErrConfig;

  // 对端先退出时,继续写它的 socket 会收到 SIGPIPE 而**直接杀死进程**
  // (默认动作是 Term)。慢的那个 rank 因此死在别人前头,现场全丢、
  // 退出码变成 141,极具误导性。忽略它,让 write 正常返回 EPIPE 走错误路径。
  signal(SIGPIPE, SIG_IGN);
  mtrace_init(cfg->rank);
  Comm* c = new Comm();
  c->cfg = *cfg;
  g_rank_for_log = cfg->rank;
  topo_init(&c->topo, cfg->nranks, cfg->gpus_per_host, cfg->rank);
  ledger_init(c);
  c->scounts.assign(cfg->nranks, 0);
  c->sdispls.assign(cfg->nranks, 0);
  c->rcounts.assign(cfg->nranks, 0);
  c->rdispls.assign(cfg->nranks, 0);

  if (net_setup(c) != 0) { c->err = "E1001 建链失败"; delete c; return mosaicErrTransport; }
  engine_start(c);
  if (!c->err.empty()) {          // 数据面建链硬错(无 TCP 回退,fail-fast)
    MLOG_ERR("CommInit 失败:%s", c->err.c_str());
    engine_stop(c);
    net_teardown(c);
    delete c;
    return mosaicErrTransport;
  }
  MLOG_INFO("comm 就绪:rank %d/%d sched=%s C=%d W=%d nv=%.1fG nic=%.1fG slice=%dB",
            cfg->rank, cfg->nranks, mosaicSchedName(cfg->sched), cfg->corridors,
            cfg->waves, cfg->nv_gbps, cfg->nic_gbps, cfg->slice_bytes);
  g_term_comm = c;
  signal(SIGTERM, on_sigterm);
  install_crash_handlers();      // 本线程(app 线程);event 线程在自己起头时再装一次
  *out = (mosaicComm_t)c;
  return mosaicSuccess;
}

static mosaicResult_t wait_coll(Comm* c);

extern "C" mosaicResult_t mosaicCommDestroy(mosaicComm_t comm) {
  Comm* c = (Comm*)comm;
  if (!c) return mosaicErrConfig;
  if (c->coll_state.load() != 0) wait_coll(c);   // 在途必收尾(防御,正常恒空)
  engine_stop(c);
  net_teardown(c);
  if (c->input_ev) cudaEventDestroy((cudaEvent_t)c->input_ev);
  delete c;
  return mosaicSuccess;
}

// 等本笔 collective 收尾(阻塞语义:每轮 A2A 结束必有同步,硬不变量)。
static mosaicResult_t wait_coll(Comm* c) {
  const uint64_t t_deadline = now_ns() + 120ull * 1000000000ull;
  for (;;) {
    int s = c->coll_state.load();
    if (s == 0) return mosaicSuccess;             // 无在途
    if (s == 2) break;
    if (s == 3) {                                 // 协议错误(E4005/E1027 等)
      // 错误路径也要停核 + 开门,否则消费者核/门核悬死、下次 cudaFree 卡住
      kdesc_release_gate(c);
      if (c->done_flag_host)
        __atomic_store_n((uint32_t*)c->done_flag_host, (uint32_t)c->coll_id, __ATOMIC_RELEASE);
      c->coll_state.store(0);
      return mosaicErrConfig;
    }
    if (now_ns() > t_deadline) {
      kdesc_release_gate(c);   // 同上:超时也要停核 + 开门
      if (c->done_flag_host)
        __atomic_store_n((uint32_t*)c->done_flag_host, (uint32_t)c->coll_id, __ATOMIC_RELEASE);
      c->err = "E2002 collective 超时(120s)";
      MLOG_ERR("%s:recv %llu/%llu send %llu/%llu", c->err.c_str(),
               (unsigned long long)c->dst.total_recv, (unsigned long long)c->total_recv_expect,
               (unsigned long long)c->posted_send, (unsigned long long)c->total_send);
      return mosaicErrTimeout;
    }
    sched_yield();
  }
  c->coll_state.store(0);
  MLOG_DBG("alltoallv %d 返回", c->coll_id);
  return mosaicSuccess;
}

namespace mosaic {
bool zerostream_arm(Comm* c);
bool gate_arm(Comm* c, void* stream, uint32_t want);

// MOSAIC_ZEROSTREAM:无流调用方(bench)也走 kdesc 零流数据面(nvshmem 面的
// 待收敛项,Raven Gate 裁决 —— File/debug-819/local-0819-transport-cleanup-plan.md §2)。
static bool zerostream_on() {
  static const int v = [] {
    const char* e = getenv("MOSAIC_ZEROSTREAM");
    return e ? atoi(e) : 0;
  }();
  return v != 0;
}
bool async_mark_input(Comm* c, void* stream);
}

// 提交(公共体)。阻塞语义:调用方随后经 wait_coll 等收尾。
static mosaicResult_t submit_coll(Comm* c,
                                  const void* sendbuf, const size_t scounts[], const size_t sdispls[],
                                  void* recvbuf, const size_t rcounts[], const size_t rdispls[],
                                  uint64_t caller_stream, int demand_mode, uint64_t row_bytes) {
  const int n = c->cfg.nranks;
  MLOG_DBG("submit 进入(下一 coll=%d)", c->coll_id + 1);
  {
    std::lock_guard<std::mutex> lk(c->mu);
    c->sendbuf = sendbuf; c->recvbuf = recvbuf;
    for (int i = 0; i < n; i++) {
      c->scounts[i] = scounts[i]; c->sdispls[i] = sdispls[i];
      c->rcounts[i] = rcounts[i]; c->rdispls[i] = rdispls[i];
    }
    c->st.bytes_sent = c->st.bytes_recv = c->st.bytes_relayed = 0;
    memset(c->st.corridor_bytes, 0, sizeof(c->st.corridor_bytes));
    memset(c->st.ingress_corr, 0, sizeof(c->st.ingress_corr));
    c->st.chA_msgs = c->st.chA_bytes = c->st.chB_msgs = c->st.chB_bytes = 0;
    c->st.plan_us_total = c->st.plan_us_max = 0; c->st.plan_calls = 0;
    c->st.stagger_us_total = 0; c->st.waves_emitted = 0;
    c->coll_start_ns = 0;
    c->coll_id++;
    // O1:在调用方流上录"输入就绪"事件并**立刻**让引擎的两条拷贝流 WaitEvent
    // (app 线程,提交路径内,先于 coll_state=1)。为什么必须在提交时而不是
    // coll_begin:src 驱动调度器(direct/FAST)的对端数据可以早于我方 coll_begin
    // 到达(ctrl 走 TCP、数据走 RDMA,两通道无序),那些 H2D 写 recvbuf 的拷贝
    // 若先于 WaitEvent 入流,就与 torch caching allocator 复用块上还没执行的
    // kernel 竞争(实测:direct 85 iters 后 loss 尾数漂移)。对端数据最早也要
    // 等我方广播 demand 行(提交之后)才可能到 —— 在提交时挂 wait 时序闭合。
    c->demand_mode = demand_mode;
    c->row_bytes_cur = row_bytes;
    c->ph_submit_ns = now_ns();
    c->ph_first_post_ns = 0;
    c->input_ev_pending.store(0);
    c->caller_stream = caller_stream;
    if (caller_stream && c->cfg.use_gpu) {
      if (!c->input_ev) {
        cudaEvent_t e;
        if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) e = nullptr;
        c->input_ev = (void*)e;
      }
      if (c->input_ev &&
          cudaEventRecord((cudaEvent_t)c->input_ev, (cudaStream_t)caller_stream) == cudaSuccess) {
        cudaStreamWaitEvent((cudaStream_t)c->stream_d2h, (cudaEvent_t)c->input_ev, 0);
        cudaStreamWaitEvent((cudaStream_t)c->stream_h2d, (cudaEvent_t)c->input_ev, 0);
        if (c->use_nvsh) nvsh_wait_input(c, c->input_ev);
      }
    }
    c->gated = false;
    // ── 完成契约(单一路径,按调用方输入分档):
    //   caller_stream != 0(torch 等流式调用方)⇒ **流序完成**:门核排在 caller
    //     流上等 done_flag,本调用提交后即返回;下一次提交先隐式收尾上一笔
    //     (一代在飞)。窗口内引擎零 CUDA 调用(kdesc 消费者核搬运)。
    //   caller_stream == 0(bench 等无流调用方)⇒ host 阻塞等收尾。
    // 流序完成当前只在 verbs 数据面启用:nvsh 面的 kdesc 化(零流窗口)在
    // Raven Gate 收敛前保持阻塞路径(clean-plan §2 待收敛项)。
    const bool stream_ordered = caller_stream != 0 && c->cfg.use_gpu && c->zs_cap &&
                                !c->use_nvsh;
    if (stream_ordered || (zerostream_on() && c->cfg.use_gpu && c->kdesc_ctr)) {
      // ── S3(a):**零流执行进同步档**(MOSAIC_ZEROSTREAM=1,默认关;
      // 待收敛项,Raven Gate 裁决 —— 见清理计划 §2)。
      // 动机:默认档收一片要走:镜像 D2H → 主机轮询发现 → push 描述符 →
      // 消费者核拷贝 → 主机读 done → 回 credit,六次跨主机/设备边界、两处受
      // 轮询周期量化;NCCL/FlashMoE 是零次(设备侧自旋 + 核内直拷)。
      // 同步档下永远只有一代在飞,坑 #41-#48 那一整类代际问题结构上不存在。
      // 没有 caller 流(bench_a2a:数据本就在设备上、无待决生产者)⇒ 输入
      // 天然就绪,不必也无法在流上打就绪标志。
      c->input_ready = !caller_stream;
      if (!caller_stream && c->kdesc_ctr)   // 无流调用方:输入天然就绪,直接开核内输入门
        __atomic_store_n((uint64_t*)(c->kdesc_ctr + 160), 1ull, __ATOMIC_RELEASE);
      const bool mk = caller_stream
                          ? async_mark_input(c, (void*)(uintptr_t)caller_stream)
                          : true;
      if (mk && zerostream_arm(c)) {
        c->kdesc_active = true;      // ⇒ 消费者核交付/组装(nvsh 面另有镜像核)
        if (stream_ordered && gate_arm(c, (void*)(uintptr_t)caller_stream,
                                       (uint32_t)c->coll_id))
          c->gated = true;           // 门已排上 caller 流,Ex 提交后即返回
        MLOG_DBG("kdesc 挂载(coll=%d gated=%d)", c->coll_id, (int)c->gated);
      }
    }
  }
  MLOG_DBG("alltoallv %d 提交(dm=%d)", c->coll_id, demand_mode);
  c->coll_state.store(1);                       // ← event 线程从这里接手
  return mosaicSuccess;
}

extern "C" mosaicResult_t mosaicAlltoallvEx(mosaicComm_t comm,
                                            const void* sendbuf, const size_t scounts[], const size_t sdispls[],
                                            void* recvbuf, const size_t rcounts[], const size_t rdispls[],
                                            uint64_t caller_stream, int demand_mode, uint64_t row_bytes) {
  Comm* c = (Comm*)comm;
  if (!c) return mosaicErrConfig;
  mosaicResult_t rc = wait_coll(c);   // 一代在飞:先收尾上一笔(流序档在此兑现同步)
  if (rc != mosaicSuccess) return rc;
  rc = submit_coll(c, sendbuf, scounts, sdispls, recvbuf, rcounts, rdispls,
                   caller_stream, demand_mode, row_bytes);
  if (rc != mosaicSuccess) return rc;
  if (c->gated) return mosaicSuccess; // 流序完成:门核兜住数据就绪,CPU 即刻返回
  return wait_coll(c);
}

extern "C" mosaicResult_t mosaicAlltoallv(mosaicComm_t comm,
                                          const void* sendbuf, const size_t scounts[], const size_t sdispls[],
                                          void* recvbuf, const size_t rcounts[], const size_t rdispls[]) {
  return mosaicAlltoallvEx(comm, sendbuf, scounts, sdispls, recvbuf, rcounts, rdispls, 0, 0, 0);
}

extern "C" mosaicResult_t mosaicAllgather(mosaicComm_t comm, const void* sendbuf,
                                         size_t bytes, void* recvbuf) {
  Comm* c = (Comm*)comm;
  if (!c || !sendbuf || !recvbuf || bytes == 0) return mosaicErrConfig;
  int ep = c->ag_epoch.fetch_add(1), i = ep & 3;
  {   // 先把自己的那份放好(对端可能已经把它们的发过来了,不能整体清)
    std::lock_guard<std::mutex> lk(c->ag_mu);
    if (c->ag_slot_epoch[i].load() != ep) {
      c->ag_buf.assign((size_t)c->cfg.nranks * bytes, 0);
      c->ag_bytes = bytes;
      c->ag_cnt[i].store(0);
      c->ag_slot_epoch[i].store(ep);
    }
    memcpy(c->ag_buf.data() + (size_t)c->cfg.rank * bytes, sendbuf, bytes);
  }
  ctrl_bcast(c, CT_AG, (uint16_t)ep, sendbuf, (uint32_t)bytes);
  const uint64_t dl = now_ns() + 60ull * 1000000000ull;
  while (c->ag_cnt[i].load() < c->cfg.nranks - 1) {
    if (now_ns() > dl) { c->err = "E2006 allgather 超时"; return mosaicErrTimeout; }
    sched_yield();
  }
  std::lock_guard<std::mutex> lk(c->ag_mu);
  memcpy(recvbuf, c->ag_buf.data(), (size_t)c->cfg.nranks * bytes);
  return mosaicSuccess;
}

extern "C" mosaicResult_t mosaicBarrier(mosaicComm_t comm) {
  Comm* c = (Comm*)comm;
  if (!c) return mosaicErrConfig;
  wait_coll(c);                       // 集合边界:先收尾在飞的 collective
  int ep = c->barrier_epoch.fetch_add(1);
  MLOG_DBG("barrier %d 进入", ep);
  ctrl_bcast(c, CT_BARRIER, (uint16_t)ep, nullptr, 0);
  const uint64_t t_deadline = now_ns() + 60ull * 1000000000ull;
  for (;;) {
    bool all = true;
    for (int p = 0; p < c->cfg.nranks; p++)
      if (p != c->cfg.rank && c->barrier_seen[p].load() <= ep) { all = false; break; }
    if (all) break;
    if (now_ns() > t_deadline) { c->err = "E2004 barrier 超时"; return mosaicErrTimeout; }
    sched_yield();
  }
  MLOG_DBG("barrier %d 通过", ep);
  return mosaicSuccess;
}

static double pctile(const double* ring, int n, int cap, double q) {
  int m = n < cap ? n : cap;
  if (m == 0) return 0;
  std::vector<double> v(ring, ring + m);
  std::sort(v.begin(), v.end());
  size_t i = (size_t)(q * (m - 1) + 0.5);
  return v[i];
}

extern "C" mosaicResult_t mosaicGetStats(mosaicComm_t comm, mosaicStats_t* out) {
  Comm* c = (Comm*)comm;
  if (!c || !out) return mosaicErrConfig;
  // 分位数从环形样本现算(03 §7 效率红线的测量口径)
  c->st.plan_us_p50 = pctile(c->plan_hist, c->plan_hist_n, 256, 0.50);
  c->st.plan_us_p99 = pctile(c->plan_hist, c->plan_hist_n, 256, 0.99);
  c->st.tick_jitter_p99_us = pctile(c->tick_hist, c->tick_hist_n, 256, 0.99);
  *out = c->st;
  return mosaicSuccess;
}

extern "C" mosaicResult_t mosaicStatsSync(mosaicComm_t comm, mosaicStats_t* out) {
  Comm* c = (Comm*)comm;
  if (!c || !out) return mosaicErrConfig;
  wait_coll(c);                       // 读的是收尾后的 completion
  // makespan = 各 rank completion 的最大值。用控制面 allgather 归约(不计入计时)。
  double mine = c->st.completion_us;
  int ep = c->stats_epoch.fetch_add(1), i = ep & 3;
  ctrl_bcast(c, CT_STATS, (uint16_t)ep, &mine, sizeof(mine));
  const uint64_t t_deadline = now_ns() + 30ull * 1000000000ull;
  while (!(c->stats_slot_epoch[i].load() == ep && c->stats_cnt[i].load() >= c->cfg.nranks - 1)) {
    if (c->cfg.nranks == 1) break;
    if (now_ns() > t_deadline) { c->err = "E2005 stats 归约超时"; break; }
    sched_yield();
  }
  c->st.makespan_us = std::max(mine, c->stats_slot_epoch[i].load() == ep ? c->stats_max[i] : mine);
  *out = c->st;
  return mosaicSuccess;
}

extern "C" const char* mosaicGetLastError(mosaicComm_t comm) {
  Comm* c = (Comm*)comm;
  return c ? c->err.c_str() : "no comm";
}
