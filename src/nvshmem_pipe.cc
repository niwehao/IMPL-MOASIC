/* nvshmem_pipe.cc —— NVSHMEM 数据面(Imp_Plan/03 §5;Raven 的正主,本机 2-PE 烟测)。
 *
 * 形态(host-issued,CPU 决策 → GPU 执行,不占训练 SM):
 *   - 对称堆:每 (src peer) 一条接收环 [nslots × (HDR_STRIDE + slice_bytes)]
 *     + signal 数组(uint64/槽,SIGNAL_ADD)+ credit 数组(uint64/peer)。
 *   - 发送:把 [头|payload] 组装进本地设备 staging 槽(头 H2D、payload D2D,
 *     皆 async 在 s_tx 流上),然后一次
 *     nvshmemx_putmem_signal_nbi_on_stream(远端槽, 本地槽, HDR+len, sig, 1, ADD)
 *     —— NVLink 邻居走 P2P,跨节点走 IB proxy,**同一 API**。
 *   - 接收(delivery 检测,03 §5):专用轮询 = cudaMemcpyAsync 把
 *     signal 数组 + 头区镜像到 host shadow(~KB 级),event 完成后 diff 出
 *     新到 slice;payload 停在设备内存,on_slice_arrived 的 cudaMemcpyDefault
 *     直接 D2D 进 recvbuf/relay 池(不过 host)。
 *   - 背压:接收端消费完槽位后把 consumed 计数 put 回发送端的 credit 槽;
 *     发送端 in-flight ≥ nslots 时不再发(与 verbs 的 RNR、TCP 的收窗同角色)。
 *
 * 本机烟测边界(如实):NVSHMEM 不支持多 PE 共享一块 GPU,本机只能
 * 2 机 × 1 PE(G=1);G>1 的机内 NV 管仍走 SHM/IPC。Raven(4 PE/节点,
 * 真 NVLink)上机内跳也切到本管,行为验证在 gpudev(Imp_Plan/06 T1)。
 * 逐 slice CRC 在本管不可用(payload 不过 host);端到端字节校验不受影响。
 *
 * 构建:make NVSHMEM=1(默认不编,零依赖不受影响);bootstrap 用
 * NVSHMEMX_INIT_WITH_UNIQUEID,uid 经既有 TCP 数据连接交换(rank0 产生),
 * 与 FAST 的 get_nvshmem_init_id + dist.broadcast 同构。
 */
#include "internal.h"

#ifdef MOSAIC_HAVE_NVSHMEM

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mosaic {

static int nvsh_slots() {
  const char* e = getenv("MOSAIC_NVSH_SLOTS");
  return e && *e ? atoi(e) : 16;
}

// **零流数据面**(host-API 发送 + 镜像核 + 消费者核)是否启用。
// ((b) 异步提交语义已删,2026-08-19;这里只剩 MOSAIC_ZEROSTREAM 一个入口。)
// 下面几处纪律(staging 槽加深、**不注册 torch 缓冲**)绑的是这条数据面。
static bool s3_zerostream_mode() {
  static const bool v = [] {
    const char* z = getenv("MOSAIC_ZEROSTREAM");
    return (z && atoi(z) != 0);
  }();
  return v;
}

// uid 同机分发用的 /dev/shm 段名(见 nvsh_setup 的 R13 注释)
static void uid_shm_name(char* buf, size_t n) {
  const char* je = getenv("MOSAIC_JOB");
  const char* se = getenv("SLURM_STEP_ID");
  snprintf(buf, n, "/mosaic_uid_%s_%s", je && *je ? je : "0", se && *se ? se : "0");
}

struct NvshCtx {
  int npes = 0, mype = 0;
  size_t slot_bytes = 0;         // HDR_STRIDE + slice_bytes
  int nslots = 0;
  // 对称区(所有 PE 同构):
  uint8_t* ring = nullptr;       // [src_pe][slot] 接收环(本 PE 的)
  uint64_t* sig = nullptr;       // [src_pe][slot] 到达信号
  uint64_t* credit = nullptr;    // [dst_pe] 对端已消费计数(发送端读自己的)
  // 本地:
  uint8_t* tx_stage = nullptr;   // 设备 staging(发送组装)
  int tx_nslots = 0;
  std::vector<int> tx_free;
  std::vector<uint64_t> sent_cnt, cons_seen;      // [peer] 已发 / 见到的 credit
  // 坑 #41:staged 路径的 seq 是在**消费者核拷完之后**(nvsh_post_staged)才
  // 领的,而准入检查在入队时就做了 ⇒ 同一 peer 排在 kq 里的 N 片全都拿陈旧的
  // sent_cnt 过检查(check-then-act),窗口被超发、远端未消费的槽被覆盖
  //(=R15 的丢片机理,另一条触发路径)。占位计数把"已预定未发出"算进窗口。
  std::vector<uint64_t> staged_pend;              // [peer] 已占 staging 槽未发出
  std::vector<uint64_t> rx_consumed;              // [peer] 已回传的**连续前缀**消费数
  // R15b:credit 必须是连续前缀,不能是聚合计数 —— WaitRelay(中转池满时
  // 晚消费)会让消费完成序偏离交付序,聚合计数被推高后发送端会提前复用
  // 未消费的槽(与 R15 同一覆盖机理,另一条触发路径)。乱序距离 ≤ nslots
  //(发送端窗口所限),所以每 peer 一个 nslots 位图 + 游标即可。
  std::vector<uint8_t> rx_done;                   // [peer][slot] 已消费待入前缀
  std::vector<uint64_t> rx_cursor;                // [peer] 连续前缀游标(seq 计)
  // shadow(host pinned):
  uint64_t* sig_shadow = nullptr;
  uint64_t* sig_raw = nullptr;    // 诊断:镜像核无条件写的原始 sig(未经校验)
  uint8_t* hdr_shadow = nullptr;
  uint64_t* credit_shadow = nullptr;
  std::vector<uint64_t> sig_seen;                 // 上次已处理的 signal 值
  std::vector<uint64_t> rx_next;                  // [peer] 下一个待消费 seq(R15:按序消费)
  cudaStream_t s_tx = nullptr, s_poll = nullptr;
  cudaEvent_t ev_poll = nullptr;
  bool poll_inflight = false;
  uint64_t poll_rounds = 0;      // 完成的镜像轮数(watchdog 归因:s_poll 活性)
  uint64_t consume_calls = 0;    // 消费段被执行的次数(区分"没消费"与"消费了但没数据")
  uint64_t host_posts = 0;       // 本轮未 flush 的 host-API put 笔数(quiet 用)
  uint64_t nvsh_phase = 0;       // nvsh_poll 内部阶段(1=tx回收 2=镜像 3=消费)
  // 发送在途登记。tx 槽回收必须 **credit 驱动**:putmem_*_nbi_on_stream 的
  // 流完成只保证操作已发起,IBRC proxy 可能还在 DMA 读源缓冲 —— 按 event 回收
  // 会在高压下把正在被读的槽改写(实测 32MB 半数字节错,log.md 坑 #21)。
  // 对端 credit(=已消费计数)覆盖 seq ⇒ 数据必已完整落对端 ⇒ 源安全。
  // event 只用于 relay 槽释放与检测器计时(D2D 组装完成即可)。
  // O5 zcopy 下 relay 槽释放也必须 credit 驱动(put 直接以 relay 池为源!),
  // TxRec.relay_credit 标记该情况。
  struct TxRec { int slot; cudaEvent_t ev; uint32_t len; int relay_slot;
                 int peer; uint64_t seq; bool ev_done; int hdr_slot = -1;
                 bool relay_credit = false; int hhdr_slot = -1; };
  std::vector<TxRec> tx_pend;
  std::vector<cudaEvent_t> ev_free;   // O6:event 池化(旧版逐 slice create/destroy ~5-10µs)
  // O5 零拷贝(MOSAIC_NVSH_ZCOPY=1):payload 直接以 sendbuf/relay 池为 put 源,
  // 省掉发送侧 D2D staging 拷贝。源必须是 NVSHMEM 注册内存 —— sendbuf 区间经
  // nvshmemx_buffer_register 注册(按指针缓存,torch caching allocator 复用地址
  // 所以命中率高);注册失败该范围自动回退 staging 路径,正确性不受影响。
  bool zcopy = false;
  uint8_t* hdr_stage = nullptr;       // 64B 头槽(对称堆,zcopy 时头单独 put)
  std::vector<int> hdr_free;
  std::vector<std::pair<const void*, size_t>> regs;      // 注册成功的范围
  std::vector<std::pair<const void*, size_t>> reg_fail;  // 失败缓存(避免反复试)
  // S3 二期 host-API 发送:头源池(pinned+注册;RDMA 源纪律 = 坑 #21)
  uint8_t* hhdr_pool = nullptr;
  uint64_t hhdr_dev = 0;          // hhdr_pool 的设备侧地址(消费者核拷头用)
  std::vector<int> hhdr_free;
};

static NvshCtx* NC(Comm* c) { return (NvshCtx*)c->nvsh; }
static bool nvsh_src_registered(NvshCtx* v, const void* p, size_t len);

int nvsh_setup(Comm* c) {
  // NVSHMEM 不支持多 PE 共享一卡。engine_start 已按 rank%G 选卡,这里只验证
  // "卡够分":G ≤ 可见 GPU 数即每 PE 独占一卡(Raven 4 卡/节点 G=4 放行,R11);
  // 本机误配(G>1 但只有 1 卡)仍照旧拦下,保护语义不变。
  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (c->cfg.gpus_per_host > ndev) {
    MLOG_ERR("E5010 NVSHMEM 需每 PE 独占一卡:G=%d > 可见 GPU 数 %d",
             c->cfg.gpus_per_host, ndev);
    return -1;
  }
  NvshCtx* v = new NvshCtx();
  c->nvsh = v;

  // uid:rank0 产生。跨机 peer 经既有 TCP 数据连接分发(与 FAST 同构);
  // **同机 peer 没有 dconn**(net_setup 对 same_host 只建 ctrl 通道 —— 本机
  // 2 机 × 1 PE 形态从未暴露,Raven 4 PE/节点第一跑就撞上,R13),改走
  // /dev/shm 小段:rank0 写 uid + ready 位,同机 sibling 轮询读。段名带
  // SLURM_STEP_ID:同一 sbatch 里 direct/fast/mosaic 是三次 srun(三个 step),
  // 不带的话后一 step 可能读到前一 step 的残段(ready 已是 1)拿到旧 uid。
  struct UidShm { volatile uint32_t ready; nvshmemx_uniqueid_t uid; };
  char shmname[96];
  uid_shm_name(shmname, sizeof shmname);
  nvshmemx_uniqueid_t uid;
  if (c->cfg.rank == 0) {
    nvshmemx_get_uniqueid(&uid);
    bool have_local = false;
    for (int p = 1; p < c->cfg.nranks; p++)
      if (c->topo.same_host(0, p)) { have_local = true; break; }
    if (have_local) {
      int fd = shm_open(shmname, O_CREAT | O_RDWR, 0600);
      if (fd < 0 || ftruncate(fd, sizeof(UidShm)) != 0) {
        MLOG_ERR("E5015 uid shm %s: %s", shmname, strerror(errno));
        if (fd >= 0) close(fd);
        return -1;
      }
      UidShm* s = (UidShm*)mmap(nullptr, sizeof(UidShm),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      close(fd);
      if (s == MAP_FAILED) { MLOG_ERR("E5015 uid mmap: %s", strerror(errno)); return -1; }
      memcpy((void*)&s->uid, &uid, sizeof uid);
      __sync_synchronize();                       // uid 先落,ready 后置
      s->ready = 1;
      munmap(s, sizeof(UidShm));                  // 段留给 sibling 读,teardown 时 unlink
    }
    for (int p = 1; p < c->cfg.nranks; p++) {
      if (c->topo.same_host(0, p)) continue;      // 同机走 shm
      nvshmemx_uniqueid_t tmp;
      if (net_blocking_exchange(c, p, &uid, &tmp, sizeof(uid)) != 0) return -1;
    }
  } else if (c->topo.same_host(c->cfg.rank, 0)) {
    // rank0 的同机 sibling:轮询 /dev/shm(30s,与 dconn 交换同档)
    bool got = false;
    uint64_t dl = now_ns() + 30ull * 1000000000ull;
    while (!got) {
      int fd = shm_open(shmname, O_RDWR, 0600);
      if (fd >= 0) {
        struct stat st{};
        if (fstat(fd, &st) == 0 && (size_t)st.st_size >= sizeof(UidShm)) {
          UidShm* s = (UidShm*)mmap(nullptr, sizeof(UidShm),
                                    PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
          if (s != MAP_FAILED) {
            if (s->ready) {
              __sync_synchronize();
              memcpy(&uid, (const void*)&s->uid, sizeof uid);
              got = true;
            }
            munmap(s, sizeof(UidShm));
          }
        }
        close(fd);
      }
      if (!got) {
        if (now_ns() > dl) { MLOG_ERR("E5015 等 uid shm(%s)超时", shmname); return -1; }
        usleep(200);
      }
    }
  } else {
    nvshmemx_uniqueid_t dummy{};
    if (net_blocking_exchange(c, 0, &dummy, &uid, sizeof(uid)) != 0) return -1;
  }
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(c->cfg.rank, c->cfg.nranks, &uid, &attr);
  // hostlib 入口:纯 host 应用无 device 内核 → 不必链 libnvshmem_device.a
  // (那个 .a 需要 nvcc -dlink,本机没有 nvcc;NVSHMEM 3.x 官方为此提供
  //  nvshmemx_hostlib_init_attr,见 include/host/nvshmemx_api.h:45)
  if (nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
    MLOG_ERR("E5011 nvshmemx_hostlib_init_attr 失败");
    return -1;
  }
  v->npes = nvshmem_n_pes();
  v->mype = nvshmem_my_pe();
  if (v->npes != c->cfg.nranks || v->mype != c->cfg.rank) {
    MLOG_ERR("E5012 PE 映射不符 mype=%d npes=%d", v->mype, v->npes);
    return -1;
  }

  const int n = c->cfg.nranks;
  v->nslots = nvsh_slots();
  v->slot_bytes = (size_t)HDR_STRIDE + c->cfg.slice_bytes;
  v->ring = (uint8_t*)nvshmem_malloc((size_t)n * v->nslots * v->slot_bytes);
  v->sig = (uint64_t*)nvshmem_calloc((size_t)n * v->nslots, sizeof(uint64_t));
  v->credit = (uint64_t*)nvshmem_calloc(n, sizeof(uint64_t));
  if (!v->ring || !v->sig || !v->credit) { MLOG_ERR("E5013 nvshmem_malloc 失败"); return -1; }

  // staging 槽数。默认 2×nslots 够同步档用(那里只有注册失败的缓冲才走 staging);
  // 异步档一律走 staging(坑 #45),在途窗口可达 nslots×peers ⇒ 槽太少会形成
  // 跨节点循环等待(坑 #40)。用 MOSAIC_TX_SLOTS 调,watchdog 的 txfree=x/y 看余量。
  {
    const char* e = getenv("MOSAIC_TX_SLOTS");
    int want = e ? atoi(e) : 0;
    v->tx_nslots = want > 0 ? want : (s3_zerostream_mode() ? 8 * v->nslots : 2 * v->nslots);
  }
  // 发送 staging 也从对称堆分配:IBRC 传输的 put 源必须是 NVSHMEM 注册过的
  // 内存,普通 cudaMalloc 会报 IB local protection error(实测,log.md 坑 #21)
  v->tx_stage = (uint8_t*)nvshmem_malloc((size_t)v->tx_nslots * v->slot_bytes);
  for (int i = v->tx_nslots - 1; i >= 0; i--) v->tx_free.push_back(i);

  // O5 零拷贝(默认关;MOSAIC_NVSH_ZCOPY=1 开)
  {
    const char* e = getenv("MOSAIC_NVSH_ZCOPY");
    v->zcopy = e && atoi(e) != 0;
    if (v->zcopy) {
      v->hdr_stage = (uint8_t*)nvshmem_malloc((size_t)v->tx_nslots * HDR_STRIDE);
      if (!v->hdr_stage) { v->zcopy = false; MLOG_WARN("E5017 zcopy 头槽分配失败,回退 staging"); }
      else {
        for (int i = v->tx_nslots - 1; i >= 0; i--) v->hdr_free.push_back(i);
        // relay 池也注册进 NVSHMEM:中转 hop 的 put 源就是 relay 槽
        if (c->relay_pool && c->relay_slots > 0) {
          size_t rb = (size_t)c->relay_slots * (size_t)c->cfg.slice_bytes;
          if (nvshmemx_buffer_register(c->relay_pool, rb) == 0)
            v->regs.push_back({c->relay_pool, rb});
          else
            MLOG_WARN("E5018 relay 池注册失败,中转 hop 走 staging(zcopy 仅首跳)");
        }
        MLOG_INFO("NVSHMEM zcopy 开:payload 直接以 sendbuf/relay 为 put 源");
      }
    }
  }
  v->sent_cnt.assign(n, 0);
  v->cons_seen.assign(n, 0);
  v->staged_pend.assign(n, 0);
  v->rx_consumed.assign(n, 0);
  v->rx_next.assign(n, 0);
  v->rx_done.assign((size_t)n * v->nslots, 0);
  v->rx_cursor.assign(n, 0);
  // Mapped:S3 二期镜像核要设备直写 shadow(UVA);对 memcpy 镜像路径无影响
  cudaHostAlloc(&v->sig_shadow, (size_t)n * v->nslots * sizeof(uint64_t), cudaHostAllocMapped);
  cudaHostAlloc(&v->hdr_shadow, (size_t)n * v->nslots * HDR_STRIDE, cudaHostAllocMapped);
  cudaHostAlloc(&v->credit_shadow, n * sizeof(uint64_t), cudaHostAllocMapped);
  cudaHostAlloc(&v->sig_raw, (size_t)n * v->nslots * sizeof(uint64_t), cudaHostAllocMapped);
  memset(v->sig_raw, 0, (size_t)n * v->nslots * sizeof(uint64_t));
  memset(v->sig_shadow, 0, (size_t)n * v->nslots * sizeof(uint64_t));
  memset(v->credit_shadow, 0, n * sizeof(uint64_t));
  v->sig_seen.assign((size_t)n * v->nslots, 0);
  // 零流档:host-API 发送的头源池(RDMA 源必须注册 —— 坑 #21;pinned + 注册)
  if (c->zs_cap) {
    const int NHH = 256;
    MLOG_INFO("S3 头池注册开始(%d 槽)", NHH);
    if (cudaHostAlloc(&v->hhdr_pool, (size_t)NHH * HDR_STRIDE,
                      cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
        nvshmemx_buffer_register(v->hhdr_pool, (size_t)NHH * HDR_STRIDE) == 0) {
      for (int i = NHH - 1; i >= 0; i--) v->hhdr_free.push_back(i);
      void* hd = nullptr;
      if (cudaHostGetDevicePointer(&hd, v->hhdr_pool, 0) == cudaSuccess)
        v->hhdr_dev = (uint64_t)(uintptr_t)hd;
      MLOG_INFO("S3 头池注册完成");
    } else {
      MLOG_WARN("E5020 host-API 头池注册失败,S3 异步 nvsh 发送不可用(将背压)");
      if (v->hhdr_pool) { cudaFreeHost(v->hhdr_pool); v->hhdr_pool = nullptr; }
    }
    // relay 池注册(zcopy 关时补上 —— host-API put 直接以 relay 为源)
    if (!nvsh_src_registered(v, c->relay_pool, 1) && c->relay_pool && c->relay_slots > 0) {
      size_t rb = (size_t)c->relay_slots * (size_t)c->cfg.slice_bytes;
      MLOG_INFO("S3 relay 池注册开始(%zu MB)", rb >> 20);
      if (nvshmemx_buffer_register(c->relay_pool, rb) == 0)
        v->regs.push_back({c->relay_pool, rb});
      else
        MLOG_WARN("E5021 relay 池注册失败,S3 异步中转 hop 将背压等待");
      MLOG_INFO("S3 relay 池注册返回");
    }
  }
  cudaStreamCreateWithFlags(&v->s_tx, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&v->s_poll, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&v->ev_poll, cudaEventDisableTiming);
  nvshmemx_barrier_all_on_stream(v->s_tx);       // 起始栅栏(仅起止,执行中无)
  cudaStreamSynchronize(v->s_tx);
  MLOG_INFO("NVSHMEM 就绪:%d PE,%d 槽/peer(%zu KB/槽)", v->npes, v->nslots,
            v->slot_bytes >> 10);
  return 0;
}

void nvsh_teardown(Comm* c) {
  NvshCtx* v = NC(c);
  if (!v) return;
  if (c->cfg.rank == 0) {          // uid 段清理(不存在则无害)
    char nm[96];
    uid_shm_name(nm, sizeof nm);
    shm_unlink(nm);
  }
  for (auto& r : v->regs) nvshmemx_buffer_unregister(const_cast<void*>(r.first));
  for (auto e : v->ev_free) cudaEventDestroy(e);
  if (v->hdr_stage) nvshmem_free(v->hdr_stage);
  if (v->ring) nvshmem_free(v->ring);
  if (v->sig) nvshmem_free(v->sig);
  if (v->credit) nvshmem_free(v->credit);
  if (v->tx_stage) nvshmem_free(v->tx_stage);
  if (v->sig_shadow) cudaFreeHost(v->sig_shadow);
  if (v->hdr_shadow) cudaFreeHost(v->hdr_shadow);
  if (v->credit_shadow) cudaFreeHost(v->credit_shadow);
  if (v->s_tx) cudaStreamDestroy(v->s_tx);
  if (v->s_poll) cudaStreamDestroy(v->s_poll);
  if (v->ev_poll) cudaEventDestroy(v->ev_poll);
  nvshmemx_hostlib_finalize();
  delete v;
  c->nvsh = nullptr;
}

// O5:ptr..ptr+len 是否落在已注册范围内(sendbuf 经 note 注册 / relay 池 / 对称堆)
static bool nvsh_src_registered(NvshCtx* v, const void* p, size_t len) {
  const uint8_t* a = (const uint8_t*)p;
  for (auto& r : v->regs) {
    const uint8_t* b = (const uint8_t*)r.first;
    if (a >= b && a + len <= b + r.second) return true;
  }
  return false;
}

// O6:event 池化
static cudaEvent_t nvsh_ev_take(NvshCtx* v) {
  if (!v->ev_free.empty()) { cudaEvent_t e = v->ev_free.back(); v->ev_free.pop_back(); return e; }
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  return e;
}

// O1:发送流等"输入就绪"事件(coll_begin 调一次;之后本 collective 所有
// 组装/put 都排在调用方流的生产 kernel 之后 —— device 侧依赖,零 host 等待)
void nvsh_wait_input(Comm* c, void* ev) {
  NvshCtx* v = NC(c);
  if (v && ev) cudaStreamWaitEvent(v->s_tx, (cudaEvent_t)ev, 0);
}

// O5:登记/注册本次 collective 的 sendbuf 区间(命中缓存则零成本)
void nvsh_note_sendbuf(Comm* c, const void* base, size_t bytes) {
  NvshCtx* v = NC(c);
  // zs_cap:零流 host-API 发送也要 sendbuf 注册(zcopy 同一套缓存/回退)
  if (!v || (!v->zcopy && !c->zs_cap) || !base || !bytes) return;
  // 坑 #45:**异步档默认不注册 torch 缓冲**。注册是按地址区间缓存的,而 torch 的
  // caching allocator 会回收 / 重映射这些段 —— 缓存说"已注册"、MR 其实指向已失效
  // 的页 ⇒ `ibv_poll_cq completion status: 4 (local protection error)`,QP 进错误
  // 态、后续 WR 全 flushed、NVSHMEM 在 quiet 里报错并把进程带走(29355321-185921
  // 实测:8 个 rank 无声消失,其余全体等它们 = 看起来像调度死锁)。
  // 零流档把 payload 先经消费者核拷进对称堆(天生永久注册)再发,多一次 D2D,
  // 换掉一整类"生命期跨出调用"的内存风险。
  // **坑 #50**:零流档下"源已注册"会让 host-API 直接以 sendbuf 为 put 源
  // (`putmem_nbi(rdst, gpu_src, …)`,非阻塞)。而 posted_send 在入队即记账、
  // 完成判据里又没有"nvsh 在途已 quiet",于是 collective 可能在**网卡还在
  // DMA 读 sendbuf** 时就宣布完成 ⇒ torch 立刻复用那块内存 ⇒ 发出去的是脏
  // 数据。现象:2-PE bit-exact 与单发 drop-in 都过,**只有连续多迭代的训练
  // loss 慢慢漂**(本机 9.086837 → 9.103096)。
  // 解法(坑 #45 同源):零流档一律不注册,全部经对称堆 staging 转发
  // —— 源变成对称堆槽,sendbuf 早在消费者核拷贝时就读完了,复用安全。
  if (s3_zerostream_mode()) return;
  if (nvsh_src_registered(v, base, bytes)) return;
  for (auto& r : v->reg_fail)
    if (r.first == base && r.second == bytes) return;      // 已知注册不动这块
  if (nvshmemx_buffer_register(const_cast<void*>(base), bytes) == 0) {
    v->regs.push_back({base, bytes});
    MLOG_DBG("zcopy:注册 sendbuf %p +%zu(缓存 %zu 条)", base, bytes, v->regs.size());
  } else {
    v->reg_fail.push_back({base, bytes});
    if (v->reg_fail.size() <= 3)
      MLOG_WARN("E5019 sendbuf 注册失败(%p +%zu),该缓冲走 staging 路径", base, bytes);
    if (v->reg_fail.size() > 64) v->reg_fail.erase(v->reg_fail.begin());
  }
}

// 发送一个 slice:gpu_src 是设备内存(sendbuf 偏移或 relay 池)。
// 返回 0 = 已提交;1 = credit/staging 不足(背压,调用方稍后重试);-1 = 错误。
int nvsh_post_slice(Comm* c, int peer, const SliceHdr& h, const void* gpu_src,
                    int relay_slot) {
  NvshCtx* v = NC(c);
  // credit:对端消费计数经对端 put 到我的 credit[peer] 槽(shadow 轮询更新)。
  // staged_pend = 已占 staging 槽但 seq 还没领的片,必须一并算进窗口(坑 #41)
  if (v->sent_cnt[peer] + v->staged_pend[peer] - v->cons_seen[peer]
      >= (uint64_t)v->nslots) return 1;

  // 远端槽位 = 我(src PE)在对端的环行 × 环内序号(两条路径共用)
  const bool zc = v->zcopy && nvsh_src_registered(v, gpu_src, h.len);
  if (zc && v->hdr_free.empty()) return 1;                 // 头槽背压
  if (!zc && v->tx_free.empty()) return 1;                 // staging 背压

  // ── S3 二期(异步窗口):host-API put,零流操作。门等待未决期间流上操作
  // 会被驱动楔死(Raven 实测),host 直发 proxy RDMA 绕开;源必须已注册
  // (sendbuf 提交时注册 / relay 池 setup 注册),头走注册的 pinned 头池。
  // 同 PE 单 QP 按发出序交付 ⇒ payload put 先发、hdr+signal 后发即可
  // (与 zcopy 流路径同一条顺序论证;多 QP 形态同样勿开,D11)。
  // 源必须是 NVSHMEM 注册内存才能走 host-API put。**注册失败是永久性的**
  // (torch 分配器给的某些 sendbuf 就是注册不上,E5019;29353960-151611 实测
  // 32 rank 里恰好 2 个 rank 注册失败,而缺数据的就是这 2 个)⇒ 这里绝不能
  // 背压,必须落回下面的流路径(与 O5 zcopy 失败时同一条回退,坑 #37)。
  // 头池空则是**临时**资源不足,背压重试即可。
  // S3 二期:host-API 发送**一律**走 kdesc staging 合并路径(engine 侧处理)。
  // 为什么不直接以 sendbuf 为源分三笔发(payload/hdr/sig):三笔之间的到达
  // 顺序不可依赖 —— 现场实测 sig 到了第 2 代而 hdr 还是第 1 代,镜像核的
  // tag 校验永远等不到匹配的头(坑 #39)。合并后 hdr 与 payload 同生共死,
  // 只剩 sig 一笔可能越序,而那种情况 tag 校验下一轮自愈。
  // 源已注册 ⇒ 直接三笔发(payload → fence → hdr → fence → sig),不占
  // staging 槽:staging 只有 2×nslots=64 个,而在途窗口可达 nslots×peers,
  // 全走 staging 会把槽耗尽并形成跨节点循环等待(164755 实测)。
  // 顺序由 fence 保证(坑 #39)。源没注册 ⇒ 返回 2 走 kdesc staging 合并路径。
  if (c->kdesc_active && v->hhdr_pool && v->hhdr_dev) {
    if (!nvsh_src_registered(v, gpu_src, h.len)) return 2;
    if (v->hhdr_free.empty()) return 1;                          // 头池背压(临时)
    uint64_t seq = v->sent_cnt[peer]++;
    int rslot = (int)(seq % v->nslots);
    uint8_t* rdst = v->ring + ((size_t)v->mype * v->nslots + rslot) * v->slot_bytes;
    uint64_t* rsig = v->sig + (size_t)v->mype * v->nslots + rslot;
    int hs = v->hhdr_free.back(); v->hhdr_free.pop_back();
    uint8_t* hslot = v->hhdr_pool + (size_t)hs * HDR_STRIDE;
    memcpy(hslot, &h, sizeof(SliceHdr));
    if (!c->cfg.verify) {          // 槽使用代数写进 crc 位(镜像核 tag 校验)
      uint32_t tag = (uint32_t)(seq / (uint64_t)v->nslots + 1);
      memcpy(hslot + 30, &tag, 4);
    }
    // host 侧无 putmem_signal_nbi(本版本仅 device/on_stream)⇒ 拆三笔同 QP
    // 顺序发:payload → 头 → 信号。信号值 = 该槽第几次使用(seq/nslots+1),
    // 与流路径 SIGNAL_ADD(+1/次,自 0 起)的绝对值恒等 —— 跨 coll 混用一致。
    // 信号**必须也走 putmem_nbi**(源 = 头槽尾部 8B):uint64_p 走别的通道,
    // sig 会先于 hdr/payload 到 —— 镜像核按 sig 变化只拷一次头,拷到旧头
    // magic 永错,片永久卡死(113830 实测,txpend 不回收 + recv 差几片)。
    uint64_t sv = seq / (uint64_t)v->nslots + 1;
    memcpy(hslot + 40, &sv, sizeof(sv));       // 34B 头之后,8B 对齐
    // fence 保证对同一 PE 的先后可见序:payload → 头 → 信号(坑 #39)
    nvshmem_putmem_nbi(rdst + HDR_STRIDE, gpu_src, h.len, peer);
    nvshmem_fence();
    nvshmem_putmem_nbi(rdst, hslot, sizeof(SliceHdr), peer);
    nvshmem_fence();
    nvshmem_putmem_nbi(rsig, hslot + 40, sizeof(uint64_t), peer);
    v->host_posts += 3;
    NvshCtx::TxRec r{-1, nullptr, h.len, relay_slot, peer, seq, true};
    r.hhdr_slot = hs;
    r.relay_credit = (relay_slot >= 0);   // 源被 put 直读 ⇒ credit 驱动释放(坑 #21)
    v->tx_pend.push_back(r);
    return 0;
  }

  uint64_t seq = v->sent_cnt[peer]++;
  int rslot = (int)(seq % v->nslots);
  uint8_t* rdst = v->ring + ((size_t)v->mype * v->nslots + rslot) * v->slot_bytes;
  uint64_t* rsig = v->sig + (size_t)v->mype * v->nslots + rslot;

  NvshCtx::TxRec r{-1, nullptr, h.len, relay_slot, peer, seq, false};
  // 流路径也写代数 tag(crc 位,verify 关时):镜像核常驻后阻塞 coll 的
  // 交付同样经它校验,没 tag 会被判"头未就绪"永卡
  SliceHdr h2 = h;
  if (!c->cfg.verify) h2.crc = (uint32_t)(seq / (uint64_t)v->nslots + 1);
  if (zc) {
    // O5 零拷贝:payload 直接以 gpu_src 为源 put 到远端槽 payload 区;
    // 64B 头单独经头槽 put 到槽首并携带 signal。顺序保证:同流 + 同 PE +
    // IBRC 单 RC QP 顺序交付 ⇒ signal 到 ⇒ 头、payload 均已落
    // (多 QP 形态勿开 zcopy:NVSHMEM_IB_NUM_RC_QPS>1 时无此保证,File/log_raven.md §6 D15)。
    int hs = v->hdr_free.back(); v->hdr_free.pop_back();
    uint8_t* hslot = v->hdr_stage + (size_t)hs * HDR_STRIDE;
    cudaMemcpyAsync(hslot, &h2, sizeof(SliceHdr), cudaMemcpyHostToDevice, v->s_tx);
    nvshmemx_putmem_nbi_on_stream(rdst + HDR_STRIDE, gpu_src, h.len, peer, v->s_tx);
    nvshmemx_putmem_signal_nbi_on_stream(rdst, hslot, sizeof(SliceHdr),
                                         rsig, 1, NVSHMEM_SIGNAL_ADD, peer, v->s_tx);
    r.hdr_slot = hs;
    // relay 源被 put 直接读 ⇒ 释放同样必须 credit 驱动(坑 #21 的同一条纪律)
    r.relay_credit = (relay_slot >= 0);
  } else {
    int ts = v->tx_free.back(); v->tx_free.pop_back();
    uint8_t* slot = v->tx_stage + (size_t)ts * v->slot_bytes;
    // 组装 [头|payload](都在 s_tx 流上,与 put 同流 ⇒ 顺序天然成立)
    cudaMemcpyAsync(slot, &h2, sizeof(SliceHdr), cudaMemcpyHostToDevice, v->s_tx);
    cudaMemcpyAsync(slot + HDR_STRIDE, gpu_src, h.len, cudaMemcpyDeviceToDevice, v->s_tx);
    nvshmemx_putmem_signal_nbi_on_stream(rdst, slot, HDR_STRIDE + h.len,
                                         rsig, 1, NVSHMEM_SIGNAL_ADD, peer, v->s_tx);
    r.slot = ts;
  }
  r.ev = nvsh_ev_take(v);
  cudaEventRecord(r.ev, v->s_tx);
  v->tx_pend.push_back(r);
  return 0;
}

// 消费拷贝完成后(pump_rx 的 event 判定)把 credit 送回发送端 —— 早回会让
// 发送端在我们的 D2D 还没离开槽位时就覆盖它(与 verbs_repost 同一条纪律)
// watchdog 归因:s_tx / s_poll 活性快照(txpend 增长=s_tx 死;rounds 冻=s_poll 死)
void nvsh_dbg(Comm* c, char* buf, int n) {
  NvshCtx* v = NC(c);
  if (!v) { if (n) buf[0] = 0; return; }
  // 坑 #36 定案后恢复窗口内查询:唯一危险的是 cudaFree(设备级隐式同步),
  // 查询类 API 在核在场时实测安全(test_persist_sync),且核已改有界生命期
  int pq = v->poll_inflight ? (int)cudaEventQuery(v->ev_poll) : 0;
  int tq = (int)cudaStreamQuery(v->s_tx);
  // 池水位:哪种资源耗尽一眼可见(txfree=staging 槽 / hhfree=host 头池 /
  // hdfree=zcopy 头槽);win = 信用窗口占用最满的 peer(sent+staged-seen)/nslots
  int wmax = -1, wpeer = -1;
  for (int p = 0; p < c->cfg.nranks; p++) {
    if (p == c->cfg.rank) continue;
    int w = (int)(v->sent_cnt[p] + v->staged_pend[p] - v->cons_seen[p]);
    if (w > wmax) { wmax = w; wpeer = p; }
  }
  int o = snprintf(buf, n, "txpend=%zu rounds=%llu consume=%llu ph=%llu mir=%d inflight=%d evpoll=%d stx=%d "
                   "| pool txfree=%zu/%d hhfree=%zu hdfree=%zu win=%d/%d@r%d",
                   v->tx_pend.size(), (unsigned long long)v->poll_rounds,
                   (unsigned long long)v->consume_calls, (unsigned long long)v->nvsh_phase,
                   (int)c->kmirror_on, (int)v->poll_inflight, pq, tq,
                   v->tx_free.size(), v->tx_nslots, v->hhdr_free.size(),
                   v->hdr_free.size(), wmax, v->nslots, wpeer);
  // 卡在哪:对还欠我数据的 src,打它当前等待槽的三层状态
  //   raw  = 镜像核无条件抄的设备 sig(到没到);pub = 过校验后发布的 sig
  //   seen = host 已消费到的 sig。raw>pub ⇒ 镜像核在拒;pub>seen ⇒ 消费卡;
  //   raw==seen ⇒ 片压根没到(发送侧问题)
  int shown = 0;
  for (int p = 0; p < c->cfg.nranks && o < n - 80 && shown < 4; p++) {
    if (p == c->cfg.rank) continue;
    uint64_t want = c->demand.empty() ? 0
                    : c->demand[(size_t)p * c->cfg.nranks + c->cfg.rank];
    if (!want || c->dst.recv_by_src[p] >= want) continue;
    int s2 = (int)(v->rx_next[p] % (uint64_t)v->nslots);
    size_t idx = (size_t)p * v->nslots + s2;
    uint32_t magic = 0;
    memcpy(&magic, v->hdr_shadow + idx * HDR_STRIDE, 4);
    o += snprintf(buf + o, n - o, " | src%d sl%d raw=%llu pub=%llu seen=%llu mg=%s",
                  p, s2, (unsigned long long)(v->sig_raw ? v->sig_raw[idx] : 0),
                  (unsigned long long)v->sig_shadow[idx],
                  (unsigned long long)v->sig_seen[idx],
                  magic == SLICE_MAGIC ? "ok" : "BAD");
    shown++;
  }
}

// ── S3 二期:kdesc staging(坑 #37 的根治)。torch 分配器给的某些 sendbuf
// 注册不上(E5019),而 host-API put 要求源已注册;回退流路径又会在门未决时
// 被通道碰撞卡死。出路:先用**消费者核**把 payload 拷进对称堆 tx_stage 槽
// (天生已注册),再 host-API 发出去 —— 全程零流。
int nvsh_stage_alloc(Comm* c, int peer, int* slot, void** dst_dev) {
  NvshCtx* v = NC(c);
  if (!v || v->tx_free.empty()) return -1;
  int ts = v->tx_free.back(); v->tx_free.pop_back();
  v->staged_pend[peer]++;               // 占住窗口名额,直到 post_staged 领 seq
  *slot = ts;
  *dst_dev = v->tx_stage + (size_t)ts * v->slot_bytes + HDR_STRIDE;
  return 0;
}
void nvsh_stage_free(Comm* c, int peer, int slot) {
  NvshCtx* v = NC(c);
  if (!v || slot < 0) return;
  v->tx_free.push_back(slot);
  if (v->staged_pend[peer]) v->staged_pend[peer]--;
}
// 消费者核已把 payload 拷进 tx_stage[slot] ⇒ 现在 host-API 发出(源已注册)
// 消费者核已把 [hdr|payload] 拼进 tx_stage[slot] ⇒ 一笔合并 put + 一笔 sig
int nvsh_post_staged(Comm* c, int peer, const SliceHdr& h, int slot, int relay_slot,
                     int hdr_slot) {
  NvshCtx* v = NC(c);
  if (!v) return 1;
  if (v->staged_pend[peer]) v->staged_pend[peer]--;   // 预定兑现成实发(坑 #41)
  uint64_t seq = v->sent_cnt[peer]++;
  int rslot = (int)(seq % v->nslots);
  uint8_t* rdst = v->ring + ((size_t)v->mype * v->nslots + rslot) * v->slot_bytes;
  uint64_t* rsig = v->sig + (size_t)v->mype * v->nslots + rslot;
  uint8_t* src = v->tx_stage + (size_t)slot * v->slot_bytes;
  uint8_t* hslot = v->hhdr_pool + (size_t)hdr_slot * HDR_STRIDE;
  uint64_t sv = seq / (uint64_t)v->nslots + 1;
  memcpy(hslot + 40, &sv, sizeof(sv));           // sig 源(必须是注册内存)
  // 顺序铁律(坑 #39):8B 的 sig 会超过大的 payload 笔先到达(实测 raw=1
  // pub=0 mg=BAD:信号到了、头还没到)。host-API 没有 put_signal 变体,
  // 用 OpenSHMEM 的 fence —— 它保证**对同一 PE** fence 之前的 put 先于
  // 之后的 put 可见。fence 是本地顺序原语,不等完成,开销远小于 quiet。
  nvshmem_putmem_nbi(rdst, src, HDR_STRIDE + h.len, peer);   // 头+payload 一笔
  nvshmem_fence();
  nvshmem_putmem_nbi(rsig, hslot + 40, sizeof(uint64_t), peer);
  v->host_posts += 2;
  NvshCtx::TxRec r{slot, nullptr, h.len, relay_slot, peer, seq, true};
  r.hhdr_slot = hdr_slot;
  r.relay_credit = (relay_slot >= 0);
  v->tx_pend.push_back(r);
  return 0;
}
// engine 侧要先把带 tag 的头写进 hhdr 槽,并拿到它的设备地址喂给消费者核
int nvsh_hdr_stage(Comm* c, const SliceHdr& h, int peer, int* hdr_slot, uint64_t* src_dev) {
  NvshCtx* v = NC(c);
  if (!v || !v->hhdr_pool || !v->hhdr_dev || v->hhdr_free.empty()) return -1;
  int hs = v->hhdr_free.back(); v->hhdr_free.pop_back();
  uint8_t* hslot = v->hhdr_pool + (size_t)hs * HDR_STRIDE;
  memset(hslot, 0, HDR_STRIDE);
  memcpy(hslot, &h, sizeof(SliceHdr));
  if (!c->cfg.verify) {   // 槽使用代数写进 crc 位(镜像核 tag 校验;与 sig 值同)
    uint32_t tag = (uint32_t)(v->sent_cnt[peer] / (uint64_t)v->nslots + 1);
    memcpy(hslot + 30, &tag, 4);
  }
  *hdr_slot = hs;
  *src_dev = v->hhdr_dev + (uint64_t)hs * HDR_STRIDE;
  return 0;
}
void nvsh_hdr_free(Comm* c, int hdr_slot) {
  NvshCtx* v = NC(c);
  if (v && hdr_slot >= 0) v->hhdr_free.push_back(hdr_slot);
}

// S3 二期:flush host-API 的 nbi put。**必需**:putmem_nbi 只保证"发起",
// 小报文(hdr/sig)会滞留在本地队列 —— 现场表现为 sig 到了而 hdr 没到
// (raw=1 pub=0 mg=BAD),接收端按序消费永久卡死(163001 实测)。
// 每轮 pump 之后 quiet 一次,把本轮批次推出去。
void nvsh_flush(Comm* c) {
  NvshCtx* v = NC(c);
  if (!v || !v->host_posts) return;
  v->host_posts = 0;
  nvshmem_quiet();
}

// S3 二期:填镜像核配置(shadow 的设备侧地址 + 对称堆指针;last_sig/stop 由
// engine 填)。0 = 成功。
int nvsh_mirror_cfg(Comm* c, KMirrorCfg* out) {
  NvshCtx* v = NC(c);
  if (!v || !v->sig_shadow) return -1;
  void *sd = nullptr, *hd = nullptr, *cd = nullptr;
  if (cudaHostGetDevicePointer(&sd, v->sig_shadow, 0) != cudaSuccess ||
      cudaHostGetDevicePointer(&hd, v->hdr_shadow, 0) != cudaSuccess ||
      cudaHostGetDevicePointer(&cd, v->credit_shadow, 0) != cudaSuccess)
    return -1;
  out->sig = (uint64_t)(uintptr_t)v->sig;
  out->credit = (uint64_t)(uintptr_t)v->credit;
  out->ring = (uint64_t)(uintptr_t)v->ring;
  void* rd = nullptr;
  out->sig_raw = (v->sig_raw && cudaHostGetDevicePointer(&rd, v->sig_raw, 0) == cudaSuccess)
                     ? (uint64_t)(uintptr_t)rd : 0;
  out->sig_shadow = (uint64_t)(uintptr_t)sd;
  out->hdr_shadow = (uint64_t)(uintptr_t)hd;
  out->credit_shadow = (uint64_t)(uintptr_t)cd;
  out->n = (uint32_t)c->cfg.nranks;
  out->nslots = (uint32_t)v->nslots;
  out->slot_bytes = (uint32_t)v->slot_bytes;
  out->hdr_stride = HDR_STRIDE;
  out->hdr_bytes = (uint32_t)sizeof(SliceHdr);
  return 0;
}

void nvsh_credit(Comm* c, int token) {
  if (token < 0) return;
  NvshCtx* v = NC(c);
  const int peer = token >> 20, s = token & 0xFFFFF;
  // R15b:标记该槽已消费,游标推进到第一个未消费槽为止 —— 报给发送端的
  // 永远是连续前缀(交付按序 ⇒ 槽↔在途 seq 一一对应,推进无歧义)。
  v->rx_done[(size_t)peer * v->nslots + s] = 1;
  uint64_t& cur = v->rx_cursor[peer];
  while (v->rx_done[(size_t)peer * v->nslots + (size_t)(cur % (uint64_t)v->nslots)]) {
    v->rx_done[(size_t)peer * v->nslots + (size_t)(cur % (uint64_t)v->nslots)] = 0;
    cur++;
  }
  if (cur == v->rx_consumed[peer]) return;        // 前缀没动(乱序中段),不发
  v->rx_consumed[peer] = cur;
  // 值语义的 p(host-issued,立即):put 源若用 host 内存(std::vector),
  // IBRC 会报 local protection error —— 未注册内存不能做 RDMA 源(坑 #21)
  nvshmem_uint64_p(v->credit + v->mype, cur, peer);
}

int nvsh_poll(Comm* c) {
  NvshCtx* v = NC(c);
  if (!v) return 0;
  int done = 0;
  const int n = c->cfg.nranks;
  v->nvsh_phase = 1;

  // 发送侧回收:event 完成 → relay 槽(仅 staging 路径)+ 检测器计时;
  // credit 覆盖 seq → tx/hdr 槽 + zcopy 路径的 relay 槽(源被 put 直接读,
  // 必须等对端消费确认 —— 坑 #21 的纪律,zcopy 把它扩展到 relay 源)
  for (size_t i = 0; i < v->tx_pend.size();) {
    auto& r = v->tx_pend[i];
    if (!r.ev_done && cudaEventQuery(r.ev) == cudaSuccess) {
      r.ev_done = true;
      v->ev_free.push_back(r.ev);          // O6:回池而不是 destroy
      r.ev = nullptr;
      if (r.relay_slot >= 0 && !r.relay_credit) {
        c->relay_free.push_back(r.relay_slot); r.relay_slot = -1;
      }
      if (c->nic_tx_outstanding > 0) c->nic_tx_outstanding--;
      det_feed(c, 0, r.len);
      done++;
    }
    if (r.ev_done && v->cons_seen[r.peer] > r.seq) {
      if (r.slot >= 0) v->tx_free.push_back(r.slot);
      if (r.hdr_slot >= 0) v->hdr_free.push_back(r.hdr_slot);
      if (r.hhdr_slot >= 0) v->hhdr_free.push_back(r.hhdr_slot);      // host-API 头池
      if (r.relay_slot >= 0) c->relay_free.push_back(r.relay_slot);   // zcopy relay
      v->tx_pend.erase(v->tx_pend.begin() + i);
      continue;
    }
    i++;
  }

  // 周期镜像 signal/hdr/credit 到 host shadow。S3 异步窗口(kmirror_on):
  // 镜像核在设备侧持续直写 shadow,host 零流操作直接消费;否则走原两相
  // memcpy 镜像。两条路径的消费逻辑同一份(下方 consume 块)。
  v->nvsh_phase = 2;
  bool consume = false;
  if (c->kmirror_on) {
    consume = true;
  } else if (!v->poll_inflight) {
    cudaMemcpyAsync(v->sig_shadow, v->sig, (size_t)n * v->nslots * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, v->s_poll);
    // 头镜像 = 规则跨步(src 每 slot_bytes 取头 sizeof(SliceHdr),dst 每
    // HDR_STRIDE)⇒ 单个 2D 拷贝。旧版逐槽 n×nslots 个 cudaMemcpyAsync,
    // 32 rank×32 槽 = 1024 个 launch/轮,event 线程单次 pass 被拖到 ~2.7ms,
    // torch 提交的 a2a 要等它一整 pass 才被察觉(MoE 察觉延迟 ~42ms/iter 主因,
    // 29349040-091053 evmax 实测归因)。拷贝字节与旧版逐一相同。
    cudaMemcpy2DAsync(v->hdr_shadow, HDR_STRIDE, v->ring, v->slot_bytes,
                      sizeof(SliceHdr), (size_t)n * v->nslots,
                      cudaMemcpyDeviceToHost, v->s_poll);
    cudaMemcpyAsync(v->credit_shadow, v->credit, n * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, v->s_poll);
    cudaEventRecord(v->ev_poll, v->s_poll);
    v->poll_inflight = true;
  } else if (cudaEventQuery(v->ev_poll) == cudaSuccess) {
    v->poll_inflight = false;
    v->poll_rounds++;
    consume = true;
  }
  if (consume) {
    v->nvsh_phase = 3;
    v->consume_calls++;
    for (int p = 0; p < n; p++) {
      if (p == c->cfg.rank) continue;
      if (v->credit_shadow[p] > v->cons_seen[p]) v->cons_seen[p] = v->credit_shadow[p];
      // R15(Raven 16 GPU 实测抓到的单片丢失):**必须按 seq 序消费,不许跳槽**。
      // 旧版按槽号扫描,某槽头部镜像暂时没跟上(magic 未就绪)会跳过它处理
      // 后面的槽 ⇒ 消费乱序 ⇒ 聚合 credit 被推高 ⇒ 发送端以为最老槽已消费,
      // seq%nslots 复用时**覆盖未消费的片**,静默丢失(全网 trace 里该片只有
      // 发端 post、无任何 deliv)。单 QP/同流交付本就有序,按序消费无额外等待;
      // 头没就绪就整个 peer 停住等下一轮镜像。
      for (;;) {
        int s = (int)(v->rx_next[p] % (uint64_t)v->nslots);
        size_t idx = (size_t)p * v->nslots + s;
        if (v->sig_shadow[idx] <= v->sig_seen[idx]) break;   // 下一片还没到
        SliceHdr h;
        memcpy(&h, v->hdr_shadow + idx * HDR_STRIDE, sizeof(h));
        if (h.magic != SLICE_MAGIC) break;       // 头还没跟上镜像,下一轮再看
        // ── 坑 #48:**代际闸门开在收方,不开在发方**。
        // 终跳的落点是 `recvbuf + rdispls[src] + off`,那是**收方当前**这一代的
        // 缓冲;若来片属于更后面的一代,现在收就是往错缓冲写。
        // 关键是**不用任何 park 缓冲**:本循环本就按 seq 严格顺序消费(R15),
        // 所以"不消费"= 不推进 rx_next、不回 credit,片就原地躺在对称堆槽里,
        // 等本 rank 进到那一代,同一个循环自然把它接上。发送方全程不必阻塞
        // —— 这才是异步档要的东西(旧的 peer 闸门 #46 是拿"重新串行化 rank"
        // 换正确性,等于把异步档的意义抵消掉)。
        // 背压天然有界:发方最多灌满 nslots 个槽就卡在 credit 上。
        // 不会死锁:发方要进下一代必须先把本代发完(send_done 要求队列全空),
        // 每对 (src,dst) 又是 FIFO ⇒ 早到片永远排在本代片之后。
        // 中转片(非终跳)落的是 relay 池、不碰 recvbuf ⇒ 与代际无关,照常放行。
        const bool fin_hop = (h.hop + 1 >= h.nhops);
        const int16_t gd = (int16_t)(h.coll - (uint16_t)(c->coll_id & 0xFFFF));
        if (fin_hop && gd > 0) break;            // 下一代的早到片:原地等
        if (fin_hop && gd < 0)                   // 上一代的迟到片:不该发生
          MLOG_ERR("E5024 迟到片 (r%d→r%d) coll=%u 本 rank 已在 coll=%d",
                   h.src, h.dst, (unsigned)h.coll, c->coll_id);
        v->sig_seen[idx] = v->sig_shadow[idx];
        void* payload = v->ring + idx * v->slot_bytes + HDR_STRIDE;   // 设备内存
        c->det_dn_hint = true;                   // 简化:批量镜像窗内视为忙
        // token 走 verbs_token 通道,pump_rx 在消费拷贝完成后调 nvsh_credit
        on_slice_arrived(c, h, payload, -1, -1, (p << 20) | s);
        v->rx_next[p]++;
        done++;
      }
    }
  }
  return done;
}

}  // namespace mosaic

#else  // !MOSAIC_HAVE_NVSHMEM —— 占位:未编 NVSHMEM 时给出清晰报错
namespace mosaic {
int nvsh_setup(Comm* c) {
  MLOG_ERR("E5014 本构建未启用 NVSHMEM(make NVSHMEM=1)");
  return -1;
}
void nvsh_teardown(Comm*) {}
int nvsh_post_slice(Comm*, int, const SliceHdr&, const void*, int) { return -1; }
int nvsh_poll(Comm*) { return 0; }
void nvsh_credit(Comm*, int) {}
void nvsh_dbg(Comm*, char* buf, int n) { if (n) buf[0] = 0; }
int nvsh_mirror_cfg(Comm*, KMirrorCfg*) { return -1; }
int nvsh_stage_alloc(Comm*, int, int*, void**) { return -1; }
void nvsh_stage_free(Comm*, int, int) {}
int nvsh_post_staged(Comm*, int, const SliceHdr&, int, int, int) { return -1; }
int nvsh_hdr_stage(Comm*, const SliceHdr&, int, int*, uint64_t*) { return -1; }
void nvsh_hdr_free(Comm*, int) {}
void nvsh_flush(Comm*) {}
void nvsh_wait_input(Comm*, void*) {}
void nvsh_note_sendbuf(Comm*, const void*, size_t) {}
}  // namespace mosaic
#endif
