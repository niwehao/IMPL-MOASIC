/* verbs.cc —— RDMA 数据面 + 控制面(本机 RoCE / 任何有 verbs 的平台)。
 *
 * 2026-08-19 重构(File/debug-819/design-verbs-gpu-staging.md):
 *   数据面 payload 全程不落 host —— GPU staging 经 dmabuf GDR 注册,
 *   SEND/RECV 都用双 SGE:[64B SliceHdr(host pinned,CPU 直读写)|payload(GPU)]。
 *   实测本机 NIC↔GPU DMA 全向 ~91.4 Gbps(= 线速,Result/local-0819-gdr-bw)。
 *   控制面独立 QP(每 remote peer 一条):报文 ≤2KB 落 host 槽,与 slice
 *   FIFO 解耦;跨机控制面**恒走 verbs SEND**(单程 ~5µs vs TCP 内核栈 30-60µs,
 *   Gate1 实测 256KB A2A 160.4→82.7µs),建链失败自动回退 TCP(告警,确定性)。
 *
 * 形态:每个远端 peer 一条数据 RC QP + 一条控制 RC QP,SEND/RECV。
 *   选 SEND/RECV 的理由是**流控**:接收端不补投 recv WR 时,发送端撞 RNR
 *   并由硬件退避 —— 链路满了发送方自然慢下来,无需自己做信用回传。
 *
 * 与 NVSHMEM 数据面(Raven)的关系:MOSAIC_NIC=nvshmem 时本文件只建**控制面**
 * (ctrl-only:不碰 GPU MR / 数据 QP)。这是清理计划 §2 的待收敛项。
 */
#include "internal.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <cstdlib>

namespace mosaic {

// 每 peer 预投的数据 recv WR 数 = 发送端最多领先的 slice 数(在途上界)。
// 默认按字节定量:窗口至少罩住 8MB 在途(wave-turnaround 实测:按片计数的
// 窗口在小切片下把线速掐到 17.8%,开够即回 ~85%)。
static int rdma_rx_slots(int slice_bytes) {
  const char* e = getenv("MOSAIC_RDMA_RX_SLOTS");
  if (e && *e) return atoi(e);
  int n = (int)(((size_t)8 << 20) / (size_t)(slice_bytes > 0 ? slice_bytes : 1 << 20));
  return n < 16 ? 16 : n;
}

// dmabuf GDR 注册(驱动 API 经 dlopen,与 engine.cc 的 JIT 同一纪律 ——
// 免链接期 libcuda 依赖)。失败返回 nullptr(数据面 setup 硬错)。
typedef int (*cu_dmabuf_fn)(int* fd, unsigned long long ptr, size_t size,
                            int handleType, unsigned long long flags);
static ibv_mr* reg_gpu_mr(ibv_pd* pd, void* p, size_t n) {
  static cu_dmabuf_fn get_fd = [] {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    return h ? (cu_dmabuf_fn)dlsym(h, "cuMemGetHandleForAddressRange") : nullptr;
  }();
  if (!get_fd) return nullptr;
  int fd = -1;
  // CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1(cuda.h 枚举;免包含驱动头)
  if (get_fd(&fd, (unsigned long long)(uintptr_t)p, n, 1, 0) != 0 || fd < 0)
    return nullptr;
  ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, n, (uint64_t)(uintptr_t)p, fd,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  close(fd);                       // reg 持有自己的引用,fd 可关
  return mr;
}

struct VerbsPeer {
  ibv_qp* qp = nullptr;            // 数据 QP(ctrl-only 形态为 null)
  ibv_qp* qp_ctrl = nullptr;       // 控制 QP(独立:ctrl 报文不劈进 GPU、不占 slice FIFO)
  int nrx = 0;
  int posted = 0;
};

struct VerbsCtx {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  // send 与 recv 分开两个 CQ:recv 侧逐 slice 消费;send 完成(staging 回收)
  // 不被 recv 侧牵连。数据/控制 QP 共用这两个 CQ,按 wr_id 位段区分。
  ibv_cq* cq_send = nullptr;
  ibv_cq* cq_recv = nullptr;
  bool data_plane = false;         // false = ctrl-only(nvshmem 数据面形态)
  int port = 1, gid_index = 3;
  // ── 数据面缓冲(data_plane 才建)
  ibv_mr* mr_stage_hdr = nullptr;  // 发送:host 头池(engine 的 stage_hdr_pool)
  ibv_mr* mr_stage_gpu = nullptr;  // 发送:GPU payload 池(engine 的 stage_gpu_pool,GDR)
  uint8_t* rx_hdr_pool = nullptr;  // 接收:host 头槽池 [peer][idx] × HDR_STRIDE
  ibv_mr* mr_rx_hdr = nullptr;
  void* rx_gpu_pool = nullptr;     // 接收:GPU payload 槽池 [peer][idx] × slice_bytes(GDR)
  ibv_mr* mr_rx_gpu = nullptr;
  size_t rx_pay_bytes = 0;         // = slice_bytes
  // ── 控制面
  static constexpr int CTRL_SLOT = 2048;
  static constexpr int NCTRL = 64;      // 发送槽(全 peer 共享池)
  static constexpr int NCTRL_RX = 64;   // 每 peer 预投的 ctrl recv 槽
  uint8_t* ctrl_pool = nullptr;         // 发送槽池
  ibv_mr* mr_ctrl = nullptr;
  std::vector<int> ctrl_free;           // 由 c->ctrl_mu 保护(app/event 两线程都发)
  uint8_t* ctrl_rx_pool = nullptr;      // 接收槽池 [peer][idx] × CTRL_SLOT
  ibv_mr* mr_ctrl_rx = nullptr;
  std::vector<VerbsPeer> peer;
};
// wr_id 位段:bit40 = 数据 recv;bit41 = ctrl 发送(低位 = 槽号);
// bit42 = ctrl recv;数据发送 = 纯低位(staging 槽号)。
static constexpr uint64_t WRID_RX = 1ull << 40;
static constexpr uint64_t WRID_CTRL = 1ull << 41;
static constexpr uint64_t WRID_CRX = 1ull << 42;

static VerbsCtx* VC(Comm* c) { return (VerbsCtx*)c->verbs; }

#pragma pack(push, 1)
struct QpInfo {          // 经既有的 TCP 数据连接交换(建链期,阻塞收发)
  uint32_t qp_num, psn;
  uint32_t ctrl_qp_num, ctrl_psn;
  uint8_t gid[16];
  uint32_t magic;
  uint8_t data_plane;    // 双方必须一致(verbs 数据面 vs ctrl-only)
  uint8_t _pad[3];
};
#pragma pack(pop)

// 自动挑选 ACTIVE 的那块卡:两机的 verbs 设备名不同(mlx5_0 vs rocep66s0f0),
// 按名字分支很容易在换机器时悄悄挂掉,按端口状态选更稳。
static ibv_context* open_active_device(int* port_out) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) { MLOG_ERR("E1010 没有 RDMA 设备"); return nullptr; }
  const char* want = getenv("MOSAIC_HCA");
  for (int i = 0; i < n; i++) {
    const char* nm = ibv_get_device_name(list[i]);
    if (want && *want && strcmp(nm, want) != 0) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (!ctx) continue;
    ibv_port_attr pa{};
    for (int p = 1; p <= 2; p++) {
      if (ibv_query_port(ctx, p, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
        MLOG_INFO("RDMA 设备 %s port %d ACTIVE", nm, p);
        *port_out = p;
        ibv_free_device_list(list);
        return ctx;
      }
    }
    ibv_close_device(ctx);
  }
  ibv_free_device_list(list);
  MLOG_ERR("E1011 没有 ACTIVE 的 RDMA 端口");
  return nullptr;
}

static bool qp_connect(VerbsCtx* v, ibv_qp* qp, uint32_t my_psn, uint32_t rem_qpn,
                       uint32_t rem_psn, const uint8_t* rem_gid) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = (uint8_t)v->port;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
    MLOG_ERR("E1012 modify INIT 失败"); return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = IBV_MTU_1024;      // RoCE 上 1024 最保险(MTU 与以太网帧无关)
  a.dest_qp_num = rem_qpn;
  a.rq_psn = rem_psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;           // RNR 退避 ~640µs:接收端没补投 WR 时靠它背压
  a.ah_attr.is_global = 1;        // RoCE v2 必须走 GRH
  a.ah_attr.port_num = (uint8_t)v->port;
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.grh.sgid_index = (uint8_t)v->gid_index;
  memcpy(&a.ah_attr.grh.dgid, rem_gid, 16);
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
    MLOG_ERR("E1013 modify RTR 失败(GID index %d 对吗?)", v->gid_index); return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;                // 无限重试:RNR 是正常的背压,不是错误
  a.sq_psn = my_psn;
  a.max_rd_atomic = 1;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
    MLOG_ERR("E1014 modify RTS 失败"); return false;
  }
  return true;
}

// 数据 recv:双 SGE [host 头槽 64B | GPU payload 槽]
static bool post_recv(Comm* c, int peer, int idx) {
  VerbsCtx* v = VC(c);
  VerbsPeer& p = v->peer[peer];
  const int nrx = p.nrx;
  ibv_sge sge[2];
  sge[0].addr = (uint64_t)(uintptr_t)(v->rx_hdr_pool + ((size_t)peer * nrx + idx) * HDR_STRIDE);
  sge[0].length = HDR_STRIDE;
  sge[0].lkey = v->mr_rx_hdr->lkey;
  sge[1].addr = (uint64_t)(uintptr_t)((uint8_t*)v->rx_gpu_pool +
                                      ((size_t)peer * nrx + idx) * v->rx_pay_bytes);
  sge[1].length = (uint32_t)v->rx_pay_bytes;
  sge[1].lkey = v->mr_rx_gpu->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = WRID_RX | ((uint64_t)peer << 20) | (uint32_t)idx;
  wr.sg_list = sge;
  wr.num_sge = 2;
  if (ibv_post_recv(p.qp, &wr, &bad)) { MLOG_ERR("E1015 post_recv 失败"); return false; }
  p.posted++;
  return true;
}

static bool post_ctrl_recv(Comm* c, int peer, int idx) {
  VerbsCtx* v = VC(c);
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)(v->ctrl_rx_pool +
                                   ((size_t)peer * VerbsCtx::NCTRL_RX + idx) * VerbsCtx::CTRL_SLOT);
  sge.length = VerbsCtx::CTRL_SLOT;
  sge.lkey = v->mr_ctrl_rx->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = WRID_CRX | ((uint64_t)peer << 20) | (uint32_t)idx;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (ibv_post_recv(v->peer[peer].qp_ctrl, &wr, &bad)) {
    MLOG_ERR("E1015 post_ctrl_recv 失败"); return false;
  }
  return true;
}

// data_plane=true:数据面 + 控制面;false:仅控制面(nvshmem 数据面形态)。
int verbs_setup(Comm* c, bool data_plane) {
  VerbsCtx* v = new VerbsCtx();
  c->verbs = v;
  v->data_plane = data_plane;
  v->gid_index = getenv("MOSAIC_GID_INDEX") ? atoi(getenv("MOSAIC_GID_INDEX")) : 3;
  v->ctx = open_active_device(&v->port);
  if (!v->ctx) return -1;
  v->pd = ibv_alloc_pd(v->ctx);
  const int nrx = data_plane ? rdma_rx_slots(c->cfg.slice_bytes) : 0;
  const int nremote = c->cfg.nranks - c->cfg.gpus_per_host;
  // CQ 深度必须罩住全部可能同时未收割的完成:数据 recv(nrx/peer)+ ctrl recv
  // (NCTRL_RX/peer)+ 发送侧(数据在途 + ctrl 池)。少算会 CQ 溢出 → QP 错误。
  const int cqe = std::max(256, (std::max(1, nrx) + VerbsCtx::NCTRL_RX + 8) *
                                    std::max(1, nremote) + VerbsCtx::NCTRL);
  v->cq_send = ibv_create_cq(v->ctx, cqe, nullptr, nullptr, 0);
  v->cq_recv = ibv_create_cq(v->ctx, cqe, nullptr, nullptr, 0);
  if (!v->pd || !v->cq_send || !v->cq_recv) { MLOG_ERR("E1016 alloc pd/cq 失败"); return -1; }

  if (data_plane) {
    if (!c->cfg.use_gpu || !c->stage_gpu_pool) {
      MLOG_ERR("E5031 verbs 数据面需要 GPU staging(use_gpu=0 的跨机形态不支持)");
      return -1;
    }
    // 发送:注册 engine 的 host 头池 + GPU payload 池(GDR)
    v->mr_stage_hdr = ibv_reg_mr(v->pd, c->stage_hdr_pool,
                                 (size_t)c->stage_slots * HDR_STRIDE,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    v->mr_stage_gpu = reg_gpu_mr(v->pd, c->stage_gpu_pool,
                                 (size_t)c->stage_slots * (size_t)c->cfg.slice_bytes);
    if (!v->mr_stage_hdr || !v->mr_stage_gpu) {
      MLOG_ERR("E1017 staging 注册失败(hdr=%p gpu=%p;GPU 侧需要 dmabuf GDR 支持)",
               (void*)v->mr_stage_hdr, (void*)v->mr_stage_gpu);
      return -1;
    }
    // 接收:host 头槽池 + GPU payload 槽池(每 peer nrx 槽)
    v->rx_pay_bytes = (size_t)c->cfg.slice_bytes;
    const size_t hdr_bytes = (size_t)HDR_STRIDE * nrx * c->cfg.nranks;
    if (posix_memalign((void**)&v->rx_hdr_pool, 4096, hdr_bytes)) {
      MLOG_ERR("E1018 rx 头池分配失败"); return -1;
    }
    memset(v->rx_hdr_pool, 0, hdr_bytes);
    v->mr_rx_hdr = ibv_reg_mr(v->pd, v->rx_hdr_pool, hdr_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    const size_t pay_bytes = v->rx_pay_bytes * nrx * c->cfg.nranks;
    if (cudaMalloc(&v->rx_gpu_pool, pay_bytes) != cudaSuccess) {
      MLOG_ERR("E1018 rx GPU 池分配失败(%zu MB)", pay_bytes >> 20); return -1;
    }
    v->mr_rx_gpu = reg_gpu_mr(v->pd, v->rx_gpu_pool, pay_bytes);
    if (!v->mr_rx_hdr || !v->mr_rx_gpu) {
      MLOG_ERR("E1019 rx 池注册失败(hdr=%p gpu=%p)", (void*)v->mr_rx_hdr, (void*)v->mr_rx_gpu);
      return -1;
    }
  }

  // 控制面槽池(发送 + 每 peer 预投接收)
  {
    if (posix_memalign((void**)&v->ctrl_pool, 4096,
                       (size_t)VerbsCtx::NCTRL * VerbsCtx::CTRL_SLOT)) {
      MLOG_ERR("E1018 ctrl 池分配失败"); return -1;
    }
    v->mr_ctrl = ibv_reg_mr(v->pd, v->ctrl_pool, (size_t)VerbsCtx::NCTRL * VerbsCtx::CTRL_SLOT,
                            IBV_ACCESS_LOCAL_WRITE);
    const size_t crx = (size_t)VerbsCtx::NCTRL_RX * VerbsCtx::CTRL_SLOT * c->cfg.nranks;
    if (posix_memalign((void**)&v->ctrl_rx_pool, 4096, crx)) {
      MLOG_ERR("E1018 ctrl rx 池分配失败"); return -1;
    }
    v->mr_ctrl_rx = ibv_reg_mr(v->pd, v->ctrl_rx_pool, crx, IBV_ACCESS_LOCAL_WRITE);
    if (!v->mr_ctrl || !v->mr_ctrl_rx) { MLOG_ERR("E1017 注册 ctrl 池失败: %s", strerror(errno)); return -1; }
    for (int i = VerbsCtx::NCTRL - 1; i >= 0; i--) v->ctrl_free.push_back(i);
  }

  v->peer.assign(c->cfg.nranks, VerbsPeer());
  ibv_gid mygid{};
  if (ibv_query_gid(v->ctx, v->port, v->gid_index, &mygid)) {
    MLOG_ERR("E1020 query_gid(index=%d) 失败", v->gid_index); return -1;
  }

  // 为每个跨机 peer 建 QP,并经**既有的 TCP 数据连接**交换 QP 信息。
  // 复用 TCP 建链省掉一整套 rendezvous —— 这也是 TCP dconn 保留下来的唯一用途。
  for (int p = 0; p < c->cfg.nranks; p++) {
    if (p == c->cfg.rank || c->topo.same_host(c->cfg.rank, p)) continue;
    VerbsPeer& vp = v->peer[p];
    if (data_plane) {
      ibv_qp_init_attr ia{};
      ia.send_cq = v->cq_send; ia.recv_cq = v->cq_recv;
      ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = (uint32_t)std::max(64, nrx * 4);
      ia.cap.max_recv_wr = (uint32_t)(nrx + 4);
      ia.cap.max_send_sge = 2; ia.cap.max_recv_sge = 2;
      ia.sq_sig_all = 1;
      vp.qp = ibv_create_qp(v->pd, &ia);
      if (!vp.qp) { MLOG_ERR("E1021 create_qp(r%d) 失败", p); return -1; }
    }
    {
      ibv_qp_init_attr ic{};
      ic.send_cq = v->cq_send; ic.recv_cq = v->cq_recv;
      ic.qp_type = IBV_QPT_RC;
      ic.cap.max_send_wr = VerbsCtx::NCTRL + 8;
      ic.cap.max_recv_wr = VerbsCtx::NCTRL_RX + 4;
      ic.cap.max_send_sge = 1; ic.cap.max_recv_sge = 1;
      ic.sq_sig_all = 1;
      vp.qp_ctrl = ibv_create_qp(v->pd, &ic);
      if (!vp.qp_ctrl) { MLOG_ERR("E1021 create_ctrl_qp(r%d) 失败", p); return -1; }
    }
    vp.nrx = nrx;

    QpInfo mine{}, rem{};
    mine.qp_num = vp.qp ? vp.qp->qp_num : 0;
    mine.psn = (uint32_t)(c->cfg.rank * 1000 + p) & 0xFFFFFF;
    mine.ctrl_qp_num = vp.qp_ctrl->qp_num;
    mine.ctrl_psn = (uint32_t)(c->cfg.rank * 1000 + p + 500000) & 0xFFFFFF;
    mine.magic = 0x51504F4B;
    mine.data_plane = data_plane ? 1 : 0;
    memcpy(mine.gid, &mygid, 16);
    if (net_blocking_exchange(c, p, &mine, &rem, sizeof(QpInfo)) != 0) {
      MLOG_ERR("E1022 QP 信息交换失败(r%d)", p); return -1;
    }
    if (rem.magic != 0x51504F4B) { MLOG_ERR("E1023 QP 信息 magic 错"); return -1; }
    if (rem.data_plane != mine.data_plane) {
      MLOG_ERR("E1031 两端数据面形态不一致(本端 %d 对端 %d)", mine.data_plane, rem.data_plane);
      return -1;
    }
    if (data_plane) {
      if (!qp_connect(v, vp.qp, mine.psn, rem.qp_num, rem.psn, rem.gid)) return -1;
      for (int i = 0; i < nrx; i++) if (!post_recv(c, p, i)) return -1;
    }
    if (!qp_connect(v, vp.qp_ctrl, mine.ctrl_psn, rem.ctrl_qp_num, rem.ctrl_psn, rem.gid))
      return -1;
    for (int i = 0; i < VerbsCtx::NCTRL_RX; i++) if (!post_ctrl_recv(c, p, i)) return -1;
  }
  MLOG_INFO("verbs 就绪:%d peer × (%s + ctrl)QP;数据 %d 预投槽/peer(payload = GPU GDR)",
            nremote, data_plane ? "data" : "无 data", nrx);
  return 0;
}

void verbs_teardown(Comm* c) {
  VerbsCtx* v = VC(c);
  if (!v) return;
  for (auto& p : v->peer) {
    if (p.qp) ibv_destroy_qp(p.qp);
    if (p.qp_ctrl) ibv_destroy_qp(p.qp_ctrl);
  }
  if (v->mr_stage_hdr) ibv_dereg_mr(v->mr_stage_hdr);
  if (v->mr_stage_gpu) ibv_dereg_mr(v->mr_stage_gpu);
  if (v->mr_rx_hdr) ibv_dereg_mr(v->mr_rx_hdr);
  if (v->mr_rx_gpu) ibv_dereg_mr(v->mr_rx_gpu);
  if (v->mr_ctrl) ibv_dereg_mr(v->mr_ctrl);
  if (v->mr_ctrl_rx) ibv_dereg_mr(v->mr_ctrl_rx);
  free(v->ctrl_pool);
  free(v->ctrl_rx_pool);
  free(v->rx_hdr_pool);
  if (v->rx_gpu_pool) cudaFree(v->rx_gpu_pool);
  if (v->cq_send) ibv_destroy_cq(v->cq_send);
  if (v->cq_recv) ibv_destroy_cq(v->cq_recv);
  if (v->pd) ibv_dealloc_pd(v->pd);
  if (v->ctx) ibv_close_device(v->ctx);
  delete v;
  c->verbs = nullptr;
}

// 发送一个 slice:双 SGE [host 头槽(此处 CPU 写)| GPU staging payload]。
int verbs_post_slice(Comm* c, int peer, const SliceHdr& h, int stage_slot) {
  VerbsCtx* v = VC(c);
  uint8_t* hslot = c->stage_hdr_pool + (size_t)stage_slot * HDR_STRIDE;
  memcpy(hslot, &h, sizeof(SliceHdr));
  ibv_sge sge[2];
  sge[0].addr = (uint64_t)(uintptr_t)hslot;
  sge[0].length = HDR_STRIDE;
  sge[0].lkey = v->mr_stage_hdr->lkey;
  sge[1].addr = (uint64_t)(uintptr_t)((uint8_t*)c->stage_gpu_pool +
                                      (size_t)stage_slot * (size_t)c->cfg.slice_bytes);
  sge[1].length = h.len;
  sge[1].lkey = v->mr_stage_gpu->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = (uint64_t)stage_slot;          // 低位段 = 数据发送,携带 staging 槽号
  wr.sg_list = sge;
  wr.num_sge = 2;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(v->peer[peer].qp, &wr, &bad)) {
    MLOG_ERR("E1024 post_send(r%d) 失败: %s", peer, strerror(errno));
    return -1;
  }
  return 0;
}

// 控制报文经**独立 ctrl QP** 发出。返回 0 = 已提交;-1 = 不可用(调用方回退 TCP)。
// app 线程(barrier/stats/AG)与 event 线程都会调 —— ibv_post_send 线程安全,
// 槽池由 c->ctrl_mu 保护。
int verbs_ctrl_send(Comm* c, int peer, const void* msg, uint32_t n) {
  VerbsCtx* v = VC(c);
  if (!v || !v->mr_ctrl) return -1;
  VerbsPeer& p = v->peer[peer];
  if (!p.qp_ctrl || n > (uint32_t)VerbsCtx::CTRL_SLOT) return -1;
  int slot;
  {
    std::lock_guard<std::mutex> lk(c->ctrl_mu);
    if (v->ctrl_free.empty()) return -1;         // 池干涸 → 回退 TCP(瞬态)
    slot = v->ctrl_free.back(); v->ctrl_free.pop_back();
  }
  uint8_t* dst = v->ctrl_pool + (size_t)slot * VerbsCtx::CTRL_SLOT;
  memcpy(dst, msg, n);
  ibv_sge sge{};
  sge.addr = (uint64_t)(uintptr_t)dst;
  sge.length = n;
  sge.lkey = v->mr_ctrl->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = WRID_CTRL | (uint32_t)slot;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(p.qp_ctrl, &wr, &bad)) {
    std::lock_guard<std::mutex> lk(c->ctrl_mu);
    v->ctrl_free.push_back(slot);
    return -1;
  }
  return 0;
}

// 轮询完成队列。返回处理的事件数。
int verbs_poll(Comm* c) {
  VerbsCtx* v = VC(c);
  if (!v) return 0;
  int done = 0;
  ibv_wc wc[32];
  int n = ibv_poll_cq(v->cq_send, 32, wc);
  for (int i = 0; i < n; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      MLOG_ERR("E1025 send wc 失败 status=%d(%s)", wc[i].status, ibv_wc_status_str(wc[i].status));
      continue;
    }
    if (wc[i].wr_id & WRID_CTRL) {        // 控制报文完成:只还槽,不进检测器/账
      std::lock_guard<std::mutex> lk(c->ctrl_mu);
      v->ctrl_free.push_back((int)(wc[i].wr_id & 0xFFFFF));
      done++;
      continue;
    }
    int slot = (int)wc[i].wr_id;
    {   // 检测器:NIC↑ 完成事件(字节数从 host 头槽读回)
      SliceHdr h;
      memcpy(&h, c->stage_hdr_pool + (size_t)slot * HDR_STRIDE, sizeof(h));
      if (c->nic_tx_outstanding > 0) c->nic_tx_outstanding--;
      det_feed(c, 0, h.len);
      MTRACE(TE_TXDONE, ((uint32_t)h.src << 16) | h.dst, h.off,
             ((uint64_t)h.len << 8) | h.hop);     // 发送完成(含对端 ack)
    }
    stage_release(c, slot);               // 发送完成 → staging 槽可回收
    done++;
  }
  int rx_done = 0;
  for (;;) {
    ibv_wc w1;
    if (ibv_poll_cq(v->cq_recv, 1, &w1) != 1) break;
    if (w1.status != IBV_WC_SUCCESS) {
      MLOG_ERR("E1025 recv wc 失败 status=%d(%s)", w1.status, ibv_wc_status_str(w1.status));
      continue;
    }
    int peer = (int)((w1.wr_id >> 20) & 0xFFFFF);
    int idx = (int)(w1.wr_id & 0xFFFFF);
    if (w1.wr_id & WRID_CRX) {            // 控制报文:到即处理、槽即补投
      uint8_t* slot = v->ctrl_rx_pool +
                      ((size_t)peer * VerbsCtx::NCTRL_RX + idx) * VerbsCtx::CTRL_SLOT;
      CtrlHdr ch;
      memcpy(&ch, slot, sizeof(ch));
      if (ch.len >= sizeof(CtrlHdr) && ch.len <= (uint32_t)VerbsCtx::CTRL_SLOT &&
          ch.type >= 1 && ch.type <= 8) {
        on_ctrl_msg(c, ch.rank, ch, slot + sizeof(CtrlHdr));
      } else {
        MLOG_ERR("E1026 ctrl 报文头不识别(r%d)", peer);
      }
      post_ctrl_recv(c, peer, idx);
      done++;
      continue;
    }
    // 数据 slice:头在 host 槽(CPU 直读),payload 在 GPU 槽(全程不落 host)
    uint8_t* hslot = v->rx_hdr_pool + ((size_t)peer * v->peer[peer].nrx + idx) * HDR_STRIDE;
    SliceHdr h;
    memcpy(&h, hslot, sizeof(h));
    if (h.magic != SLICE_MAGIC) {
      MLOG_ERR("E1026 数据 QP 收到非 slice 报文(r%d)", peer);
      v->peer[peer].posted--;
      post_recv(c, peer, idx);
      continue;
    }
    // DN busy 提示:本次调用里已有前序完成 —— 上一事件之后端口一直有排队
    // (模拟器 _lg_arr>0 的对应物)
    c->det_dn_hint = (rx_done > 0);
    rx_done++;
    v->peer[peer].posted--;
    MTRACE(TE_SLICE_RECV, ((uint32_t)h.src << 16) | h.dst, h.off,
           ((uint64_t)h.len << 8) | h.hop);       // CQ 命中 = 片到达 dst
    void* gpay = (uint8_t*)v->rx_gpu_pool +
                 ((size_t)peer * v->peer[peer].nrx + idx) * v->rx_pay_bytes;
    // 中转但中转池空:扣着槽交给 wait_relay(on_slice_arrived 内部处理),
    // 槽在交付拷贝(D2D)完成后由 verbs_repost 归还
    on_slice_arrived(c, h, gpay, -1, -1, (peer << 20) | idx);
    done++;
  }
  return done;
}

// 交付拷贝(D2D)完成后补投这个接收槽。**不补投就是背压**:发送端撞 RNR
// 自动退避 —— NIC 下行满了发送方自然慢下来。
void verbs_repost(Comm* c, int token) {
  if (token < 0) return;
  post_recv(c, token >> 20, token & 0xFFFFF);
}

}  // namespace mosaic
