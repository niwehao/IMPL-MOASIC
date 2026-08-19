/* verbs.cc —— vNIC 的 RoCE RDMA 实现(替代 tcp_pipe 作为数据面)。
 *
 * 为什么要有它(而不是一直用 TCP):
 *   TCP 版把 vNIC 的物理上限压在 ~9.2 Gbps/rank,于是模拟的额定值只能取到 2 Gbps
 *   才能保证"限速器是绑定约束"。结论不受影响,但代价是:实验慢 10 倍、
 *   world=2 的真机参考点没意义(被自己的传输层拖死)、而且 Raven 上要用的
 *   verbs 代码路径完全没被验证过。RDMA 版把上限抬到 ~90 Gbps,三件事一起解决。
 *
 * 形态:每个远端 peer 一条 RC QP,SEND/RECV(不是 WRITE)。
 *   选 SEND/RECV 的理由是**流控**:接收端不补投 recv WR 时,发送端会撞 RNR
 *   并由硬件退避 —— 这正是我们要模拟的"链路满了发送方自然慢下来",
 *   而且比 TCP 的接收窗口更贴近真实 NIC 的行为。用 WRITE 的话得自己做信用回传。
 *
 * 与 TCP 路径共用的部分:staging 槽布局、slice 头、限速判定点、上层状态机
 * 完全一致 —— 换传输不改协议(Imp_Plan/01 的分层前提)。
 */
#include "internal.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <cstdlib>

namespace mosaic {

// 每 peer 预投的 recv WR 数。这个数同时是"发送端最多能领先多少个 slice",
// 即在途上界 —— 与 TCP 版的 socket 缓冲深度同角色。
static int rdma_rx_slots() {
  const char* e = getenv("MOSAIC_RDMA_RX_SLOTS");
  return e && *e ? atoi(e) : 16;
}

struct VerbsPeer {
  ibv_qp* qp = nullptr;
  uint32_t qp_num = 0;
  uint32_t psn = 0;
  uint8_t gid[16] = {0};
  uint8_t* rx_base = nullptr;      // 本 peer 的接收槽区(host pinned)
  int nrx = 0;
  int posted = 0;
};

struct VerbsCtx {
  ibv_context* ctx = nullptr;
  ibv_pd* pd = nullptr;
  // send 与 recv 分开两个 CQ:recv 侧逐 slice 消费;send 完成(staging 回收)
  // 不被 recv 侧牵连。
  ibv_cq* cq_send = nullptr;
  ibv_cq* cq_recv = nullptr;
  ibv_mr* mr_stage = nullptr;      // 发送:staging 池
  ibv_mr* mr_rx = nullptr;         // 接收:预投槽池
  void* rx_pool = nullptr;
  size_t rx_slot_bytes = 0;
  int port = 1, gid_index = 3;
  std::vector<VerbsPeer> peer;
  // O3 控制面槽池(MOSAIC_CTRL=verbs):小报文经数据 QP 发送。控制报文
  // 不进检测器/discharge 计数 —— 与 TCP 控制面同口径
  // (协议开销单独按 chA/chB_bytes 计量,禁零成本同步红线不破)。
  static constexpr int CTRL_SLOT = 2048;
  static constexpr int NCTRL = 64;
  uint8_t* ctrl_pool = nullptr;
  ibv_mr* mr_ctrl = nullptr;
  std::vector<int> ctrl_free;      // 由 c->ctrl_mu 保护(app/event 两线程都发)
};
static constexpr uint64_t WRID_CTRL = 1ull << 41;   // send wr_id 高位标记 = 控制报文

static VerbsCtx* VC(Comm* c) { return (VerbsCtx*)c->verbs; }

#pragma pack(push, 1)
struct QpInfo {          // 经已有的 TCP 数据连接交换(建链期,阻塞收发)
  uint32_t qp_num, psn;
  uint8_t gid[16];
  uint32_t magic;
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

static bool to_rtr_rts(Comm* c, VerbsCtx* v, VerbsPeer& p, const QpInfo& rem) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = (uint8_t)v->port;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(p.qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
    MLOG_ERR("E1012 modify INIT 失败"); return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  a.path_mtu = IBV_MTU_1024;      // RoCE 上 1024 最保险(MTU 与以太网帧无关)
  a.dest_qp_num = rem.qp_num;
  a.rq_psn = rem.psn;
  a.max_dest_rd_atomic = 1;
  a.min_rnr_timer = 12;           // RNR 退避 ~640µs:接收端没补投 WR 时靠它背压
  a.ah_attr.is_global = 1;        // RoCE v2 必须走 GRH
  a.ah_attr.port_num = (uint8_t)v->port;
  a.ah_attr.grh.hop_limit = 64;
  a.ah_attr.grh.sgid_index = (uint8_t)v->gid_index;
  memcpy(&a.ah_attr.grh.dgid, rem.gid, 16);
  if (ibv_modify_qp(p.qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
    MLOG_ERR("E1013 modify RTR 失败(GID index %d 对吗?)", v->gid_index); return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTS;
  a.timeout = 14;
  a.retry_cnt = 7;
  a.rnr_retry = 7;                // 无限重试:RNR 是正常的背压,不是错误
  a.sq_psn = p.psn;
  a.max_rd_atomic = 1;
  if (ibv_modify_qp(p.qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
    MLOG_ERR("E1014 modify RTS 失败"); return false;
  }
  return true;
}

static bool post_recv(Comm* c, int peer, int idx) {
  VerbsCtx* v = VC(c);
  VerbsPeer& p = v->peer[peer];
  ibv_sge sge{};
  sge.addr = (uint64_t)(p.rx_base + (size_t)idx * v->rx_slot_bytes);
  sge.length = (uint32_t)v->rx_slot_bytes;
  sge.lkey = v->mr_rx->lkey;
  ibv_recv_wr wr{}, *bad = nullptr;
  wr.wr_id = ((uint64_t)1 << 40) | ((uint64_t)peer << 20) | (uint32_t)idx;  // 高位标记 = recv
  wr.sg_list = &sge;
  wr.num_sge = 1;
  if (ibv_post_recv(p.qp, &wr, &bad)) { MLOG_ERR("E1015 post_recv 失败"); return false; }
  p.posted++;
  return true;
}

int verbs_setup(Comm* c) {
  VerbsCtx* v = new VerbsCtx();
  c->verbs = v;
  v->gid_index = getenv("MOSAIC_GID_INDEX") ? atoi(getenv("MOSAIC_GID_INDEX")) : 3;
  v->ctx = open_active_device(&v->port);
  if (!v->ctx) return -1;
  v->pd = ibv_alloc_pd(v->ctx);
  const int nrx = rdma_rx_slots();
  const int nremote = c->cfg.nranks - c->cfg.gpus_per_host;
  const int cqe = std::max(64, 4 * nrx * std::max(1, nremote));
  v->cq_send = ibv_create_cq(v->ctx, cqe, nullptr, nullptr, 0);
  v->cq_recv = ibv_create_cq(v->ctx, cqe, nullptr, nullptr, 0);
  if (!v->pd || !v->cq_send || !v->cq_recv) { MLOG_ERR("E1016 alloc pd/cq 失败"); return -1; }

  // 发送侧:直接注册已有的 staging 池(cudaHostAlloc 的锁页内存,可注册)
  const size_t stage_bytes = (size_t)c->stage_slots * (size_t)(HDR_STRIDE + c->cfg.slice_bytes);
  v->mr_stage = ibv_reg_mr(v->pd, c->stage_pool, stage_bytes,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!v->mr_stage) { MLOG_ERR("E1017 注册 staging 失败: %s", strerror(errno)); return -1; }

  // 接收侧:每 peer 一组预投槽(与 staging 分开 —— 生命周期不同)
  v->rx_slot_bytes = (size_t)HDR_STRIDE + c->cfg.slice_bytes;
  const size_t rx_bytes = v->rx_slot_bytes * (size_t)nrx * (size_t)c->cfg.nranks;
  if (posix_memalign(&v->rx_pool, 4096, rx_bytes)) { MLOG_ERR("E1018 rx 池分配失败"); return -1; }
  memset(v->rx_pool, 0, rx_bytes);
  v->mr_rx = ibv_reg_mr(v->pd, v->rx_pool, rx_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!v->mr_rx) { MLOG_ERR("E1019 注册 rx 池失败: %s", strerror(errno)); return -1; }

  // O3 控制面槽池(常备;只有 MOSAIC_CTRL=verbs 时才会被用到)
  {
    if (posix_memalign((void**)&v->ctrl_pool, 4096, (size_t)VerbsCtx::NCTRL * VerbsCtx::CTRL_SLOT)) {
      MLOG_ERR("E1018 ctrl 池分配失败"); return -1;
    }
    v->mr_ctrl = ibv_reg_mr(v->pd, v->ctrl_pool, (size_t)VerbsCtx::NCTRL * VerbsCtx::CTRL_SLOT,
                            IBV_ACCESS_LOCAL_WRITE);
    if (!v->mr_ctrl) { MLOG_ERR("E1017 注册 ctrl 池失败: %s", strerror(errno)); return -1; }
    for (int i = VerbsCtx::NCTRL - 1; i >= 0; i--) v->ctrl_free.push_back(i);
  }

  v->peer.assign(c->cfg.nranks, VerbsPeer());
  ibv_gid mygid{};
  if (ibv_query_gid(v->ctx, v->port, v->gid_index, &mygid)) {
    MLOG_ERR("E1020 query_gid(index=%d) 失败", v->gid_index); return -1;
  }

  // 为每个跨机 peer 建 QP,并经**已有的 TCP 数据连接**交换 QP 信息。
  // 复用 TCP 建链省掉一整套 rendezvous —— 这也是 TCP 通道保留下来的唯一用途。
  for (int p = 0; p < c->cfg.nranks; p++) {
    if (p == c->cfg.rank || c->topo.same_host(c->cfg.rank, p)) continue;
    VerbsPeer& vp = v->peer[p];
    ibv_qp_init_attr ia{};
    ia.send_cq = v->cq_send; ia.recv_cq = v->cq_recv;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = (uint32_t)std::max(64, nrx * 4);
    ia.cap.max_recv_wr = (uint32_t)(nrx + 4);
    ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    ia.sq_sig_all = 1;
    vp.qp = ibv_create_qp(v->pd, &ia);
    if (!vp.qp) { MLOG_ERR("E1021 create_qp(r%d) 失败", p); return -1; }
    vp.qp_num = vp.qp->qp_num;
    vp.psn = (uint32_t)(c->cfg.rank * 1000 + p) & 0xFFFFFF;
    vp.nrx = nrx;
    vp.rx_base = (uint8_t*)v->rx_pool + (size_t)p * nrx * v->rx_slot_bytes;

    QpInfo mine{}, rem{};
    mine.qp_num = vp.qp_num; mine.psn = vp.psn; mine.magic = 0x51504F4B;
    memcpy(mine.gid, &mygid, 16);
    if (net_blocking_exchange(c, p, &mine, &rem, sizeof(QpInfo)) != 0) {
      MLOG_ERR("E1022 QP 信息交换失败(r%d)", p); return -1;
    }
    if (rem.magic != 0x51504F4B) { MLOG_ERR("E1023 QP 信息 magic 错"); return -1; }
    if (!to_rtr_rts(c, v, vp, rem)) return -1;
    for (int i = 0; i < nrx; i++) if (!post_recv(c, p, i)) return -1;
  }
  MLOG_INFO("RoCE 就绪:%d 条 RC QP,每 peer %d 个预投槽(%zu KB/槽)",
            nremote, nrx, v->rx_slot_bytes >> 10);
  return 0;
}

void verbs_teardown(Comm* c) {
  VerbsCtx* v = VC(c);
  if (!v) return;
  for (auto& p : v->peer) if (p.qp) ibv_destroy_qp(p.qp);
  if (v->mr_stage) ibv_dereg_mr(v->mr_stage);
  if (v->mr_rx) ibv_dereg_mr(v->mr_rx);
  if (v->mr_ctrl) ibv_dereg_mr(v->mr_ctrl);
  free(v->ctrl_pool);
  free(v->rx_pool);
  if (v->cq_send) ibv_destroy_cq(v->cq_send);
  if (v->cq_recv) ibv_destroy_cq(v->cq_recv);
  if (v->pd) ibv_dealloc_pd(v->pd);
  if (v->ctx) ibv_close_device(v->ctx);
  delete v;
  c->verbs = nullptr;
}

// 发送一个 slice:staging 槽里已经是 [头 HDR_STRIDE 字节][payload],一次 SEND 发走
int verbs_post_slice(Comm* c, int peer, const SliceHdr& h, void* slot_base, int stage_slot) {
  VerbsCtx* v = VC(c);
  memcpy(slot_base, &h, sizeof(SliceHdr));
  ibv_sge sge{};
  sge.addr = (uint64_t)slot_base;
  sge.length = (uint32_t)(HDR_STRIDE + h.len);
  sge.lkey = v->mr_stage->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = (uint64_t)stage_slot;          // 低位段 = send,携带 staging 槽号
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(v->peer[peer].qp, &wr, &bad)) {
    MLOG_ERR("E1024 post_send(r%d) 失败: %s", peer, strerror(errno));
    return -1;
  }
  return 0;
}

// O3:控制报文经数据 QP 发出。返回 0 = 已提交;-1 = 不可用(调用方回退 TCP)。
// app 线程(barrier/stats/AG)与 event 线程都会调 —— ibv_post_send 线程安全,
// 槽池由 c->ctrl_mu 保护(与 TCP 控制队列同一把锁,顺路保证同 peer 的
// TCP/verbs 混发不会因回退交错乱序 —— 回退只在池干涸的瞬态发生)。
int verbs_ctrl_send(Comm* c, int peer, const void* msg, uint32_t n) {
  VerbsCtx* v = VC(c);
  if (!v || !v->mr_ctrl) return -1;
  VerbsPeer& p = v->peer[peer];
  if (!p.qp || n > (uint32_t)VerbsCtx::CTRL_SLOT) return -1;
  int slot;
  {
    std::lock_guard<std::mutex> lk(c->ctrl_mu);
    if (v->ctrl_free.empty()) return -1;         // 池干涸 → 回退 TCP(瞬态)
    slot = v->ctrl_free.back(); v->ctrl_free.pop_back();
  }
  uint8_t* dst = v->ctrl_pool + (size_t)slot * VerbsCtx::CTRL_SLOT;
  memcpy(dst, msg, n);
  ibv_sge sge{};
  sge.addr = (uint64_t)dst;
  sge.length = n;
  sge.lkey = v->mr_ctrl->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = WRID_CTRL | (uint32_t)slot;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  if (ibv_post_send(p.qp, &wr, &bad)) {
    std::lock_guard<std::mutex> lk(c->ctrl_mu);
    v->ctrl_free.push_back(slot);
    return -1;
  }
  return 0;
}

// 轮询完成队列。返回处理的事件数。
// 控制报文(O3,首 4 字节 ≠ SLICE_MAGIC)不进检测器/计数,
// 到即处理、槽即补投 —— 与 TCP 控制面同语义。
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
    {   // 检测器:NIC↑ 完成事件(字节数从槽首的 SliceHdr 读回)
      SliceHdr h;
      memcpy(&h, (uint8_t*)c->stage_pool + (size_t)slot * (HDR_STRIDE + (size_t)c->cfg.slice_bytes),
             sizeof(h));
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
    uint8_t* slot = v->peer[peer].rx_base + (size_t)idx * v->rx_slot_bytes;
    uint32_t m0;
    memcpy(&m0, slot, sizeof(m0));
    if (m0 != SLICE_MAGIC) {
      // O3 控制报文:不过限速门/检测器/token bucket(与 TCP 控制面同口径),
      // 到即处理、槽即补投(不占数据背压的账)
      CtrlHdr ch;
      memcpy(&ch, slot, sizeof(ch));
      if (ch.len >= sizeof(CtrlHdr) && ch.len <= v->rx_slot_bytes && ch.type >= 1 && ch.type <= 8) {
        v->peer[peer].posted--;
        on_ctrl_msg(c, ch.rank, ch, slot + sizeof(CtrlHdr));
        post_recv(c, peer, idx);
        done++;
        continue;
      }
      MLOG_ERR("E1026 RDMA 报文 magic/头都不识别(r%d)", peer);
      v->peer[peer].posted--;
      post_recv(c, peer, idx);
      continue;
    }
    // DN busy 提示:本次调用里已有前序完成 —— 说明上一事件之后端口一直有排队
    // (模拟器 _lg_arr>0 的对应物)
    c->det_dn_hint = (rx_done > 0);
    rx_done++;
    v->peer[peer].posted--;
    SliceHdr h;
    memcpy(&h, slot, sizeof(h));
    MTRACE(TE_SLICE_RECV, ((uint32_t)h.src << 16) | h.dst, h.off,
           ((uint64_t)h.len << 8) | h.hop);       // CQ 命中 = 片到达 dst
    // 中转但中转池空:与 pump_shm 同款的队头处理 —— 扣着槽交给 wait_relay
    // (on_slice_arrived 内部处理),槽在 H2D 完成后由 verbs_repost 归还
    on_slice_arrived(c, h, slot + HDR_STRIDE, -1, -1, (peer << 20) | idx);
    done++;
  }
  return done;
}

// H2D 完成后补投这个接收槽。**不补投就是背压**:发送端会撞 RNR 自动退避,
// 这正是"vNIC 下行满了发送方慢下来"的忠实模拟。
void verbs_repost(Comm* c, int token) {
  if (token < 0) return;
  post_recv(c, token >> 20, token & 0xFFFFF);
}

}  // namespace mosaic
