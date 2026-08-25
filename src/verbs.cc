/* verbs.cc —— RDMA 数据面 + 控制面(RoCE 与 IB 同一份代码,运行时探测)。
 *
 * ── 统一搬运 API:搬运代码只有一份,平台差异只落在「池在哪块内存」。
 *   拷贝一律由消费者核(SM,consumer_kernel.cu)做,UVA 指针对 host pinned
 *   与显存一视同仁;NIC 一侧经 MR 直接 DMA 池内存。因此四种组合:
 *     TX staging 池:host pinned(NIC 从 host MR 读)| 显存(dmabuf MR,GDR 读)
 *     RX 槽池:      host pinned(NIC 写 host,交付 UVA 读)| 显存(GDR 写,交付 D2D)
 *   全部走同一条代码路径,仅差池的 cudaHostAlloc/cudaMalloc 与 MR 注册方式。
 *   模式由 xport_profile 在 init 用 loopback QP **双向**带宽实测选定
 *   (NCCL init 挑选机制的同款思路;两个选择都是本 rank 局部的,对端不感知)。
 *   平台先验(profile 应能复现,MOSAIC_XPORT 可显式钉住):
 *     GDR 读过 PCIe host-bridge P2P,Raven 实测 ~121 Gbps(host 读 197 满速);
 *     GDR 写单向满速(197),但 Raven 双向持续流下随机进入劣化态(收端 HCA
 *     排水滞流 → 链路信用饿死发端,双稳态 195/26 Gbps;RX=host 恒满速)——
 *     所以 profile 必须测双向,不测单向。
 *   RX=显存时 CQE 不保证 payload 已落 GPU(PCIe posted write 与 CQE 到达
 *   不同终点),交付前经 loopback flush QP 发一笔 RDMA READ 强制排水
 *   (NCCL gdrFlush 同款)。
 *
 *   SEND/RECV 都用双 SGE:[64B SliceHdr(host pinned,CPU 直读写)|payload]。
 *   控制面独立 QP(每 remote peer 一条):报文 ≤2KB 落 host 槽,与 slice
 *   FIFO 解耦;跨机控制面**恒走 verbs SEND**(单程 ~5µs vs TCP 内核栈 30-60µs),
 *   建链失败自动回退 TCP(告警,确定性)。
 *
 * 平台差异全部运行时探测,无静态名单:
 *   链路层 = 端口属性(IB → LID 路由 + 端口 active MTU;RoCE → GRH + MTU1024);
 *   HCA = 收集全部 ACTIVE 端口,有 IB 就只在 IB 里选(以太口不算数据网),
 *         再按 local rank 轮转(多网卡节点各 rank 分摊,单卡机退化为原行为)。
 *
 * 形态:每个远端 peer 一条数据 RC QP + 一条控制 RC QP,SEND/RECV。
 *   选 SEND/RECV 的理由是**流控**:接收端不补投 recv WR 时,发送端撞 RNR
 *   并由硬件退避 —— 链路满了发送方自然慢下来,无需自己做信用回传。
 *
 */
#include "internal.h"

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <deque>

namespace mosaic {

// 每 peer 预投的数据 recv WR 数 = 发送端最多领先的 slice 数(在途上界)。
// 默认按字节定量:窗口罩住 32MB 在途(标定:8MB 窗在小切片下实测把线速
// 掐到 17.8%;32MB/pair 传输在窗口不足时每片撞一次 RNR 退避)。上限 64 槽
// 防止小 slice 下 host 槽池爆炸(64 × slice × peer 数)。
int rdma_rx_slots(int slice_bytes) {   // engine 的 W 规则也读(聚合接收窗)
  const char* e = getenv("MOSAIC_RDMA_RX_SLOTS");
  if (e && *e) return atoi(e);
  int n = (int)(((size_t)32 << 20) / (size_t)(slice_bytes > 0 ? slice_bytes : 1 << 20));
  return n < 16 ? 16 : (n > 64 ? 64 : n);
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
  bool data_plane = false;         // false = ctrl-only(调试形态,正常恒 true)
  int port = 1, gid_index = 3;
  bool roce = true;                // 端口链路层:RoCE(GRH)| IB(LID 路由)
  ibv_mtu mtu = IBV_MTU_1024;      // 端口 active MTU(与对端取 min)
  uint16_t lid = 0;                // IB LID(RoCE 恒 0)
  uint64_t dev_key = 0;            // 设备身份 hash(devname:port)—— NIC 单元共识用
  double link_gbps = 0;            // 端口物理速率(active_speed × width)
  // ── 数据面缓冲(data_plane 才建)
  ibv_mr* mr_stage_hdr = nullptr;  // 发送:host 头池(engine 的 stage_hdr_pool)
  ibv_mr* mr_stage_pay = nullptr;  // 发送:host pinned payload 池(engine 的 stage_pay_pool)
  uint8_t* rx_hdr_pool = nullptr;  // 接收:host 头槽池 [peer][idx] × HDR_STRIDE
  ibv_mr* mr_rx_hdr = nullptr;
  void* rx_pay_pool = nullptr;     // 接收:payload 槽池 [peer][idx] × slice_bytes(位置随 rx_gdr)
  void* rx_pay_raw = nullptr;      // rx_gdr 时对齐分配的原始指针(释放用)
  ibv_mr* mr_rx_pay = nullptr;
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
  // ── RX=显存(rx_gdr)的交付顺序保障:CQE 到达不保证 payload 已写完显存
  // (PCIe posted write 到 GPU BAR 与 CQE 到 host RAM 是不同终点)。到达的
  // slice 先入 gdr_pend,经 loopback flush QP 发 8B RDMA READ(非 posted,
  // 强制同路径 posted write 先排水),READ 完成后才交付读完成前入队的那批。
  struct GdrPend { SliceHdr h; void* gpay; int token; };
  ibv_qp* qp_flush = nullptr;           // 自连 RC QP(仅 rx_gdr 时建)
  std::deque<GdrPend> gdr_pend;
  size_t flush_cover = 0;               // 在途 flush 覆盖 gdr_pend 前多少条
  uint64_t flush_post_ns = 0;           // 在途 flush 的投递时刻(活性兜底判据)
  uint8_t* flush_buf = nullptr;         // 8B host 落点
  ibv_mr* mr_flush = nullptr;
};
// wr_id 位段:bit40 = 数据 recv;bit41 = ctrl 发送(低位 = 槽号);
// bit42 = ctrl recv;bit43 = GDR flush READ;数据发送 = 纯低位(staging 槽号)。
static constexpr uint64_t WRID_RX = 1ull << 40;
static constexpr uint64_t WRID_CTRL = 1ull << 41;
static constexpr uint64_t WRID_CRX = 1ull << 42;
static constexpr uint64_t WRID_FLUSH = 1ull << 43;

// ── dmabuf 注册(显存池唯一的注册途径;经典 nv_peer_mem 不要求)。
// libcuda 经 dlopen(与 engine.cc 的 PTX JIT 同款做法,CPU-only 构建不受影响)。
// dmabuf 导出范围必须 host 页对齐,而 cudaMalloc 小块只保证 1KB 对齐(驱动
// 大分配里的子分配)⇒ 所有会被 dmabuf 注册的显存池一律经
// mosaic_gpu_alloc_aligned 拿 64KB 对齐指针(多留一页余量,导出范围不越界)。
void* mosaic_gpu_alloc_aligned(size_t bytes, void** raw) {
  void* p = nullptr;
  const size_t need = ((bytes + 4095) & ~(size_t)4095) + 65536;
  if (cudaMalloc(&p, need) != cudaSuccess) return nullptr;
  *raw = p;
  return (void*)(((uintptr_t)p + 65535) & ~(uintptr_t)65535);
}
typedef int (*cu_dmabuf_fn)(int* fd, unsigned long long ptr, size_t size, int type,
                            unsigned long long flags);
static bool cu_dmabuf_fd(void* ptr, size_t len, int* fd_out) {
  static cu_dmabuf_fn f_dmabuf = nullptr;
  static bool init = false;
  if (!init) {
    init = true;
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (h) f_dmabuf = (cu_dmabuf_fn)dlsym(h, "cuMemGetHandleForAddressRange");
  }
  if (!f_dmabuf || ((uintptr_t)ptr & 4095)) return false;
  int fd = -1;
  const size_t size = (len + 4095) & ~(size_t)4095;
  if (f_dmabuf(&fd, (unsigned long long)(uintptr_t)ptr, size, /*DMA_BUF_FD*/ 1, 0) != 0 ||
      fd < 0)
    return false;
  *fd_out = fd;
  return true;
}
// 统一注册入口:host 内存 ibv_reg_mr;显存 dmabuf 导出 + ibv_reg_dmabuf_mr
// (iova 传指针本身 ⇒ SGE/远端地址两种池同一写法)。失败返回 null。
static ibv_mr* reg_mr_any(ibv_pd* pd, void* ptr, size_t len, bool gpu, int access) {
  if (!gpu) return ibv_reg_mr(pd, ptr, len, access);
  int fd = -1;
  if (!cu_dmabuf_fd(ptr, len, &fd)) {
    MLOG_WARN("E1038 dmabuf 导出失败(ptr=%p len=%zu)", ptr, len);
    return nullptr;
  }
  ibv_mr* mr = ibv_reg_dmabuf_mr(pd, 0, len, (uint64_t)(uintptr_t)ptr, fd, access);
  if (!mr) MLOG_WARN("E1038 reg_dmabuf 失败(errno=%d len=%zu)", errno, len);
  close(fd);                                 // MR 持有 dmabuf 引用,fd 可即关
  return mr;
}

static VerbsCtx* VC(Comm* c) { return (VerbsCtx*)c->verbs; }

#pragma pack(push, 1)
struct QpInfo {          // 经既有的 TCP 数据连接交换(建链期,阻塞收发)
  uint32_t qp_num, psn;
  uint32_t ctrl_qp_num, ctrl_psn;
  uint8_t gid[16];
  uint32_t magic;
  uint8_t data_plane;    // 双方必须一致(verbs 数据面 vs ctrl-only)
  uint8_t mtu;           // 本端 active MTU(ibv_mtu 枚举;双方取 min)
  uint16_t lid;          // IB LID;非 0 = 对端走 LID 路由(RoCE 恒 0 → GRH)
};
#pragma pack(pop)

// 自动挑卡:按端口状态选(设备名跨机不同,按名字分支换机器就悄悄挂)。
// 全部 ACTIVE 端口里,有 IB 链路层就只在 IB 里选(以太口不算数据网);
// 多候选按 local rank 轮转(多网卡节点各 rank 分摊带宽,单卡机退化不变)。
static ibv_context* open_active_device(int local_rank, int* port_out) {
  int n = 0;
  ibv_device** list = ibv_get_device_list(&n);
  if (!list || n == 0) { MLOG_ERR("E1010 没有 RDMA 设备"); return nullptr; }
  const char* want = getenv("MOSAIC_HCA");
  struct Cand { int dev, port; bool ib; };
  std::vector<Cand> cand;
  std::vector<ibv_context*> opened(n, nullptr);
  bool any_ib = false;
  for (int i = 0; i < n; i++) {
    const char* nm = ibv_get_device_name(list[i]);
    if (want && *want && strcmp(nm, want) != 0) continue;
    ibv_context* ctx = ibv_open_device(list[i]);
    if (!ctx) continue;
    opened[i] = ctx;
    ibv_port_attr pa{};
    for (int p = 1; p <= 2; p++) {
      if (ibv_query_port(ctx, p, &pa) == 0 && pa.state == IBV_PORT_ACTIVE) {
        bool ib = (pa.link_layer != IBV_LINK_LAYER_ETHERNET);
        cand.push_back({i, p, ib});
        any_ib |= ib;
      }
    }
  }
  if (any_ib)
    cand.erase(std::remove_if(cand.begin(), cand.end(),
                              [](const Cand& x) { return !x.ib; }),
               cand.end());
  ibv_context* ret = nullptr;
  if (!cand.empty()) {
    const Cand& pick = cand[(size_t)local_rank % cand.size()];
    ret = opened[pick.dev];
    *port_out = pick.port;
    MLOG_INFO("RDMA 设备 %s port %d ACTIVE(候选 %zu,local_rank %d)",
              ibv_get_device_name(list[pick.dev]), pick.port, cand.size(), local_rank);
    opened[pick.dev] = nullptr;
  }
  for (int i = 0; i < n; i++)
    if (opened[i]) ibv_close_device(opened[i]);
  ibv_free_device_list(list);
  if (!ret) MLOG_ERR("E1011 没有 ACTIVE 的 RDMA 端口");
  return ret;
}

static bool qp_connect(VerbsCtx* v, ibv_qp* qp, uint32_t my_psn, uint32_t rem_qpn,
                       uint32_t rem_psn, const QpInfo& rem) {
  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.pkey_index = 0;
  a.port_num = (uint8_t)v->port;
  // REMOTE_READ:flush QP 对自身发 RDMA READ(rx_gdr 交付排水),其余 QP 不用但无害
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
    MLOG_ERR("E1012 modify INIT 失败"); return false;
  }
  memset(&a, 0, sizeof(a));
  a.qp_state = IBV_QPS_RTR;
  // 路径 MTU = 双方 active MTU 取 min(IB HDR = 4096;RoCE 以太帧下 = 1024)
  a.path_mtu = (ibv_mtu)std::min((int)v->mtu, (int)rem.mtu);
  a.dest_qp_num = rem_qpn;
  a.rq_psn = rem_psn;
  a.max_dest_rd_atomic = 1;
  // RNR 退避 0.16ms(编码 8):信用反压下发送端不超过对端授权窗,RNR 是
  // 稀有兜底事件,取高档防极端 incast 下重试包频锤热 HCA(标定:10µs 档
  // 实测数百 WR × 100kHz 重试把热 rank 出向发包引擎饿死 ⇒ E1025 status=12;
  // 0.16ms 使重试频率降 16×,空转税由窗内深度掩盖)。
  a.min_rnr_timer = 8;
  a.ah_attr.port_num = (uint8_t)v->port;
  if (rem.lid) {                  // IB:LID 路由(对端 lid 非 0 才可能)
    a.ah_attr.is_global = 0;
    a.ah_attr.dlid = rem.lid;
  } else {                        // RoCE v2:必须走 GRH
    a.ah_attr.is_global = 1;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.grh.sgid_index = (uint8_t)v->gid_index;
    memcpy(&a.ah_attr.grh.dgid, rem.gid, 16);
  }
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
  sge[1].addr = (uint64_t)(uintptr_t)((uint8_t*)v->rx_pay_pool +
                                      ((size_t)peer * nrx + idx) * v->rx_pay_bytes);
  sge[1].length = (uint32_t)v->rx_pay_bytes;
  sge[1].lkey = v->mr_rx_pay->lkey;
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

// ─────────────────────────────── xport profile(统一搬运 API 的模式选择)
// loopback 形态:同一 HCA 上两条 RC QP 互连,A→B 与 B→A 同时打满 —— TX DMA
// 读与 RX DMA 写两个引擎持续并发,复现数据面的「双向持续流」排水路径
// (GDR 写的劣化态只在双向出现,单向测必误选,见文件头注)。每组合搬
// 2×64MB 计聚合 Gbps,跑两轮取第二轮(首轮暖 QP/页表);与最优差 <5% 的
// 组合里偏向 GDR 位更少的(排水劣化风险最低)。病态组合(3s 不完)记 0。
void xport_profile(Comm* c) {
  c->tx_gdr = false;
  c->rx_gdr = false;
  // MOSAIC_XPORT:统一搬运 API 的池组合(平台配置类旋钮)。
  //   ①控制什么:TX staging 池 / RX 槽池在 host pinned 还是显存(dmabuf/GDR);
  //   ②默认 auto = 本函数 loopback 双向实测选 argmax(local-0825-xport 标定,
  //     NCCL init 挑选机制同款);③钉 gdr 省一次 SM 拷贝路的 host RAM 往返但
  //     吃 PCIe P2P 上限(Raven GDR 读 ~121Gbps)与双向排水劣化风险,钉 pin
  //     恒走 host-bounce(两平台实测满线速的保守档)。四组合功能上都能跑。
  const char* e = getenv("MOSAIC_XPORT");
  bool by_env = false, want_tx = false, want_rx = false;
  if (e && *e && strcmp(e, "auto") != 0) {
    char a[8] = {0}, b[8] = {0};
    if (sscanf(e, "%3[a-z],%3[a-z]", a, b) == 2 &&
        (!strcmp(a, "gdr") || !strcmp(a, "pin")) &&
        (!strcmp(b, "gdr") || !strcmp(b, "pin"))) {
      want_tx = !strcmp(a, "gdr");
      want_rx = !strcmp(b, "gdr");
      by_env = true;
    } else {
      MLOG_WARN("E1034 MOSAIC_XPORT=%s 无效(auto|pin,pin|gdr,pin|pin,gdr|gdr,gdr),按 auto", e);
    }
  }
  int local_rank = 0;
  for (int p = 0; p < c->cfg.rank; p++)
    if (c->topo.same_host(c->cfg.rank, p)) local_rank++;
  VerbsCtx pv;                       // 借 qp_connect 的 port/mtu/gid_index 语境
  pv.ctx = open_active_device(local_rank, &pv.port);
  if (!pv.ctx) return;               // 无 RDMA 设备:数据面建链稍后自会硬错
  {
    ibv_port_attr pa{};
    if (ibv_query_port(pv.ctx, pv.port, &pa)) { ibv_close_device(pv.ctx); return; }
    pv.roce = (pa.link_layer == IBV_LINK_LAYER_ETHERNET);
    pv.mtu = pa.active_mtu;
    pv.lid = pv.roce ? 0 : pa.lid;
    pv.gid_index = getenv("MOSAIC_GID_INDEX") ? atoi(getenv("MOSAIC_GID_INDEX"))
                                              : (pv.roce ? 3 : 0);
  }
  static constexpr size_t SLICE = 1 << 20;
  // INFLIGHT < NRXS ⇒ 发送永远追不上补投,结构上零 RNR —— 否则 min_rnr
  // 0.16ms 的退避把四组合一起钳死在同一个假值,测不出内存路径差
  static constexpr int NRXS = 16;    // 每 QP 预投槽
  static constexpr int NSEND = 64;   // 每 QP 计时发送数
  static constexpr int INFLIGHT = 4;
  const size_t rx_bytes = SLICE * NRXS * 2;
  ibv_pd* pd = ibv_alloc_pd(pv.ctx);
  ibv_cq* cq = pd ? ibv_create_cq(pv.ctx, 1024, nullptr, nullptr, 0) : nullptr;
  void* htx = nullptr; void* hrx = nullptr;
  void* gtx_raw = nullptr; void* grx_raw = nullptr;
  cudaHostAlloc(&htx, SLICE, cudaHostAllocPortable);
  cudaHostAlloc(&hrx, rx_bytes, cudaHostAllocPortable);
  void* gtx = mosaic_gpu_alloc_aligned(SLICE, &gtx_raw);
  void* grx = mosaic_gpu_alloc_aligned(rx_bytes, &grx_raw);
  bool gdr_ok = gtx && grx;
  ibv_mr* mhtx = pd && htx ? ibv_reg_mr(pd, htx, SLICE, IBV_ACCESS_LOCAL_WRITE) : nullptr;
  ibv_mr* mhrx = pd && hrx ? ibv_reg_mr(pd, hrx, rx_bytes, IBV_ACCESS_LOCAL_WRITE) : nullptr;
  ibv_mr* mgtx = pd && gdr_ok ? reg_mr_any(pd, gtx, SLICE, true, IBV_ACCESS_LOCAL_WRITE) : nullptr;
  ibv_mr* mgrx = pd && gdr_ok ? reg_mr_any(pd, grx, rx_bytes, true, IBV_ACCESS_LOCAL_WRITE) : nullptr;
  gdr_ok = mgtx && mgrx;
  ibv_qp* qp[2] = {nullptr, nullptr};
  bool ok = pd && cq && mhtx && mhrx;
  for (int i = 0; ok && i < 2; i++) {
    ibv_qp_init_attr ia{};
    ia.send_cq = cq; ia.recv_cq = cq;
    ia.qp_type = IBV_QPT_RC;
    ia.cap.max_send_wr = NSEND + 8;
    ia.cap.max_recv_wr = NRXS + 4;
    ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
    ia.sq_sig_all = 1;
    qp[i] = ibv_create_qp(pd, &ia);
    ok = qp[i] != nullptr;
  }
  ibv_gid gid{};
  if (ok) ok = ibv_query_gid(pv.ctx, pv.port, pv.gid_index, &gid) == 0;
  for (int i = 0; ok && i < 2; i++) {
    QpInfo rem{};
    rem.mtu = (uint8_t)pv.mtu;
    rem.lid = pv.lid;
    memcpy(rem.gid, &gid, 16);
    ok = qp_connect(&pv, qp[i], 100u + i, qp[1 - i]->qp_num, 100u + (1 - i), rem);
  }
  // 每组合两个量,取 min 打分(单位都是 **每方向 Gbps**):
  //   uni  = 单向(只 A→B):上限 = 线速,能分辨「NIC 读显存被 PCIe P2P 钳制」
  //          一类的单路径上限(hairpin 双向每方向只有半线速,会盖住它);
  //   bi   = 双向(A↔B 同时打满):暴露「RX 落显存 + 双向持续流」的排水劣化。
  double bw_uni[2][2] = {{0, 0}, {0, 0}};
  double bw_bi[2][2] = {{0, 0}, {0, 0}};
  auto run_combo = [&](bool tg, bool rg, bool bidir) -> double {
    uint8_t* txp = (uint8_t*)(tg ? gtx : htx);
    uint8_t* rxp = (uint8_t*)(rg ? grx : hrx);
    ibv_mr* mtx = tg ? mgtx : mhtx;
    ibv_mr* mrx = rg ? mgrx : mhrx;
    auto post_r = [&](int q, int slot) -> bool {
      ibv_sge s{};
      s.addr = (uint64_t)(uintptr_t)(rxp + ((size_t)q * NRXS + slot) * SLICE);
      s.length = (uint32_t)SLICE;
      s.lkey = mrx->lkey;
      ibv_recv_wr wr{}, *bad = nullptr;
      wr.wr_id = WRID_RX | ((uint64_t)q << 20) | (uint32_t)slot;
      wr.sg_list = &s; wr.num_sge = 1;
      return ibv_post_recv(qp[q], &wr, &bad) == 0;
    };
    auto post_s = [&](int q) -> bool {
      ibv_sge s{};
      s.addr = (uint64_t)(uintptr_t)txp;
      s.length = (uint32_t)SLICE;
      s.lkey = mtx->lkey;
      ibv_send_wr wr{}, *bad = nullptr;
      wr.wr_id = (uint64_t)q;
      wr.sg_list = &s; wr.num_sge = 1;
      wr.opcode = IBV_WR_SEND;
      wr.send_flags = IBV_SEND_SIGNALED;
      return ibv_post_send(qp[q], &wr, &bad) == 0;
    };
    const int nsend_q = bidir ? 2 : 1;         // 发送方:双向 = 两条 QP,单向只 A
    int rposted[2] = {0, 0}, sent[2] = {0, 0}, sdone[2] = {0, 0}, rdone = 0;
    for (int q = 0; q < 2; q++)
      for (int i = 0; i < NRXS; i++)
        if (post_r(q, i)) rposted[q]++;
    const uint64_t t0 = now_ns();
    const uint64_t deadline = t0 + 3000000000ull;
    while (rdone < nsend_q * NSEND || sdone[0] + sdone[1] < nsend_q * NSEND) {
      for (int q = 0; q < nsend_q; q++)
        while (sent[q] < NSEND && sent[q] - sdone[q] < INFLIGHT && post_s(q)) sent[q]++;
      ibv_wc wc[16];
      int n = ibv_poll_cq(cq, 16, wc);
      for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) return 0;
        if (wc[i].wr_id & WRID_RX) {
          rdone++;
          int q = (int)((wc[i].wr_id >> 20) & 0xF);
          int slot = (int)(wc[i].wr_id & 0xFFFFF);
          if (rposted[q] < NSEND && post_r(q, slot)) rposted[q]++;
        } else {
          sdone[(int)wc[i].wr_id]++;
        }
      }
      if (now_ns() > deadline) return 0;   // 病态组合:按不可行记 0
    }
    const uint64_t dt = now_ns() - t0;
    // 每方向 Gbps:单向 = 全量/时;双向 = 聚合的一半,同一表达式
    return dt ? (double)((uint64_t)NSEND * SLICE) * 8.0 / (double)dt : 0;
  };
  if (ok) {
    for (int pass = 0; pass < 2; pass++) {     // 跑两轮取第二轮(首轮暖机)
      for (int t = 0; t < (gdr_ok ? 2 : 1); t++)
        for (int r = 0; r < (gdr_ok ? 2 : 1); r++) {
          bw_uni[t][r] = run_combo(t, r, false);
          bw_bi[t][r] = run_combo(t, r, true);
        }
    }
  }
  double bw[2][2];                             // 打分 = min(单向, 双向每方向)
  for (int t = 0; t < 2; t++)
    for (int r = 0; r < 2; r++)
      bw[t][r] = std::min(bw_uni[t][r], bw_bi[t][r]);
  if (by_env) {
    if ((want_tx || want_rx) && !gdr_ok) {
      MLOG_WARN("E1035 MOSAIC_XPORT 要求 GDR 但 dmabuf 注册不可用,退 pin,pin");
      want_tx = want_rx = false;
    }
    c->tx_gdr = want_tx;
    c->rx_gdr = want_rx;
  } else {
    double mx = 0;
    for (int t = 0; t < 2; t++)
      for (int r = 0; r < 2; r++)
        if (bw[t][r] > mx) mx = bw[t][r];
    int bt = 0, br = 0, bits = 3;
    bool found = false;
    for (int t = 0; t < 2; t++)
      for (int r = 0; r < 2; r++)
        if (mx > 0 && bw[t][r] >= mx * 0.95 && (!found || t + r < bits)) {
          found = true; bt = t; br = r; bits = t + r;
        }
    c->tx_gdr = bt == 1;
    c->rx_gdr = br == 1;
  }
  MLOG_INFO("xport profile(每方向 Gbps,uni/bi):pin,pin=%.0f/%.0f gdr,pin=%.0f/%.0f "
            "pin,gdr=%.0f/%.0f gdr,gdr=%.0f/%.0f%s → TX=%s RX=%s%s",
            bw_uni[0][0], bw_bi[0][0], bw_uni[1][0], bw_bi[1][0],
            bw_uni[0][1], bw_bi[0][1], bw_uni[1][1], bw_bi[1][1],
            gdr_ok ? "" : "(dmabuf 不可用)",
            c->tx_gdr ? "gdr" : "pin", c->rx_gdr ? "gdr" : "pin",
            by_env ? "(MOSAIC_XPORT 钉住)" : "");
  for (auto* q : qp) if (q) ibv_destroy_qp(q);
  for (auto* m : {mhtx, mhrx, mgtx, mgrx}) if (m) ibv_dereg_mr(m);
  if (cq) ibv_destroy_cq(cq);
  if (pd) ibv_dealloc_pd(pd);
  ibv_close_device(pv.ctx);
  if (htx) cudaFreeHost(htx);
  if (hrx) cudaFreeHost(hrx);
  if (gtx_raw) cudaFree(gtx_raw);
  if (grx_raw) cudaFree(grx_raw);
}

// data_plane=true:数据面 + 控制面;false:仅控制面(调试形态)。
int verbs_setup(Comm* c, bool data_plane) {
  VerbsCtx* v = new VerbsCtx();
  c->verbs = v;
  v->data_plane = data_plane;
  int local_rank = 0;
  for (int p = 0; p < c->cfg.rank; p++)
    if (c->topo.same_host(c->cfg.rank, p)) local_rank++;
  v->ctx = open_active_device(local_rank, &v->port);
  if (!v->ctx) return -1;
  {
    ibv_port_attr pa{};
    if (ibv_query_port(v->ctx, v->port, &pa)) { MLOG_ERR("E1011 query_port 失败"); return -1; }
    v->roce = (pa.link_layer == IBV_LINK_LAYER_ETHERNET);
    v->mtu = pa.active_mtu;
    v->lid = v->roce ? 0 : pa.lid;
    // 设备身份 + 物理口速率(NIC 单元探测:同机同 devkey ⇒ 共享同一块 HCA)
    {
      const char* nm = ibv_get_device_name(v->ctx->device);
      uint64_t h = 1469598103934665603ull;                       // FNV-1a
      for (const char* p = nm; nm && *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ull; }
      h ^= (uint8_t)v->port; h *= 1099511628211ull;
      v->dev_key = h ? h : 1;
      double lane = 0;                                           // Gbps / lane(编码值见 IBTA)
      switch (pa.active_speed) {
        case 1: lane = 2.5; break;   case 2: lane = 5; break;
        case 4: lane = 10; break;    case 8: lane = 10.3; break;
        case 16: lane = 14.06; break; case 32: lane = 25.78; break;
        case 64: lane = 50; break;   case 128: lane = 100; break;
        default: lane = 0; break;
      }
      int width = pa.active_width == 1 ? 1 : pa.active_width == 2 ? 4
                : pa.active_width == 4 ? 8 : pa.active_width == 8 ? 12 : 0;
      v->link_gbps = lane * width;
      MLOG_INFO("NIC 单元:%s port %d key=%016llx 物理口 %.0f Gbps", nm, v->port,
                (unsigned long long)v->dev_key, v->link_gbps);
    }
  }
  // GID:RoCE v2 惯例 index 3;IB 只用 index 0(且仅当对端无 LID 时才进 GRH)
  v->gid_index = getenv("MOSAIC_GID_INDEX") ? atoi(getenv("MOSAIC_GID_INDEX"))
                                            : (v->roce ? 3 : 0);
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
    if (!c->cfg.use_gpu || !c->stage_pay_pool) {
      MLOG_ERR("E5031 verbs 数据面需要 staging(use_gpu=0 的跨机形态不支持)");
      return -1;
    }
    // 发送:注册 engine 的 host 头池 + payload staging 池(位置随 c->tx_gdr,
    // engine_start 按同一位分配;NIC 从对应 MR 读)
    v->mr_stage_hdr = ibv_reg_mr(v->pd, c->stage_hdr_pool,
                                 (size_t)c->stage_slots * HDR_STRIDE,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    v->mr_stage_pay = reg_mr_any(v->pd, c->stage_pay_pool,
                                 (size_t)c->stage_slots * (size_t)c->cfg.slice_bytes,
                                 c->tx_gdr,
                                 IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!v->mr_stage_hdr || !v->mr_stage_pay) {
      MLOG_ERR("E1017 staging 注册失败(hdr=%p pay=%p tx_gdr=%d)",
               (void*)v->mr_stage_hdr, (void*)v->mr_stage_pay, (int)c->tx_gdr);
      return -1;
    }
    // 接收:host 头槽池 + payload 槽池(位置随 c->rx_gdr;每 peer nrx 槽)
    v->rx_pay_bytes = (size_t)c->cfg.slice_bytes;
    const size_t hdr_bytes = (size_t)HDR_STRIDE * nrx * c->cfg.nranks;
    if (posix_memalign((void**)&v->rx_hdr_pool, 4096, hdr_bytes)) {
      MLOG_ERR("E1018 rx 头池分配失败"); return -1;
    }
    memset(v->rx_hdr_pool, 0, hdr_bytes);
    v->mr_rx_hdr = ibv_reg_mr(v->pd, v->rx_hdr_pool, hdr_bytes,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    const size_t pay_bytes = v->rx_pay_bytes * nrx * c->cfg.nranks;
    if (c->rx_gdr)
      v->rx_pay_pool = mosaic_gpu_alloc_aligned(pay_bytes, &v->rx_pay_raw);
    else if (cudaHostAlloc(&v->rx_pay_pool, pay_bytes, cudaHostAllocPortable) != cudaSuccess)
      v->rx_pay_pool = nullptr;
    if (!v->rx_pay_pool) {
      MLOG_ERR("E1018 rx payload 池分配失败(%zu MB,rx_gdr=%d)",
               pay_bytes >> 20, (int)c->rx_gdr); return -1;
    }
    // rx_gdr 时加 REMOTE_READ:flush QP 的 RDMA READ 要以本池为远端
    v->mr_rx_pay = reg_mr_any(v->pd, v->rx_pay_pool, pay_bytes, c->rx_gdr,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                  (c->rx_gdr ? IBV_ACCESS_REMOTE_READ : 0));
    if (!v->mr_rx_hdr || !v->mr_rx_pay) {
      MLOG_ERR("E1019 rx 池注册失败(hdr=%p pay=%p rx_gdr=%d)",
               (void*)v->mr_rx_hdr, (void*)v->mr_rx_pay, (int)c->rx_gdr);
      return -1;
    }
    // flush QP(仅 rx_gdr):自连 RC,交付前用 8B RDMA READ 把 GPU BAR 上
    // 未排水的 posted write 顶出去(见文件头注;NCCL gdrFlush 同款)
    if (c->rx_gdr) {
      if (posix_memalign((void**)&v->flush_buf, 4096, 4096) ||
          !(v->mr_flush = ibv_reg_mr(v->pd, v->flush_buf, 4096, IBV_ACCESS_LOCAL_WRITE))) {
        MLOG_ERR("E1036 flush 缓冲注册失败"); return -1;
      }
      ibv_qp_init_attr ia{};
      ia.send_cq = v->cq_send; ia.recv_cq = v->cq_send;
      ia.qp_type = IBV_QPT_RC;
      ia.cap.max_send_wr = 8; ia.cap.max_recv_wr = 1;
      ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
      ia.sq_sig_all = 1;
      v->qp_flush = ibv_create_qp(v->pd, &ia);
      ibv_gid fgid{};
      QpInfo self{};
      if (!v->qp_flush || ibv_query_gid(v->ctx, v->port, v->gid_index, &fgid)) {
        MLOG_ERR("E1036 flush QP 创建失败"); return -1;
      }
      self.mtu = (uint8_t)v->mtu;
      self.lid = v->lid;
      memcpy(self.gid, &fgid, 16);
      if (!qp_connect(v, v->qp_flush, 7, v->qp_flush->qp_num, 7, self)) {
        MLOG_ERR("E1036 flush QP 自连失败"); return -1;
      }
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
    mine.mtu = (uint8_t)v->mtu;
    mine.lid = v->lid;
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
      if (!qp_connect(v, vp.qp, mine.psn, rem.qp_num, rem.psn, rem)) return -1;
      for (int i = 0; i < nrx; i++) if (!post_recv(c, p, i)) return -1;
      // 信用反压初始授权 = 全窗 nrx(两侧由同一 rdma_rx_slots 公式导出,
      // 天然一致):cr_limit 是发送侧对 p 的额度,cr_granted 是接收侧
      // 对 p 已回流的累计基准(首条 CT_CREDIT 携带 nrx+bank,单调)。
      c->cr_limit[p] = (uint64_t)nrx;
      c->cr_granted[p] = (uint64_t)nrx;
      c->cr_w0[p] = (uint64_t)nrx;
    }
    if (!qp_connect(v, vp.qp_ctrl, mine.ctrl_psn, rem.ctrl_qp_num, rem.ctrl_psn, rem))
      return -1;
    for (int i = 0; i < VerbsCtx::NCTRL_RX; i++) if (!post_ctrl_recv(c, p, i)) return -1;
  }
  MLOG_INFO("verbs 就绪:%d peer × (%s + ctrl)QP;数据 %d 预投槽/peer"
            "(%s,xport TX=%s RX=%s)",
            nremote, data_plane ? "data" : "无 data", nrx, v->roce ? "RoCE" : "IB",
            c->tx_gdr ? "gdr" : "pin", c->rx_gdr ? "gdr" : "pin");
  return 0;
}

void verbs_teardown(Comm* c) {
  VerbsCtx* v = VC(c);
  if (!v) return;
  for (auto& p : v->peer) {
    if (p.qp) ibv_destroy_qp(p.qp);
    if (p.qp_ctrl) ibv_destroy_qp(p.qp_ctrl);
  }
  if (v->qp_flush) ibv_destroy_qp(v->qp_flush);
  if (v->mr_stage_hdr) ibv_dereg_mr(v->mr_stage_hdr);
  if (v->mr_stage_pay) ibv_dereg_mr(v->mr_stage_pay);
  if (v->mr_rx_hdr) ibv_dereg_mr(v->mr_rx_hdr);
  if (v->mr_rx_pay) ibv_dereg_mr(v->mr_rx_pay);
  if (v->mr_ctrl) ibv_dereg_mr(v->mr_ctrl);
  if (v->mr_ctrl_rx) ibv_dereg_mr(v->mr_ctrl_rx);
  if (v->mr_flush) ibv_dereg_mr(v->mr_flush);
  free(v->ctrl_pool);
  free(v->ctrl_rx_pool);
  free(v->rx_hdr_pool);
  free(v->flush_buf);
  if (v->rx_pay_pool) {
    if (c->rx_gdr) cudaFree(v->rx_pay_raw); else cudaFreeHost(v->rx_pay_pool);
  }
  if (v->cq_send) ibv_destroy_cq(v->cq_send);
  if (v->cq_recv) ibv_destroy_cq(v->cq_recv);
  if (v->pd) ibv_dealloc_pd(v->pd);
  if (v->ctx) ibv_close_device(v->ctx);
  delete v;
  c->verbs = nullptr;
}

// 发送一个 slice:双 SGE [host 头槽(此处 CPU 写)| host pinned staging payload]。
int verbs_post_slice(Comm* c, int peer, const SliceHdr& h, int stage_slot) {
  VerbsCtx* v = VC(c);
  uint8_t* hslot = c->stage_hdr_pool + (size_t)stage_slot * HDR_STRIDE;
  memcpy(hslot, &h, sizeof(SliceHdr));
  ibv_sge sge[2];
  sge[0].addr = (uint64_t)(uintptr_t)hslot;
  sge[0].length = HDR_STRIDE;
  sge[0].lkey = v->mr_stage_hdr->lkey;
  sge[1].addr = (uint64_t)(uintptr_t)((uint8_t*)c->stage_pay_pool +
                                      (size_t)stage_slot * (size_t)c->cfg.slice_bytes);
  sge[1].length = h.len;
  sge[1].lkey = v->mr_stage_pay->lkey;
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
  c->node_inflight[c->topo.host(peer)] += h.len;   // 每目的节点在途账(完成/失败时退)
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

// rx_gdr 交付排水:对 gdr_pend 当前积压发一笔 8B RDMA READ(同 HCA 自连),
// READ 完成 = 之前所有落显存的 posted write 已排干,覆盖的那批才许交付。
// flush QP 坏死(极端错误路径)时直接交付,不让接收侧永卡。
static void post_flush(Comm* c) {
  VerbsCtx* v = VC(c);
  if (v->flush_cover || v->gdr_pend.empty()) return;
  ibv_sge s{};
  s.addr = (uint64_t)(uintptr_t)v->flush_buf;
  s.length = 8;
  s.lkey = v->mr_flush->lkey;
  ibv_send_wr wr{}, *bad = nullptr;
  wr.wr_id = WRID_FLUSH;
  wr.sg_list = &s;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = (uint64_t)(uintptr_t)v->rx_pay_pool;
  wr.wr.rdma.rkey = v->mr_rx_pay->rkey;
  if (v->qp_flush && ibv_post_send(v->qp_flush, &wr, &bad) == 0) {
    v->flush_cover = v->gdr_pend.size();
    v->flush_post_ns = now_ns();
    return;
  }
  MLOG_ERR("E1037 flush READ 投递失败,该批直接交付(顺序保障降级)");
  while (!v->gdr_pend.empty()) {
    VerbsCtx::GdrPend p = v->gdr_pend.front();
    v->gdr_pend.pop_front();
    on_slice_arrived(c, p.h, p.gpay, -1, p.token);
  }
}

// 轮询完成队列。返回处理的事件数。
int verbs_poll(Comm* c) {
  VerbsCtx* v = VC(c);
  if (!v) return 0;
  int done = 0;
  ibv_wc wc[32];
  int n = ibv_poll_cq(v->cq_send, 32, wc);
  for (int i = 0; i < n; i++) {
    if (wc[i].wr_id & WRID_FLUSH) {       // flush READ 完成:交付覆盖的那批
      if (wc[i].status != IBV_WC_SUCCESS)
        MLOG_ERR("E1037 flush READ wc 失败 status=%d(%s),该批仍交付",
                 wc[i].status, ibv_wc_status_str(wc[i].status));
      size_t nfl = v->flush_cover;
      v->flush_cover = 0;
      while (nfl-- && !v->gdr_pend.empty()) {
        VerbsCtx::GdrPend p = v->gdr_pend.front();
        v->gdr_pend.pop_front();
        on_slice_arrived(c, p.h, p.gpay, -1, p.token);
        done++;
      }
      post_flush(c);                      // 还有积压就续一笔
      continue;
    }
    if (wc[i].status != IBV_WC_SUCCESS) {
      // 死法解剖要知道"对谁":数据 WR 从 host 头槽读回 (src→dst,hop),
      // 定位是哪个 peer 的排水路把这条 QP 拖死(坑 #58 归因的第一问)
      if (!(wc[i].wr_id & (WRID_CTRL | WRID_RX | WRID_CRX))) {
        SliceHdr dh;
        memcpy(&dh, c->stage_hdr_pool + (size_t)(int)wc[i].wr_id * HDR_STRIDE, sizeof(dh));
        MLOG_ERR("E1025 send wc 失败 status=%d(%s) peer=r%d (s%u→d%u hop%u len=%u)",
                 wc[i].status, ibv_wc_status_str(wc[i].status),
                 dh.path[dh.hop + 1], dh.src, dh.dst, dh.hop, dh.len);
      } else {
        MLOG_ERR("E1025 send wc 失败 status=%d(%s) ctrl_wr=%d",
                 wc[i].status, ibv_wc_status_str(wc[i].status),
                 (int)((wc[i].wr_id >> 40) & 7));
      }
      if (!(wc[i].wr_id & (WRID_CTRL | WRID_RX | WRID_CRX))) {
        // 数据 WR 失败:在途账必须退(否则 node_inflight 永久占位 = 死锁),
        // staging 槽一并回收(此前是泄漏)
        int slot = (int)wc[i].wr_id;
        SliceHdr h;
        memcpy(&h, c->stage_hdr_pool + (size_t)slot * HDR_STRIDE, sizeof(h));
        int pn = c->topo.host(h.path[h.hop + 1]);
        if (c->node_inflight[pn] >= h.len) c->node_inflight[pn] -= h.len; else c->node_inflight[pn] = 0;
        stage_release(c, slot);
        if (c->nic_tx_outstanding > 0) c->nic_tx_outstanding--;
      }
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
      {   // 每目的节点在途账:send 完成 = 对端已 ACK,离开在途
        int pn = c->topo.host(h.path[h.hop + 1]);
        if (c->node_inflight[pn] >= h.len) c->node_inflight[pn] -= h.len; else c->node_inflight[pn] = 0;
      }
      det_feed(c, 0, h.len);
      src_wave_ack(c, h);                         // src 相波推进的 ACK 计数
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
          ch.type >= 1 && ch.type <= CT_MAX) {
        on_ctrl_msg(c, ch.rank, ch, slot + sizeof(CtrlHdr));
      } else {
        MLOG_ERR("E1026 ctrl 报文头不识别(r%d)", peer);
      }
      post_ctrl_recv(c, peer, idx);
      done++;
      continue;
    }
    // 数据 slice:头在 host 槽(CPU 直读),payload 在 RX 槽池(消费者核交付)
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
    void* gpay = (uint8_t*)v->rx_pay_pool +
                 ((size_t)peer * v->peer[peer].nrx + idx) * v->rx_pay_bytes;
    // 中转但中转池空:扣着槽交给 wait_relay(on_slice_arrived 内部处理),
    // 槽在交付拷贝完成后由 verbs_repost 归还。
    // rx_gdr:CQE ≠ payload 已落显存,先入 pend 等 flush READ(见 post_flush)
    if (c->rx_gdr) {
      v->gdr_pend.push_back({h, gpay, (peer << 20) | idx});
    } else {
      on_slice_arrived(c, h, gpay, -1, (peer << 20) | idx);
    }
    done++;
  }
  if (c->rx_gdr) {
    // 活性兜底:GPU BAR 排水停滞的平台(Raven 实测 pin,gdr 挂死)会把 flush
    // READ 一起卡死 —— 超过 500ms 未完成就告警并交付该批(此时 payload 大概率
    // 早已落定,只是"确认"被卡;正确性仍由端到端 bit-exact 校验兜底)。
    // 该路径反复触发 = 平台不适合 rx_gdr,profile 的双向腿本就会把它排到最后。
    if (v->flush_cover && now_ns() - v->flush_post_ns > 500000000ull) {
      MLOG_WARN("E1039 flush READ %zu 条超时 500ms,越过确认直接交付(平台排水停滞)",
                v->flush_cover);
      size_t nfl = v->flush_cover;
      v->flush_cover = 0;               // 在途 READ 的迟到完成会被 WRID_FLUSH 分支吸收(cover=0 交付 0 条)
      while (nfl-- && !v->gdr_pend.empty()) {
        VerbsCtx::GdrPend p = v->gdr_pend.front();
        v->gdr_pend.pop_front();
        on_slice_arrived(c, p.h, p.gpay, -1, p.token);
        done++;
      }
    }
    post_flush(c);
  }
  return done;
}

// 交付拷贝(D2D)完成后补投这个接收槽。**不补投就是背压**:发送端撞 RNR
// 自动退避 —— NIC 下行满了发送方自然慢下来。
void verbs_repost(Comm* c, int token) {
  if (token < 0) return;
  post_recv(c, token >> 20, token & 0xFFFFF);
}

// NIC 物理身份(nic_unit_exchange 用):false = 无 verbs 面(TCP/离线形态)
bool verbs_nic_ident(Comm* c, uint64_t* key, double* gbps) {
  VerbsCtx* v = VC(c);
  if (!v || !v->ctx) return false;
  *key = v->dev_key;
  *gbps = v->link_gbps;
  return true;
}

}  // namespace mosaic
