/* internal.h —— libmosaic 内部契约:所有跨文件共享的类型都在这里。
 *
 * 分层:
 *   L0 transport  = 机内 CUDA IPC 环(net.cc)/ 跨机 RDMA verbs(verbs.cc,
 *                   双侧 host-bounce)/ TCP(bootstrap + 控制面回退)
 *   L1 engine     = 事件循环(slice 状态机 + kdesc 消费者核 / CUDA 异步拷贝)
 *   L2 core       = Topo/Corridor + Ledger + DstCtrl 波引擎(纯 CPU,不碰 CUDA)
 *   L3 api        = mosaic.h
 *
 * 线程模型:app 线程(调用者)+ 1 个 event 线程。协议决策、CUDA 提交、网络
 * 收发全在 event 线程里做 —— 单线程事件循环 = 无锁 = 决策可复现。
 */
#ifndef MOSAIC_INTERNAL_H
#define MOSAIC_INTERNAL_H

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mosaic/mosaic.h"

namespace mosaic {

// ─────────────────────────────────────────────────────────── 常量
static constexpr int MAX_RANKS = 64;
static constexpr int NCORR = 6;          // DIRECT, RX, TX, SPREAD lane0..2
static constexpr int MAX_HOPS = 3;       // ≤3 跳(报告 §4.1 的候选空间上界)
static constexpr uint32_t SLICE_MAGIC = 0x4D534C43u;  // 'MSLC'
// 线格式:[HDR_STRIDE 字节的 SliceHdr 区 | payload]。RDMA SEND 用双 SGE
// 一次 WR 发「头+payload」。
static constexpr int HDR_STRIDE = 64;

// 资源账本的种类。RES_SPINE 只对跨 rail 的跨机 hop 计费。
enum ResKind { RES_NIC_UP = 0, RES_NIC_DN, RES_NV_UP, RES_NV_DN, RES_SPINE_UP, RES_KINDS };

enum CorrId { CORR_DIRECT = 0, CORR_RX = 1, CORR_TX = 2, CORR_SPREAD0 = 3 };

// ─────────────────────────────────────────────────────────── 线格式
#pragma pack(push, 1)
struct SliceHdr {
  uint32_t magic;
  uint16_t src, dst;      // 逻辑 pair 的两端(不是本 hop 的两端)
  uint64_t off;           // 本 slice 在 (src→dst) 消息里的字节偏移
  uint32_t len;
  uint16_t wave;
  uint8_t corr, hop, nhops, flags;
  uint8_t path[4];        // path[0]=src … path[nhops]=dst
  uint32_t crc;           // 槽使用代数 tag(线格式布局位,偏移 30 由断言钉死)
  // **代际号 = coll_id 低 16 位**(坑 #48 的根治)。同步档里"当前 collective"
  // 天然唯一,所以老协议没带它;异步档一旦允许多代同时在网上,接收侧就再也
  // 分不清"上一代的迟到片"与"下一代的早到片" —— 而终跳的落点
  // `recvbuf + rdispls[src] + off` 用的是**收方当前**那一代的缓冲。
  // 结论:代际必须**随数据走**,不能靠"大家都在同一代"这个隐含全局同步。
  // 白送的字段:HDR_STRIDE=64 而头只有 36 字节,本就整块传输,零额外开销。
  uint16_t coll;
};
#pragma pack(pop)
static_assert(sizeof(SliceHdr) == 36, "SliceHdr 必须是紧凑的 36 字节");
static_assert(sizeof(SliceHdr) <= HDR_STRIDE, "SliceHdr 必须放得进头区");
// SliceHdr 是线格式:字段偏移进过设备核硬编码 —— 中间插字段会静默挪动
// 后续偏移,交付永久卡死且不报错。用断言钉死布局。
static_assert(offsetof(SliceHdr, crc) == 30, "SliceHdr 线格式布局不可挪动");

// 控制面报文类型。通道 A = 遥测(有界陈旧),通道 B = 策略广播(必须可靠,
// 因为它同时是**执行指令** —— 跨机走 verbs RC SEND,可靠性与 TCP 等强)。
enum CtrlType : uint8_t {
  CT_HELLO = 1, CT_DEMAND = 2, CT_BARRIER = 3, CT_POLICY = 4,
  CT_TELEM = 5, CT_STATS = 6, CT_FIN = 7, CT_AG = 8,
  // 接收端信用回流(File/design-credit-backpressure-0821.md):body = uint64
  // 累计授权 slice 数(绝对值,收端取 max ⇒ verbs/TCP 混跑与乱序安全)
  CT_CREDIT = 9,
  // src 驱动 dispatch(File/design-global-view-src-driven-0823.md 任务二):
  // src 波的策略广播。**纯账本视图报文,收方永不执行**(数据由发布者自己
  // 发出);body 仍是 PolicyEnt[],但 ent.src 字段承载 **dst 端点**,
  // 链 = chain_of(发布者, ent.src, ent.corr)。aux 编码与 CT_POLICY 同
  // (高 8 位 coll_id mod 256,低 8 位 wave)。
  CT_POLICY_SRC = 10,
};
static constexpr uint8_t CT_MAX = CT_POLICY_SRC;   // ctrl 报文类型校验上界

#pragma pack(push, 1)
struct CtrlHdr {
  uint32_t len;      // 含本头的总字节数
  uint8_t type;
  uint8_t rank;      // 发送者
  uint16_t aux;      // type 相关(barrier epoch / policy wave / …)
};
struct TelemMsg {    // 通道 A:每 sync_us 一发。基础 60B;fabric 账贡献(per-rail
  // 上/下行,G≤8)再加 128B —— 超出报告 §11.2 的 64B 口径,原因:模拟器的
  // fabric discharge 由交换机自有计数(INT/HPCC 捎带,近零成本)承担,真机模拟
  // 没有交换机,只能由端侧摊派 per-rail 贡献。字节数进 chA_bytes 计量,
  // Raven 实现同此口径(HDR 交换机同样拿不到 per-flow 计数)。
  uint64_t seq;
  uint64_t disch[RES_KINDS];
  float nic_up_gbps, nic_dn_gbps;  // 检测器当前生效速率(属主实测;=额定即未锁定)
  uint8_t det_state;               // bit0 = NIC↑ 锁定, bit1 = NIC↓ 锁定
  uint8_t _pad[3];
  uint64_t fab_up[8], fab_dn[8];   // 本 rank 对各 rail fabric 账的 discharge 贡献
};
struct PolicyEnt {   // 通道 B 的一条:某 src 在某 corridor 上发多少
  uint16_t src;
  uint8_t corr;
  uint8_t _pad;
  uint32_t len;
  uint64_t off;
};
// P2 定向遥测的最短形态:只含检测器状态(quorum 语义要求 det 位全员可见)。
// 接收端按 CtrlHdr.len 区分三种 CT_TELEM 形态:
//   sizeof(TelemDet)=20 → 仅 det;60(TelemMsg 去 fab 前缀)→ disch+det;
//   sizeof(TelemMsg)=188 → 全量(含 fabric 贡献)。
struct TelemDet {
  uint64_t seq;
  float nic_up_gbps, nic_dn_gbps;
  uint8_t det_state;
  uint8_t _pad[3];
};
#pragma pack(pop)
static constexpr uint32_t TELEM_NOFAB_BYTES = 60;   // = offsetof(TelemMsg, fab_up)
// CT_DEMAND 捎带的读集位图:bit0-4 = ResKind,bit5 = 需要 fabric 贡献
static constexpr uint8_t INTENT_FAB = 0x20;
static constexpr uint8_t INTENT_FULL = 0xFF;

// ─────────────────────────────────────────────────────────── 工具
uint64_t now_ns();
void mlog(int level, const char* fmt, ...);   // 0=err 1=warn 2=info 3=debug
extern int g_log_level;
extern int g_rank_for_log;

// 二进制 trace ring(Imp_Plan/06 §5.2:Raven 不可交互调试的现场记录)。
// MOSAIC_TRACE=1 时启用:mmap 文件上的 32B 定长事件环(满即覆盖,保最近窗口;
// mmap 由内核落盘,进程崩溃也留现场)。后处理:tools/trace2csv.py。
enum TraceEv : uint16_t {
  TE_WAVE_EMIT = 1,   // a=wave, b=bytes, c=plan_us
  TE_SLICE_POST = 2,  // a=(src<<16|dst), b=off, c=len<<8|hop
  TE_SLICE_DELIV = 3, // a=(src<<16|dst), b=off, c=len<<8|hop
  TE_CHA_PUB = 4,     // a=seq
  TE_CHB_PUB = 5,     // a=wave, b=bytes
  TE_STAGGER = 6,     // a=wave, b=wait_us
  TE_DET = 7,         // a=which(0↑1↓), b=locked, c=rate_mbps
  TE_DUMP = 8,        // 状态转储标记
  // ── 单片生命周期插桩(local-0819;全部在 MTRACE 门控内,默认路径逐字节不变)
  TE_SLICE_TX = 9,     // src:D2H 完成、RDMA WR 实投  a=(src<<16|dst), b=off, c=len<<8|hop
  TE_SLICE_RECV = 10,  // dst:recv CQ 命中(片到达)  同上
  TE_TXDONE = 11,      // src:send CQ 完成、staging 槽回收  同上
  TE_REPOST = 12,      // dst:补投 recv WR = 放 credit  同上
  TE_COLL_BEGIN = 13,  // a=coll_id, b=total_send, c=total_recv_expect
  TE_COLL_END = 14,    // a=coll_id, b=total_recv
  TE_LOOP = 15,        // 事件循环一圈  b=上一圈耗时 ns(仅 MOSAIC_TRACE>=2)
  TE_KDESC_PUSH = 16,  // dst:交付描述符入环  a=(src<<16|dst), b=off, c=len<<8|hop
  TE_KDESC_SAMP = 17,  // 消费者核 in-vivo 采样  b=head(pushed), c=done(仅 TRACE 开)
};
#pragma pack(push, 1)
struct TraceRec { uint64_t t_ns; uint16_t ev; uint16_t rank; uint32_t a; uint64_t b, c; };
#pragma pack(pop)
static_assert(sizeof(TraceRec) == 32, "TraceRec 必须 32 字节");
void mtrace_init(int rank);
void mtrace(uint16_t ev, uint32_t a, uint64_t b, uint64_t c);
extern bool g_trace_on;
extern bool g_trace_loop;   // MOSAIC_TRACE>=2:逐圈事件循环打点(量很大,单独开)
#define MTRACE(ev, a, b, c) do { if (::mosaic::g_trace_on) ::mosaic::mtrace(ev, a, b, c); } while (0)

// 状态转储(watchdog / SIGTERM;06 §5.3「卡死必留尸检」)
struct Comm;
void dump_state(Comm* c, const char* why);
// 崩溃信号 handler(SEGV/BUS/FPE/ILL)+ alternate stack。**每线程**都要装一次:
// sigaltstack 是按线程记账的,不装 = 栈溢出型崩溃一个字都留不下(坑 #44)
void mosaic_install_crash_handlers_this_thread();

#define MLOG_ERR(...)  ::mosaic::mlog(0, __VA_ARGS__)
#define MLOG_WARN(...) ::mosaic::mlog(1, __VA_ARGS__)
#define MLOG_INFO(...) ::mosaic::mlog(2, __VA_ARGS__)
#define MLOG_DBG(...)  do { if (::mosaic::g_log_level >= 3) ::mosaic::mlog(3, __VA_ARGS__); } while (0)

// ─────────────────────────────────────────────────────────── L2:拓扑与走廊
struct Chain {          // 一条具体的链(某 corridor 在某 pair 上的实例)
  uint8_t nhops;
  uint8_t path[MAX_HOPS + 1];
};

struct Topo {
  int nranks = 0, G = 0, H = 0;
  // MOSAIC_LP_TABLE(T5,LP 落点表):lp_j[r][rd] = 权重降序的中转轨表
  // (tools/mosaic_lp_intra.py 的 jdist 输出;表随 Impl 树 rpush 同步)。
  // 空 = 未加载 ⇒ 走原轮转公式,行为不变。
  std::vector<std::vector<std::vector<uint8_t>>> lp_j;
  int rank = 0;
  // MOSAIC_NET_MODE=rail(网络虚拟化铁律,根 CLAUDE.md):把树叠成 rail ⇒
  // **跨 rail 的跨机对不存在直连通道**,必须先在节点内换轨。0 = origin(旧行为:
  // 跨 rail 允许 DIRECT,只多记一笔 spine 账 —— 那是"计费"不是"禁止")。
  int net_rail = 0;
  int host(int r) const { return r / G; }
  int rail(int r) const { return r % G; }
  bool same_host(int a, int b) const { return host(a) == host(b); }
  // NIC 物理单元:nic_leader[r] = 与 r 共享同一块物理 HCA 的同机 rank 里
  // local_rank 最小者(共享只可能发生在同机)。由 net_setup 尾的
  // nic_unit_exchange 运行时探测填写(SHM 同机共识 + 跨机 pairwise 交换);
  // 默认恒等 = 每 rank 私有 NIC(本机 world=2、离线回放不变)。
  // NIC 账一律记在 leader 的资源位上 —— 节点共享 HCA(如 Raven 每节点
  // 1×HDR200 喂 4 GPU)不再被注水当成 4 根独立管子错误分流
  // (log_raven.md 2026-08-21「资源单元」条)。
  uint8_t nic_leader[MAX_RANKS] = {0};
  int nic_res_up(int r) const { return res(nic_leader[r], RES_NIC_UP); }
  int nic_res_dn(int r) const { return res(nic_leader[r], RES_NIC_DN); }

  // pair (s,d) 上第 c 条 corridor 的链;返回 false = 该 corridor 对本 pair 不可用
  bool chain_of(int s, int d, int c, Chain* out) const;
  // 该 pair 允许的 corridor 列表(受 cfg.corridors 限制)
  int corridors_of(int s, int d, int c_limit, int* out) const;
  // 资源空间 = 每 rank 5 本边缘账 + 每 rail 的 fabric 聚合账(上/下行各一,
  // 对应模拟器 rail 布线 + spineagg 档的 leaf 段账,mosaic_schedule.h
  // lg_res_of_hop;2026-08-14 决策对拍补齐,见 log.md 坑 #18)
  int nres() const { return nranks * RES_KINDS + 2 * G; }
  int res(int r, int kind) const { return r * RES_KINDS + kind; }
  int res_fab_up(int rail) const { return nranks * RES_KINDS + rail; }
  int res_fab_dn(int rail) const { return nranks * RES_KINDS + G + rail; }
  // 一条 hop(a→b)占用哪些资源;写入 out,返回条数
  int hop_res(int a, int b, int* out) const;
  // 一条链占用的全部资源签名
  int chain_res(const Chain& ch, int* out) const;
};

// ─────────────────────────────────────────────────────────── L2:账本
struct Ledger {
  std::vector<uint64_t> charge;      // 已分配(自己的决策 + 他人通道 B)
  std::vector<uint64_t> discharge;   // 已送达(自己的计数 + 他人通道 A)
  std::vector<double> cap_Bps;       // 每资源的额定字节/秒
  // 他人 charges 按发布者分开存,这样收到某 dst 的新一份就整体替换旧的
  // (通道 B 的"只存最新一份"语义,单写者 ⇒ 不会分叉)
  std::vector<std::vector<uint64_t>> peer_charge;  // [publisher][res]
  // NIC 单元聚合:charge 都记 leader 位,但 discharge 是每 rank 按自己的
  // Bucket 计数**绝对值赋值**的(engine.cc:449/1365)—— 共享 HCA 的单元里,
  // leader 位的在途要把成员位的 discharge 也扣掉。disch_agg[leader_res] =
  // 成员的 res 下标表(nic_unit_exchange 后由 engine_start 填;空 = 旧行为)
  std::vector<std::vector<int>> disch_agg;

  void init(int nres, int nranks);
  uint64_t outstanding(int res) const;
  double proj_us(int res) const;                     // 投影完成时间(µs)
  void charge_chain(const int* r, int n, uint64_t bytes);
  void set_peer_charge(int publisher, const uint64_t* v, int nres);
  void clear_epoch();
};

// ─────────────────────────────────────────────────────────── L2:NIC 域异常检测器
// 逐行对照模拟器 mosaic_schedule.h:1961-2016(x41/x44 定稿语义):
//   只测 NIC 端口(NV/fabric 的忙时吞吐会内生性下垂,信之即 censoring 螺旋);
//   忙时聚合吞吐(ratio-of-sums,500µs 忙时折算一次);<0.45×额定连续 3 折才
//   锁定(锁定后 0.5/0.5 平滑,地板 额定/16);quorum≤2:全域已锁 ≥2 时不再新锁
//   (相关性塌陷 = 拥塞态,冻结而非降速);恢复即回额定。
struct Detector {
  double bacc = 0, tacc_ns = 0;    // 忙时累计(字节,ns)
  uint64_t prev_ns = 0;
  bool busy_after = false;
  int streak = 0;
  double nom_gbps = 0, rate_gbps = 0;
  bool locked = false;
};

// ─────────────────────────────────────────────────────────── L1:资源计数器
// 真机为准:额定值(nv/nic/spine)只喂决策层的代价模型,数据面永不限速。
// passed 计数 = 该资源的 discharge,喂通道 A 遥测与账本(决策层语义)。
struct Bucket {
  uint64_t passed = 0;       // 累计通过字节(= 该资源的 discharge 计数)
  void init() { passed = 0; }
  void take(uint64_t n) { passed += n; }
};

// ─────────────────────────────────────────────────────────── L0:传输
struct ShmRing;   // net.cc
struct Conn;      // net.cc:一条 TCP 连接(数据或控制)

// ─────────────────────────────────────────────────────────── L1:slice 状态机
enum SliceStage : uint8_t { ST_D2H = 0, ST_WIRE, ST_H2D };

struct PendSlice {          // 一个在途的 slice(出向或入向)
  SliceHdr h;
  void* stage = nullptr;    // host staging(pinned)或 SHM slot payload
  void* gpu_src = nullptr;  // D2H 的源(sendbuf 或 relay 池)
  void* gpu_dst = nullptr;  // H2D 的目的(recvbuf 或 relay 池)
  int relay_slot = -1;      // 占用的 relay 池槽位,-1 = 未占用
  int shm_ring = -1;        // 走 SHM 时的环号
  int stage_slot = -1;      // staging 池槽位
  void* ev = nullptr;       // cudaEvent_t
  int peer = -1;            // 本 hop 的对端 rank
  bool via_nic = false;
};

// ─────────────────────────────────────────────────────────── L2:每目的地控制环
struct DstCtrl {
  bool active = false;
  uint64_t remaining[MAX_RANKS] = {0};
  uint64_t cursor[MAX_RANKS] = {0};
  int wave_idx = 0;
  uint64_t wave_bytes = 0, wave_delivered = 0;
  uint64_t total_expect = 0, total_recv = 0;
  uint64_t recv_by_src[MAX_RANKS] = {0};   // 诊断:逐 src 交付账目(查重复/丢失)
  bool pending_trigger = false;
  uint64_t trigger_ns = 0;      // 错峰:到点才决策
  uint64_t trigger_set_ns = 0;  // 进入槽位等待的时刻(用于计量错峰代价)
  uint64_t wave_start_ns = 0;   // 本波发出的时刻
  uint64_t wave_deadline_ns = 0;// 兜底触发:慢走廊的尾部不该饿死流水线
};

// ─────────────────────────────────────────────────────────── L2:每源控制环
// src 驱动 dispatch(任务二,File/design-global-view-src-driven-0823.md):
// dispatch 相每个 src 发出的总量相等 ⇒ 决策权给"总量相等"的那一侧 ——
// src 对自己的行注水发射,决策者之间因规模相同天然公平,无需协调。
// 「波」在此相 = **src 的流水单元**:上一波跨机字节的 RC ACK(send CQ 完成
// = 对端已收)过 MOSAIC_WAVE_THRESH 即发下一波。combine 相仍走 DstCtrl。
struct SrcCtrl {
  bool active = false;
  uint64_t remaining[MAX_RANKS] = {0};   // [dst] 本 coll 我的行还没发射的字节
  uint64_t cursor[MAX_RANKS] = {0};      // [dst] 行内连续区间游标
  int wave_idx = 0;
  uint64_t wave_nic_bytes = 0;   // 上一波 hop0 为跨机的字节(ack 门的分母;
                                 // NV hop0 无 ACK 事件,不进分子分母)
  uint64_t wave_acked = 0;       // 其中已获 RC ACK 的字节(src_wave_ack 计)
  bool pending_trigger = false;
  uint64_t trigger_ns = 0;       // 错峰:到点才决策(与 dst 环同一套槽位)
  uint64_t trigger_set_ns = 0;
  uint64_t wave_start_ns = 0;
  uint64_t wave_deadline_ns = 0; // 兜底:按本 rank 出向(NV↑+NIC↑)额定估计
};

struct Stats : mosaicStats_t {};

// ─────────────────────────────────────────────────────────── Comm
struct Comm {
  mosaicConfig_t cfg{};
  Topo topo;
  Ledger ledger;
  std::string err;

  // ---- 传输 ----
  std::vector<Conn*> conn;         // [peer] 控制连接(全 mesh)
  std::vector<Conn*> dconn;        // [peer] 数据连接(仅跨机)
  std::vector<ShmRing*> tx_ring;   // [peer] 本机内发送环
  std::vector<ShmRing*> rx_ring;   // [peer] 本机内接收环
  int listen_fd = -1;
  int epfd = -1;
  void* verbs = nullptr;           // verbs.cc 的上下文(RoCE 数据面);null = TCP
  bool use_verbs = false;
  // NIC 单元探测(nic_unit_exchange 填):每 rank 所用物理口的速率(Gbps)。
  // 账本 cap 只取 leader 位;非 leader 的 NIC 资源位 cap=0(账全记 leader)
  double nic_unit_gbps[MAX_RANKS] = {0};
  void* erpc = nullptr;            // erpc_ctrl.cc 的上下文(控制面兼容档);null = 未启用

  // ---- CUDA ----
  void* stream_d2h = nullptr;
  void* stream_h2d = nullptr;

  std::vector<void*> ev_pool;
  void* relay_pool = nullptr;      // GPU 上的中转缓冲
  int relay_slots = 0;
  std::vector<int> relay_free;
  // verbs 数据面 staging(2026-08-19 GDR 重构):payload 在 GPU 池(dmabuf 注册,
  // RDMA 直读写),64B 头在 host 池(CPU 直写、双 SGE 同一 WR 发出)。
  uint8_t* stage_hdr_pool = nullptr;   // host pinned,NST × HDR_STRIDE
  // TX payload staging:host pinned(消费者核 D2H 组装,NIC 从 host MR 读)。
  // 不用 GPU 池:NIC 对显存的 GDR **读**要过 PCIe host-bridge P2P,Raven 实测
  // 被钉在 ~121 Gbps(host 读 197 = 满线速);GDR **写**无损,RX 池仍在 GPU。
  void* stage_pay_pool = nullptr;      // NST × slice_bytes
  int stage_slots = 0;
  std::vector<int> stage_free;

  // ---- 限速器 ----
  Bucket buck[RES_KINDS];

  // ---- 检测器与协议门 ----
  Detector det[2];                 // [0]=NIC↑ [1]=NIC↓(只测 NIC 域)
  std::vector<double> nominal_Bps; // 每资源额定(检测器恢复与 telemetry 覆盖的基线)
  std::vector<uint8_t> peer_det;   // [rank] 最新 det_state(quorum 全域计数用)
  int nic_tx_outstanding = 0;      // 在途 NIC 发送数(检测器 busy 判定)
  // 每目的**节点**在途字节(发送端流控,2026-08-21):RC send 完成 = 对端已
  // ACK,是精确的在途信号。动机:QP 级联死(E1025)= RNR 风暴,静态"每波
  // 体积"不是死亡预测量(29429499 α2 W5=196MB/波死 vs α1 W2=293MB/波稳),
  // 只有源头限流能移除悬崖 —— 悬崖消失后 W 在线探索才全域安全。
  uint64_t node_inflight[MAX_RANKS] = {0};   // 按 host id 索引
  // ── 接收端信用反压(2026-08-21,File/design-credit-backpressure-0821.md):
  // credit 只由接收端 rx_retire(交付完成、槽补投)铸造 ⇒ 稳态进水率 =
  // 真实排水率,RNR/链路信用级联(坑 #58 的 E1025 悬崖)构造性消失。
  // 单位 = slice(一片占一个预投 recv 槽)。累计计数器协议,全部只在
  // event 线程读写(单线程事件循环,无锁)。
  uint64_t cr_sent[MAX_RANKS] = {0};        // 发送侧:已派发给 peer 的 slice 累计
  uint64_t cr_limit[MAX_RANKS] = {0};       // 发送侧:收到的授权累计(init=nrx,verbs_setup)
  uint64_t cr_base_sent[MAX_RANKS] = {0};   // 本 coll 起点快照(coll_begin)
  uint64_t cr_base_limit[MAX_RANKS] = {0};
  // 本 coll 对 peer 的初始突发额度(slice):uniform=nrx(全窗,行为不变);
  // 热目的列 = MOSAIC_CREDIT_HOT_W0(闭环只管稳态,初始突发要靠 demand
  // 矩阵预清 —— NCCL/EQDS 做不到的点,设计 §3.4)
  uint64_t cr_w0[MAX_RANKS] = {0};
  uint64_t cr_bank[MAX_RANKS] = {0};        // 接收侧:已 retire 未回流的 credit
  uint64_t cr_granted[MAX_RANKS] = {0};     // 接收侧:已回流累计(init=nrx)

  int pc_k = 0;                    // 本次 collective 生效的 pair_color K
  // ── W 在线自适应(2026-08-21 设计 §1):按 demand 形状签名记忆各 W 档的
  // completion EMA,向下探测、劣化回退。全部本地观测(G.2 合规);
  // 探索下界 = 1(悬崖保护在传输层信用反压),上界 = cfg.waves。
  struct WaveArm {
    uint64_t key = 0;            // 形状签名(列和 log2 分桶排序 hash)
    int cur_w = 0;               // 当前档
    int best_w = 0;
    double best_ms = 0;          // 已知最优档的 EMA
    double ema_ms = 0;           // 当前档 EMA(α=0.3)
    int samples = 0;             // 当前档样本数
    uint16_t bad_mask = 0;       // 实测劣化的 W 档位图(在线绕开共振档)
    uint64_t last_ns = 0;        // LRU
  };
  static constexpr int NWARM = 8;
  WaveArm warm[NWARM];
  WaveArm* warm_cur = nullptr;   // 本 coll 命中的 arm(finish 时喂 completion)
  int warm_wsafe = 1;            // 探索下界(恒 1;finish 时探测用)
  int pace_hot = -1;               // PACE:预测 straggler;-1 = 门未开
  // 发送序的热度表(coll_begin 由 demand 列和填充):dst 入向总字节。
  // hot-pair-first(方案 B):同 (wave,hop,PC) 结构序内,热 dst 的片先发 ——
  // makespan 由最热的边定,让它的入向尽早聚齐(尺寸冷热不均时收益)。
  std::vector<uint8_t> pace_cold;  // [rank] 冷 dst 标记
  std::vector<int> peer_wave;      // [rank] 该 dst 最新已发波号+1(来自通道 B)
  std::vector<uint8_t> peer_fin;   // [rank] 该 dst 本次 collective 已收满
  uint64_t rng = 0;                // 注入用确定性 RNG(种子=rank)
  double drop_a = 0;               // MOSAIC_DROP_A:通道 A 人为丢包率
  bool det_dn_hint = false;        // 检测器 DN busy 提示(verbs_poll 每事件设置)
  // fabric 账的本 rank discharge 贡献(收到跨轨 hop 时:+fab_up[rail(发送方)]
  // 与 +fab_dn[rail(me)];与模拟器"hop 交付时全资源 discharge"同点)
  uint64_t fab_own[2][8] = {{0}};  // [0]=up [1]=dn,按 rail
  std::vector<uint64_t> peer_fab;  // [rank][2][8] 平铺:各 rank 最新发布的贡献

  // ---- 效率红线测量(03 §7)----
  double plan_hist[256]; int plan_hist_n = 0;      // 注水耗时环形样本
  double tick_hist[256]; int tick_hist_n = 0;      // 通道 A 节拍抖动样本(µs)

  // ---- 调试/对拍 ----
  std::vector<PolicyEnt> dbg_last_ents;            // 最近一波的完整分配(对拍读)

  // ---- 协议状态 ----
  DstCtrl dst;                     // 本 rank 作为"目的地"的控制环
  SrcCtrl srcc;                    // 本 rank 作为"源"的控制环(src 驱动 dispatch)
  // 本 coll 的调用相位(L4 按调用位置填,经 mosaicAlltoallvEx 传入):
  // 0 = 未知(dst 驱动,旧行为);1 = dispatch;2 = combine。
  // **不复用 demand_mode**:那是可被 MOSAIC_DEMAND_HINT=0 关掉的纯优化提示,
  // 调度方向不许与之耦合。契约:同一 collective 全体 rank 的 phase 必须一致
  // (与 splits 一致性同级的调用方责任)。
  int coll_phase = 0;
  bool src_mode = false;           // 本 coll 决策者 = src(phase=1 × MOSAIC_SRC_DRIVEN)
  // [rank] 本 coll 该 peer 的 demand 行已到 = **peer 已提交本 coll** ⇒ 它的
  // recv 缓冲已切换。src 驱动的发送门:不变量「数据到达时对端已在本代」在
  // dst 驱动下靠"dst 先发策略"结构性成立,src 驱动改由此门承接 —— 对某 dst
  // 的任何字节只在收到它本 coll 的行之后才允许注水/发射(逐 dst 放行,
  // 不是等满 n-1 行,SSP 的免等启动只对迟到行的那几个 dst 局部退化)。
  std::vector<uint8_t> peer_row_seen;
  std::vector<uint64_t> demand;    // [n*n] 本次 collective 的全局矩阵(字节)
  uint64_t telem_next_ns = 0, telem_seq = 0;
  std::vector<uint64_t> peer_disch_seq;

  // ---- S2 自动选档(MOSAIC_AUTO_TIER=1;sota_plan 2026-08-16)----
  // coll_begin 按 demand 形态选本次生效的 W/C(0 = 用 cfg;只向下收,
  // 保证 compute_intent 的读集超集性质不破)。分布式视图的正统延伸:
  // 全局矩阵已在手,热点检测 → 挑走廊族/波数。
  int eff_waves = 0, eff_corridors = 0;
  bool tier_blind = false;   // S2v2:本 collective 决策盲档(非热点小报文)

  // ---- 2026-08-15 优化批(O0-O5,全部 env 可关;见 File/log_raven.md §6)----
  // O1 异步提交:app 线程在调用方流上录"输入就绪"事件,event 线程让内部拷贝流
  // WaitEvent 它 —— CPU 协议与调用方流上的尾部计算重叠,GPU 依赖交给 device 侧。
  void* input_ev = nullptr;              // 持久 cudaEvent(app 线程懒创建)
  std::atomic<int> input_ev_pending{0};  // 1 = 本次提交带事件,coll_begin 消费
  // O2 demand hint:本次提交的模式(0/1/2)与行字节;上次 collective 的全局矩阵
  int demand_mode = 0;
  uint64_t row_bytes_cur = 0, row_bytes_prev = 0;
  std::vector<uint64_t> demand_prev;     // [n*n] 上次矩阵(coll_try_finish 时快照)
  bool demand_prev_valid = false;        // 有矩阵可供 PACE/PC(SSP 档,允许残缺)
  bool demand_prev_exact = false;        // 上次行收齐,矩阵精确(hint 推导的前提)
  // O7 SSP demand(MOSAIC_DEMAND_SSP;算法级修改,用户授权 2026-08-15):
  // dispatch 不等 allgather —— 注水只读本地列(rcounts),PACE/PC 用上次矩阵
  // (有界陈旧,与通道 A 同哲学);demand 行降级为后台广播,到了就刷新。
  bool ssp_active = false;               // 本次 collective 走了 SSP 启动
  bool pace_use_prev = false;            // PACE/PC 数据源 = demand_prev
  // P2 定向遥测:各 dst 经 CT_DEMAND 捎带的"对我的读集"(bit0-4=ResKind,
  // bit5=fab)。初值 INTENT_FULL(意图未到前保守全量发);最新值胜出。
  std::vector<uint8_t> peer_intent;
  // 走廊承诺位(中转义务门,log_raven.md 2026-08-21):每个 dst 在 CT_DEMAND 里
  // 捎带 1 字节 —— 本 coll 它是否可能计划非直发路径(=1 才可能产生中转义务)。
  // coll_try_finish 只等承诺位=1 的 dst 的 FIN;全 0(如共享 HCA 拓扑上的
  // uniform)= 完成判据与直发等价,零额外耦合。=0 的 dst 必须兑现承诺
  // (emit_wave 强制 c_limit=1)。
  std::vector<uint8_t> peer_corr_intent;
  uint8_t my_corr_intent = 0;
  std::vector<uint64_t> demand_check;    // verify 档:推导矩阵暂存,收齐后对拍
  bool demand_check_on = false;
  // O0 相位时戳(MOSAIC_PHASE=1 时每 collective 打一行)
  uint64_t ph_submit_ns = 0, ph_begin_ns = 0, ph_first_post_ns = 0;
  uint64_t ph_notice_ns = 0;   // event 线程察觉提交(demand 块入口)
  uint64_t ph_bcast_ns = 0;    // demand 行 ctrl_bcast 返回时刻(串行发送段尾)
  uint64_t ph_submit_snap = 0, ph_notice_snap = 0,  // begin 时快照(打印用,防下笔提交竞态)
           ph_bcast_snap = 0;
  // O0b evloop 分段计时(仅 MOSAIC_PHASE=1;归因"察觉延迟卡在哪个泵")。
  // 按 g_ev_phase 编号记单次最大耗时;demand 块入口快照+清零,PHASE 行报 argmax
  uint64_t ev_seg_max[16] = {0};
  uint64_t ev_seg_snap[16] = {0};
  // kdesc 零流数据面能力(消费者核资源就绪才可用)
  bool zs_cap = false;
  // 流序完成(2026-08-19,一代在飞):caller 提供流时,完成 = caller 流上的
  // 有界自旋门核(等 done_flag ≥ coll_id),CPU 提交后即返回;下一次提交先
  // 隐式收尾上一笔(恒只有一代在飞,坑 #41-#48 的多代问题结构上不存在)。
  // 门未决窗口内引擎**零 CUDA 调用**(Result/local-0819-gate-probe 实测:
  // 并发 cudaFree 会楔死其他线程的流调用 —— 引擎只写 pinned/发 verbs WR,
  // 搬运全在消费者核)。
  bool gated = false;               // 本 collective 走流序完成
  // ── dev-pipe2:机内 final-hop 直写(1-copy)。本 rank 作为接收方,每 coll
  // 在 submit(app 线程,门外)导出 recvbuf 所在分配的 IPC 句柄;engine 在
  // input 就绪后经 SHM 邮箱发布窗口。发送方按句柄内容缓存映射(物理分配
  // 唯一 ⇒ 免 VA 复用陈旧问题);缓存 miss 时该 coll 走环路径,句柄排队到
  // 下一次 submit 在 app 线程(门外,无 cudaFree 楔死风险)补 open。
  bool my_rwin_valid = false;
  uint8_t my_rwin_handle[64] = {0};
  uint64_t my_rwin_base_off = 0;    // recvbuf - 分配基址
  bool rwin_published = false;
  uint64_t dlv_base[MAX_RANKS] = {0};    // 本 coll 起点的 DLV 快照
  uint64_t dlv_folded[MAX_RANKS] = {0};  // 本 coll 已折入账目的直写字节
  std::mutex ipc_mu;                // app(open)与 engine(查表/入队)交接
  std::vector<std::array<uint8_t, 64>> ipc_pending;   // 待 open 的句柄
  std::map<std::array<uint8_t, 64>, void*> ipc_map;   // 句柄内容 → 本进程映射基址
  void* done_flag_host = nullptr;   // uint32_t*(pinned mapped;引擎完成时写 coll_id)
  uint64_t done_flag_dev = 0;
  void* gate_mod = nullptr;         // 自旋门核 CUmodule(PTX JIT)
  void* gate_fn = nullptr;
  bool self_copy_pending = false;   // 坑 #51:自拷贝等 input_ready 后交消费者核
  int self_kq_pending = 0;          // 在途自拷贝 kdesc 数(recv-即-开门要等它清零)
  // S3 方案 N:GPU 消费者内核 —— host 只写 pinned 描述符,交付拷贝与
  // 完成判定全在设备(契约见 consumer_kernel.cu)。
  struct KDesc { uint64_t src, dst, len, pad; };
  KDesc* kdesc_ring = nullptr;      // 描述符环(pinned mapped,host 写)
  uint64_t kdesc_ring_dev = 0;
  uint8_t* kdesc_ctr = nullptr;     // 计数块 192B:+0 head / +64 done / +128 stop
  uint64_t kdesc_ctr_dev = 0;
  uint64_t kdesc_prog_dev = 0;      // 设备侧 prog[grid](cudaMalloc)
  uint32_t kdesc_cap = 0;           // 2 的幂
  uint64_t kdesc_pushed = 0;        // host 累计推入(head 的镜像)
  uint64_t kdesc_consumed = 0;      // host 已收编的 done
  void* kcons_mod = nullptr;
  void* kcons_fn = nullptr;
  void* stream_cons = nullptr;      // 消费者核专用流(门在 caller 流上的 host 回调)
  uint64_t kdesc_gen = 0;           // 代际号(每 coll +1;核见变即退,坑 #36)
  bool kdesc_active = false;        // 本次 collective 走消费者内核交付
  void* kflag_fn = nullptr;         // mosaic_flagset CUfunction(input 就绪标志)
  bool input_ready = false;         // 本 coll input_ev 已确认(host-API hop0 门)
  uint64_t caller_stream = 0;       // 本 coll 的调用方流(诊断:查它是否卡住)
  // 跨机控制面的档位。两者**互斥**,engine_start 的能力探测二选一置位:
  //   ctrl_verbs —— RDMA 档,**默认且首选**:独立 ctrl QP 走 RDMA SEND;
  //   ctrl_erpc  —— 兼容档,仅当机器没有 RDMA 卡:eRPC + DPDK 用户态以太网
  //                 (代价见 erpc_ctrl.cc 头注;有卡时不要用)。
  // 两档都在池满/超长/建链失败时自动回退 TCP;同机 peer 恒 TCP。
  bool ctrl_verbs = false;
  bool ctrl_erpc = false;

  // ---- 本次 collective ----
  const void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  std::vector<size_t> scounts, sdispls, rcounts, rdispls;
  uint64_t total_send = 0, posted_send = 0, total_recv_expect = 0;
  std::atomic<int> coll_state{0};   // 0=idle 1=running 2=done
  uint64_t coll_start_ns = 0, coll_end_ns = 0;
  int coll_id = 0;

  // ---- 队列 ----
  std::deque<PendSlice> q_ready;    // 已装配、等待发射
  std::deque<PendSlice> q_d2h;      // 在途 D2H(按 stream 顺序)
  std::deque<PendSlice> q_h2d;      // 在途 H2D
  std::deque<PendSlice> q_relay;    // 等待续跳(已落 relay 池)

  // ---- 线程 ----
  // cudaSetDevice 是**每线程**属性:engine_start 在 app 线程选的卡,event
  // 线程不会自动继承(默认 dev0)—— dev≠0 的 rank 会在 event 线程创建的
  // event 上 eventRecord 报 invalid resource handle(Raven R13)。
  // 选定的卡存这里,event_loop 进门先 setDevice。
  int cuda_dev = 0;
  std::thread ev_thread;
  std::atomic<bool> running{false};
  std::mutex mu;                    // 只保护 app↔event 的少量交接
  // 控制连接的发送队列会被两个线程碰:app 线程(barrier/stats 里主动发)和
  // event 线程(轮询里 flush)。必须加锁 —— 不加就是 deque 并发改写,实测直接段错误。
  std::mutex ctrl_mu;
  Stats st{};

  // barrier
  std::atomic<int> barrier_epoch{0};
  // event 线程写、app 线程读 ⇒ 必须是原子的
  std::atomic<int> barrier_seen[MAX_RANKS];
  Comm() {
    for (int i = 0; i < MAX_RANKS; i++) barrier_seen[i].store(0);
    for (int i = 0; i < 4; i++) { stats_cnt[i].store(0); stats_slot_epoch[i].store(-1);
                                  ag_cnt[i].store(0); ag_slot_epoch[i].store(-1); }
  }
  // 小 allgather(控制面)
  std::vector<uint8_t> ag_buf;
  std::atomic<int> ag_cnt[4];
  std::atomic<int> ag_slot_epoch[4];
  std::atomic<int> ag_epoch{0};
  size_t ag_bytes = 0;
  std::mutex ag_mu;
  // makespan 归约。必须按 epoch 分槽:对端的 STATS 可能**早于**本 rank 进入
  // statsSync 就到达,用"进来先清零计数"的写法会把先到的那几条抹掉 → 永久等待。
  // (barrier 用的是单调 epoch 比较,天然免疫;这里补齐同样的性质。)
  std::atomic<int> stats_epoch{0};
  std::atomic<int> stats_cnt[4];
  std::atomic<int> stats_slot_epoch[4];
  double stats_max[4] = {0, 0, 0, 0};
};

// ─────────────────────────────────────────────────────────── 跨文件函数
// net.cc
int  net_setup(Comm* c);
void net_teardown(Comm* c);
int  ctrl_send(Comm* c, int peer, CtrlType t, uint16_t aux, const void* body, uint32_t n);
int  ctrl_bcast(Comm* c, CtrlType t, uint16_t aux, const void* body, uint32_t n);
// 事件循环里调用:收控制/数据,返回处理的事件数
int  net_poll(Comm* c);
// S3 二期:coll 收尾 —— 写 stop=已推入总数让消费者核排干后**退出**(核绝不
// 常驻:坑 #36,常驻 × torch 的 cudaFree 设备级隐式同步 = 永久死锁)。
// **错误/超时路径也必须调**,否则核悬死、下一次 cudaFree 就卡住。
// 换代安全由 gen 保证(下一次 arm 先 gen++ 杀残留老核),这里只管收尾。
inline void kdesc_release_gate(Comm* c) {
  if (c->kdesc_ctr)
    __atomic_store_n((uint64_t*)(c->kdesc_ctr + 128), c->kdesc_pushed, __ATOMIC_RELEASE);
  c->kdesc_active = false;
}

ShmRing* shm_ring_create(const std::string& name, int nslots, int slot_bytes, bool creator);
void shm_ring_set_dev(ShmRing* r, void* p);
void shm_ring_put_ipc(ShmRing* r, const void* handle, size_t n);
bool shm_ring_get_ipc(ShmRing* r, void* handle, size_t n);
void shm_ring_destroy(ShmRing* r, const std::string& name, bool creator);
bool shm_ring_reserve(ShmRing* r, void** payload, SliceHdr** hdr);
void shm_ring_publish(ShmRing* r);
bool shm_ring_peek(ShmRing* r, void** payload, SliceHdr** hdr);
void shm_ring_commit(ShmRing* r);
void shm_ring_pop(ShmRing* r);
int  shm_ring_inflight(ShmRing* r);
void shm_ring_region(ShmRing* r, void** base, size_t* bytes);
// dev-pipe2 机内直写邮箱(net.cc;方向约定:s→d 数据环上,RWIN 由 d 写 s 读,
// DLV 由 s 写 d 读)。dst_off == ~0ull = 接收方声明"本 coll 不可直写,走环路径"。
void shm_rwin_publish(ShmRing* r, uint32_t coll, const void* handle64, uint64_t dst_off);
bool shm_rwin_read(ShmRing* r, uint32_t coll, void* handle64, uint64_t* dst_off);
void shm_dlv_add(ShmRing* r, uint64_t bytes);
uint64_t shm_dlv_read(ShmRing* r);
void shm_ring_stat(ShmRing* r, uint64_t* head, uint64_t* tail, int* inflight, int* pending);
void stage_release(Comm* c, int slot);
// 建链期的阻塞式小报文交换(经已有 TCP 数据连接;verbs 用它换 QP 信息)
int  net_blocking_exchange(Comm* c, int peer, const void* mine, void* remote, size_t n);
// NIC 物理单元共识(verbs 建链后、event 线程启动前调;net.cc):
// 同机经 SHM 表、跨机经 pairwise 交换,填 topo.nic_leader[] 与 nic_unit_gbps[]。
// 无 verbs 面(TCP/离线)= 不动恒等映射。
void nic_unit_exchange(Comm* c);
bool verbs_nic_ident(Comm* c, uint64_t* key, double* gbps);

// net.cc:已组帧(含 CtrlHdr)报文直接走 TCP 队列
int  ctrl_send_tcp(Comm* c, int peer, const void* framed, uint32_t n);

// verbs.cc(RDMA 数据面 + 控制面;data_plane=false 仅供无数据面调试形态)
int  verbs_setup(Comm* c, bool data_plane);
int  rdma_rx_slots(int slice_bytes);   // 每 pair 预投 recv 槽数(verbs.cc;W 流控规则用)
void verbs_teardown(Comm* c);
int  verbs_post_slice(Comm* c, int peer, const SliceHdr& h, int stage_slot);
int  verbs_poll(Comm* c);
void verbs_repost(Comm* c, int token);
// O3:控制面报文经数据 QP 发送(小报文,注册槽池;0 = 已发,-1 = 不可用/无槽,
// 调用方回退 TCP)。app 线程与 event 线程都会调,内部用 ctrl_mu 保护槽池。
int  verbs_ctrl_send(Comm* c, int peer, const void* msg, uint32_t n);

// erpc_ctrl.cc(控制面兼容档:eRPC over DPDK,不需要 RDMA verbs 权限之外的
// RDMA 能力 —— 用户态以太网,给没有 RDMA 卡的机器兜底)。
// setup 必须在 event 线程调用(erpc::Rpc 与创建线程绑定);send 返回
// 0 = 已提交,-1 = 不可用/无槽(调用方回退 TCP);poll 由 net_poll 每圈驱动。
int  erpc_ctrl_setup(Comm* c);
void erpc_ctrl_teardown(Comm* c);
int  erpc_ctrl_poll(Comm* c);
int  erpc_ctrl_send(Comm* c, int peer, const void* msg, uint32_t n);

// topo.cc
void topo_init(Topo* t, int nranks, int G, int rank);
void ledger_init(Comm* c);

// engine.cc:检测器喂入(属主本地;which: 0=NIC↑ 完成事件, 1=NIC↓ 到达事件)
void det_feed(Comm* c, int which, uint64_t bytes);
// engine.cc:测试入口 —— 直接驱动一次波装配(replay_core_test 对拍用,
// 不经 event 线程;调用前 c 的 demand/dst.remaining/ledger 须已就位)
void engine_test_emit_wave(Comm* c);
// 同上,src 驱动版(调用前 srcc.remaining 须已就位;行门在离线态视为全开)
void engine_test_emit_wave_src(Comm* c);
// engine.cc:src 波的 ACK 计数(verbs_poll 在数据 WR send CQ 成功时调;
// RC send 完成 = 对端已收,hop0 且 h.src==本 rank 的片才计)
void src_wave_ack(Comm* c, const SliceHdr& h);

// engine.cc
void engine_start(Comm* c);
void engine_stop(Comm* c);
void engine_submit(Comm* c);           // app 线程提交一次 collective

void on_ctrl_msg(Comm* c, int peer, const CtrlHdr& h, const void* body);
// 一个 slice 的当前 hop 已经落到本 rank 的 host 内存里。
// shm_ring_idx >= 0 → payload 在该 SHM 环的槽里(消费完要 pop);
// stage_slot   >= 0 → payload 在 TCP staging 池的槽里(消费完要 release);
// verbs_token >= 0 → payload 在 RDMA 预投槽里(H2D 完成后 verbs_repost 归还)。
void on_slice_arrived(Comm* c, const SliceHdr& h, void* host_payload,
                      int shm_ring_idx, int stage_slot, int verbs_token = -1);

}  // namespace mosaic
#endif
