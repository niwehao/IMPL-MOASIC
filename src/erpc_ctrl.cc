/* erpc_ctrl.cc —— S0d(sota_plan 2026-08-16):控制面走 eRPC(kernel-bypass)。
 *
 * 背景:S0c 实测控制面 kernel TCP@管理网 p50≈35µs/p99≈56µs,而 eRPC-IB
 * (UD QP,可靠 RPC,NSDI'19 开源实现,Code/eRPC)p50 2.7µs —— 13×。
 * DPDK 数据面在 Raven 非 root 被平台锁死(DevX 固件拒 + CAP_NET_RAW,
 * 见 Doc/dpdk_ctrlplane_survey.md §4b),eRPC 换 transport=infiniband
 * 即为本平台可执行的同类最优;权限环境变化时改 transport 即迁 DPDK。
 *
 * 形态:MOSAIC_CTRL=erpc(默认关 = TCP 旧行为;与 O3 verbs 并列第三条路)。
 * 只接**跨机** ctrl 报文;同机照旧 TCP。会话未就绪/无缓冲时回退 TCP ——
 * 两条通路交错到达安全(报文自带长度分帧 + 各类型天然容乱序:TELEM 带
 * seq、DEMAND/POLICY/FIN 带 cid、BARRIER/STATS/AG 带 epoch)。
 *
 * 线程契约:erpc::Rpc 非线程安全,create/req/poll 全部只在 event 线程;
 * app 线程(barrier/stats/allgather)的发送进 outq,由 event 线程代发。
 * 这与"协议决策全在单 event 线程"的可复现性契约一致。
 *
 * 构建:make ERPC=1(Raven build_mosaic.sh 默认带);不带 = 编 stub,
 * 运行时 MOSAIC_CTRL=erpc 自动降级 TCP 并告警。
 */
#include "internal.h"

#ifdef MOSAIC_HAVE_ERPC

#include <deque>
#include <utility>

#include "rpc.h"   // eRPC(Code/eRPC/src;补丁见 common/raven/patches/erpc_raven.patch)

namespace mosaic {

static constexpr uint8_t kReqCtrl = 1;
static constexpr size_t kMaxCtrl = 8192;   // 控制报文上限(policy ~3KB,留裕量)
static constexpr int kPoolSlots = 128;     // 在途请求缓冲池(req+resp 成对)

struct ErpcCtx {
  Comm* comm = nullptr;
  erpc::Nexus* nexus = nullptr;
  erpc::Rpc<erpc::CTransport>* rpc = nullptr;
  int sess[MAX_RANKS];
  uint8_t sess_ok[MAX_RANKS] = {0};
  uint64_t retry_ns[MAX_RANKS] = {0};   // 会话重试退避(对端 Rpc 可能晚于我注册)
  int retries[MAX_RANKS] = {0};
  std::vector<std::string> peer_uri;
  erpc::MsgBuffer req_buf[kPoolSlots], resp_buf[kPoolSlots];
  std::vector<int> pool_free;
  // app 线程外发队列(event 线程代发)
  std::mutex out_mu;
  std::deque<std::pair<int, std::vector<uint8_t>>> outq;
  std::thread::id ev_tid;
  uint64_t sent = 0, fell_back = 0;
};

static ErpcCtx* EC(Comm* c) { return (ErpcCtx*)c->erpc; }

// 会话事件:connected 才开闸;断连回 TCP(不重连 —— 控制面有 TCP 兜底)
static void sm_handler(int session_num, erpc::SmEventType ev, erpc::SmErrType err, void* ctx) {
  auto* ec = (ErpcCtx*)ctx;
  for (int p = 0; p < ec->comm->cfg.nranks; p++) {
    if (ec->sess[p] != session_num) continue;
    if (ev == erpc::SmEventType::kConnected) {
      ec->sess_ok[p] = 1;
      MLOG_INFO("eRPC ctrl 会话就绪 r%d(sess %d,第 %d 次尝试)", p, session_num,
                ec->retries[p]);
    } else if (ev == erpc::SmEventType::kConnectFailed) {
      // 对端 Rpc 可能还没注册(启动竞态,"Invalid remote Rpc ID")——
      // 标记待重试,poll 里按退避重建会话
      ec->sess_ok[p] = 0;
      ec->sess[p] = -1;
      ec->retry_ns[p] = now_ns() + 20 * 1000 * 1000;   // 20ms 退避
      if (ec->retries[p] <= 3)
        MLOG_INFO("eRPC ctrl 会话连接失败 r%d err=%d(将重试)", p, (int)err);
    } else {
      ec->sess_ok[p] = 0;
      MLOG_WARN("eRPC ctrl 会话事件 r%d ev=%d err=%d(回退 TCP)", p, (int)ev, (int)err);
    }
    return;
  }
}

// 服务端:收到对端 ctrl 报文 → 走与 TCP 完全相同的 on_ctrl_msg 入口
static void req_handler(erpc::ReqHandle* rh, void* ctx) {
  auto* ec = (ErpcCtx*)ctx;
  const erpc::MsgBuffer* m = rh->get_req_msgbuf();
  const size_t n = m->get_data_size();
  if (n >= sizeof(CtrlHdr)) {
    CtrlHdr h;
    memcpy(&h, m->buf_, sizeof(h));
    on_ctrl_msg(ec->comm, h.rank, h, m->buf_ + sizeof(CtrlHdr));
  }
  auto& resp = rh->pre_resp_msgbuf_;
  ec->rpc->resize_msg_buffer(&resp, 1);
  ec->rpc->enqueue_response(rh, &resp);
}

static void cont_func(void* ctx, void* tag) {
  auto* ec = (ErpcCtx*)ctx;
  ec->pool_free.push_back((int)(intptr_t)tag);   // event 线程回调,无锁
}

// event 线程内的真正发送(池空 = 背压,返回 -1 → 调用方回退 TCP)
static int send_now(ErpcCtx* ec, int peer, const void* msg, uint32_t n) {
  if (!ec->sess_ok[peer] || n > kMaxCtrl) return -1;
  if (ec->pool_free.empty()) return -1;
  int s = ec->pool_free.back();
  ec->pool_free.pop_back();
  ec->rpc->resize_msg_buffer(&ec->req_buf[s], n);
  memcpy(ec->req_buf[s].buf_, msg, n);
  ec->rpc->enqueue_request(ec->sess[peer], kReqCtrl, &ec->req_buf[s], &ec->resp_buf[s],
                           cont_func, (void*)(intptr_t)s);
  ec->sent++;
  return 0;
}

int erpc_ctrl_setup(Comm* c) {
  // 必须由 event 线程调用(Rpc 与创建线程绑定)
  const int me = c->cfg.rank, n = c->cfg.nranks;
  const char* hosts = getenv("MOSAIC_HOSTS");
  if (!hosts || !*hosts) { MLOG_WARN("E1030 MOSAIC_CTRL=erpc 但无 MOSAIC_HOSTS"); return -1; }
  std::vector<std::string> ip;
  {
    std::string s(hosts), cur;
    for (char ch : s) { if (ch == ',') { ip.push_back(cur); cur.clear(); } else cur += ch; }
    ip.push_back(cur);
  }
  const int G = c->cfg.gpus_per_host;
  auto uri_of = [&](int r) {
    int base = getenv("MOSAIC_ERPC_PORT") ? atoi(getenv("MOSAIC_ERPC_PORT")) : 31850;
    const std::string& h = ip.size() == (size_t)n ? ip[r] : ip[r / G];
    return h + ":" + std::to_string(base + r);
  };
  auto* ec = new ErpcCtx();
  ec->comm = c;
  ec->ev_tid = std::this_thread::get_id();
  for (int p = 0; p < MAX_RANKS; p++) ec->sess[p] = -1;
  try {
    ec->nexus = new erpc::Nexus(uri_of(me), 0 /*numa*/, 0 /*bg threads*/);
    ec->nexus->register_req_func(kReqCtrl, req_handler);
    int phy = getenv("MOSAIC_ERPC_PHYPORT") ? atoi(getenv("MOSAIC_ERPC_PHYPORT")) : 0;
    ec->rpc = new erpc::Rpc<erpc::CTransport>(ec->nexus, ec, (uint8_t)me, sm_handler, (uint8_t)phy);
    for (int s = 0; s < kPoolSlots; s++) {
      ec->req_buf[s] = ec->rpc->alloc_msg_buffer_or_die(kMaxCtrl);
      ec->resp_buf[s] = ec->rpc->alloc_msg_buffer_or_die(16);
      ec->pool_free.push_back(s);
    }
    ec->peer_uri.resize(n);
    for (int p = 0; p < n; p++) {
      if (p == me || c->topo.same_host(me, p)) continue;
      ec->peer_uri[p] = uri_of(p);
      ec->sess[p] = ec->rpc->create_session(ec->peer_uri[p], (uint8_t)p);
      ec->retries[p] = 1;
      if (ec->sess[p] < 0) MLOG_WARN("eRPC create_session r%d 失败(poll 重试)", p);
    }
  } catch (const std::exception& e) {
    MLOG_WARN("E1031 eRPC ctrl 初始化失败:%s(控制面留在 TCP)", e.what());
    delete ec->rpc; delete ec->nexus; delete ec;
    return -1;
  }
  c->erpc = ec;
  MLOG_INFO("控制面 = eRPC(%s;跨机 ctrl 走 UD RPC,同机仍 TCP)", uri_of(me).c_str());
  return 0;
}

void erpc_ctrl_teardown(Comm* c) {
  auto* ec = EC(c);
  if (!ec) return;
  c->erpc = nullptr;
  MLOG_INFO("eRPC ctrl 关闭:sent=%llu fallback=%llu",
            (unsigned long long)ec->sent, (unsigned long long)ec->fell_back);
  delete ec->rpc;     // 析构内部注销 session/缓冲
  delete ec->nexus;
  delete ec;
}

int erpc_ctrl_poll(Comm* c) {
  auto* ec = EC(c);
  if (!ec || !ec->rpc) return 0;
  // 轮询节流:event 循环每圈都进这里,而 run_event_loop_once 自身要 ~µs 级
  // (rdtsc/队列/CQ 扫描)—— 不节流会拖慢所有泵(8 rank 实测慢 5-8%)。
  // 默认 5µs 一次:对 2.7µs RTT 的报文延迟贡献 ≤ 半个节拍,可 env 调。
  {
    static thread_local uint64_t next_ns = 0;
    static const uint64_t gap =
        (getenv("MOSAIC_ERPC_POLL_US") ? atoll(getenv("MOSAIC_ERPC_POLL_US")) : 5) * 1000ull;
    uint64_t t0 = now_ns();
    if (t0 < next_ns) return 0;
    next_ns = t0 + gap;
  }
  // 会话重建(启动竞态 / create 失败;上限 200 次防漏)
  {
    uint64_t t = now_ns();
    for (int p = 0; p < c->cfg.nranks; p++) {
      if (p == c->cfg.rank || ec->peer_uri[p].empty()) continue;
      if (ec->sess[p] >= 0 || ec->retries[p] > 200 || t < ec->retry_ns[p]) continue;
      ec->sess[p] = ec->rpc->create_session(ec->peer_uri[p], (uint8_t)p);
      ec->retries[p]++;
      ec->retry_ns[p] = t + 20 * 1000 * 1000;
    }
  }
  // 代发 app 线程排队的报文(仍不通则回 TCP —— 直接塞回 TCP 队列)
  for (;;) {
    std::pair<int, std::vector<uint8_t>> it;
    {
      std::lock_guard<std::mutex> lk(ec->out_mu);
      if (ec->outq.empty()) break;
      it = std::move(ec->outq.front());
      ec->outq.pop_front();
    }
    if (send_now(ec, it.first, it.second.data(), (uint32_t)it.second.size()) != 0) {
      ec->fell_back++;
      ctrl_send_tcp(c, it.first, it.second.data(), (uint32_t)it.second.size());
    }
  }
  ec->rpc->run_event_loop_once();
  return 0;
}

int erpc_ctrl_send(Comm* c, int peer, const void* msg, uint32_t n) {
  auto* ec = EC(c);
  if (!ec || !ec->rpc) return -1;
  if (std::this_thread::get_id() == ec->ev_tid) return send_now(ec, peer, msg, n);
  // app 线程:入队由 event 线程代发(barrier/stats 语义上等得起 1 圈)
  std::lock_guard<std::mutex> lk(ec->out_mu);
  ec->outq.emplace_back(peer, std::vector<uint8_t>((const uint8_t*)msg, (const uint8_t*)msg + n));
  return 0;
}

}  // namespace mosaic

#else  // !MOSAIC_HAVE_ERPC —— stub:运行时选 erpc 会自动降级 TCP

namespace mosaic {
int erpc_ctrl_setup(Comm*) { return -1; }
void erpc_ctrl_teardown(Comm*) {}
int erpc_ctrl_poll(Comm*) { return 0; }
int erpc_ctrl_send(Comm*, int, const void*, uint32_t) { return -1; }
}  // namespace mosaic

#endif
