/* net.cc —— L0 传输层。平台差异**只**存在于本文件(Imp_Plan/01 §3)。
 *
 *  NV 管(机内)= POSIX 共享内存 SPSC 环 + cudaMemcpyAsync(D2H/H2D)
 *                plan 明确要求"用共享内存把本机划分不同区域模拟 NVLink"。
 *  vNIC(跨机)= 每有序 pair 一条 TCP(直连 100GbE 上限速到 nic_gbps,
 *                所以 TCP 与 RoCE 在本模拟里等价 —— 见 log.md「为什么 TCP 够用」)。
 *  控制面    = 全 mesh TCP,承载通道 A(遥测)与通道 B(策略广播)。
 *
 * 通道 B 走可靠传输的理由(重要):论文里通道 B 是"只存最新一份"的账本视图输入,
 * 但在真机上它**同时是执行指令**(src 靠它才知道这一波发哪些字节)。丢一条 =
 * 丢数据,所以必须可靠。账本视图仍按"最新一份"语义使用,两者不冲突。
 */
#include "internal.h"

#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdlib>

namespace mosaic {

// ─────────────────────────────────────────────────────────── SHM 环
struct ShmRingHdr {
  std::atomic<uint64_t> head;   // 生产者已发布的条数
  std::atomic<uint64_t> tail;   // 消费者已回收的条数
  uint32_t nslots, slot_bytes;
  // IPC 模式:payload 放在**设备内存**里(模拟 NVLink 的 GPU↔GPU 直达),
  // 共享内存只承载控制信息。IPC 句柄直接写在环头里 —— 双方本来就共享这块
  // 内存,不需要再为句柄交换设计一套协议。
  std::atomic<uint32_t> ipc_ready;
  uint8_t ipc_handle[128];
};
struct ShmSlotHdr {
  std::atomic<uint64_t> ready;  // == seq+1 表示 payload 有效
  SliceHdr h;
};

struct ShmRing {
  void* base = nullptr;
  size_t bytes = 0;
  ShmRingHdr* hdr = nullptr;
  uint8_t* slots = nullptr;     // nslots × (sizeof(ShmSlotHdr) + slot_bytes)
  int nslots = 0, slot_bytes = 0;
  size_t stride = 0;
  bool host_payload = true;     // false = payload 在设备内存(IPC 模式)
  void* dev_ring = nullptr;     // 生产者:自己 cudaMalloc 的;消费者:IPC 打开的对端指针
  // 生产者:local_head = 已发布数,pending = 已占用但等 D2H 完成才能发布的数。
  // 消费者:local_tail = 已回收数,inflight = 已取走但等 H2D 完成才能回收的数。
  // 两个"在途"计数不能省:CUDA 拷贝是异步的,槽位的生命周期比一次函数调用长。
  uint64_t local_head = 0, local_tail = 0;
  int pending = 0, inflight = 0;
  uint8_t* slot_ptr(uint64_t seq) const { return slots + (seq % nslots) * stride; }
};

ShmRing* shm_ring_create(const std::string& name, int nslots, int slot_bytes, bool creator,
                         bool payload_in_shm) {
  size_t stride = sizeof(ShmSlotHdr) + (payload_in_shm ? (size_t)slot_bytes : 0);
  size_t bytes = sizeof(ShmRingHdr) + (size_t)nslots * stride;
  int fd = -1;
  for (int tries = 0; tries < 20000; tries++) {   // 消费者可能早于生产者到达
    fd = shm_open(name.c_str(), creator ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
    if (fd >= 0) break;
    if (!creator && errno == ENOENT) { usleep(500); continue; }
    MLOG_ERR("shm_open(%s) 失败: %s", name.c_str(), strerror(errno));
    return nullptr;
  }
  if (fd < 0) { MLOG_ERR("shm_open(%s) 超时", name.c_str()); return nullptr; }
  if (creator && ftruncate(fd, bytes) != 0) {
    MLOG_ERR("ftruncate(%s) 失败: %s", name.c_str(), strerror(errno));
    close(fd); return nullptr;
  }
  // 非 creator 要等 creator 把大小设好,否则 mmap 出来是空洞
  if (!creator) {
    struct stat st{};
    for (int t = 0; t < 20000; t++) {
      if (fstat(fd, &st) == 0 && (size_t)st.st_size >= bytes) break;
      usleep(500);
    }
  }
  void* base = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) { MLOG_ERR("mmap(%s) 失败: %s", name.c_str(), strerror(errno)); return nullptr; }

  ShmRing* r = new ShmRing();
  r->base = base; r->bytes = bytes;
  r->hdr = (ShmRingHdr*)base;
  r->slots = (uint8_t*)base + sizeof(ShmRingHdr);
  r->nslots = nslots; r->slot_bytes = slot_bytes; r->stride = stride;
  r->host_payload = payload_in_shm;
  if (creator) {
    r->hdr->head.store(0); r->hdr->tail.store(0);
    r->hdr->nslots = nslots; r->hdr->slot_bytes = slot_bytes;
    for (int i = 0; i < nslots; i++)
      ((ShmSlotHdr*)(r->slots + (size_t)i * stride))->ready.store(0);
  }
  return r;
}

void shm_ring_destroy(ShmRing* r, const std::string& name, bool creator) {
  if (!r) return;
  munmap(r->base, r->bytes);
  if (creator) shm_unlink(name.c_str());
  delete r;
}

// 生产者:占一个槽(payload 由调用方用 cudaMemcpyAsync 填),D2H 完成后调 publish。
// 允许同时占多个槽;publish 按占用顺序发布(同一 stream 的拷贝顺序完成,天然成立)。
bool shm_ring_reserve(ShmRing* r, void** payload, SliceHdr** hdr) {
  uint64_t seq = r->local_head + (uint64_t)r->pending;
  uint64_t tail = r->hdr->tail.load(std::memory_order_acquire);
  if (seq - tail >= (uint64_t)r->nslots) return false;   // 满 = 背压
  uint8_t* s = r->slot_ptr(seq);
  *hdr = &((ShmSlotHdr*)s)->h;
  *payload = r->host_payload ? (void*)(s + sizeof(ShmSlotHdr))
                             : (void*)((uint8_t*)r->dev_ring + (seq % r->nslots) * (size_t)r->slot_bytes);
  r->pending++;
  return true;
}

void shm_ring_publish(ShmRing* r) {
  uint8_t* s = r->slot_ptr(r->local_head);
  ((ShmSlotHdr*)s)->ready.store(r->local_head + 1, std::memory_order_release);
  r->local_head++;
  if (r->pending > 0) r->pending--;
  r->hdr->head.store(r->local_head, std::memory_order_release);
}

// 消费者:先 peek(不改状态)看清是不是中转、要不要中转槽,决定收下了再 commit。
// 环是 FIFO,pop 必须与 commit 同序 —— 所以头部消费不了时整条环停住(队头阻塞),
// 这与真实环形缓冲的行为一致,不能"跳过头去收后面的"。
bool shm_ring_peek(ShmRing* r, void** payload, SliceHdr** hdr) {
  uint64_t seq = r->local_tail + (uint64_t)r->inflight;
  uint8_t* s = r->slot_ptr(seq);
  if (((ShmSlotHdr*)s)->ready.load(std::memory_order_acquire) != seq + 1) return false;
  *hdr = &((ShmSlotHdr*)s)->h;
  *payload = r->host_payload ? (void*)(s + sizeof(ShmSlotHdr))
                             : (void*)((uint8_t*)r->dev_ring + (seq % r->nslots) * (size_t)r->slot_bytes);
  return true;
}

void shm_ring_commit(ShmRing* r) { r->inflight++; }

void shm_ring_pop(ShmRing* r) {
  r->local_tail++;
  if (r->inflight > 0) r->inflight--;
  r->hdr->tail.store(r->local_tail, std::memory_order_release);
}

int shm_ring_inflight(ShmRing* r) { return r ? r->inflight : 0; }


void shm_ring_stat(ShmRing* r, uint64_t* head, uint64_t* tail, int* inflight, int* pending) {
  if (!r) { *head = *tail = 0; *inflight = *pending = 0; return; }
  *head = r->hdr->head.load(); *tail = r->hdr->tail.load();
  *inflight = r->inflight; *pending = r->pending;
}

// engine.cc 要把整段 SHM 钉页(cudaHostRegister),但那边看不到 ShmRing 的定义
bool shm_ring_is_dev(ShmRing* r) { return r && !r->host_payload; }
void* shm_ring_dev(ShmRing* r) { return r ? r->dev_ring : nullptr; }
void shm_ring_set_dev(ShmRing* r, void* p) { if (r) r->dev_ring = p; }
// IPC 句柄的发布/读取(经环头这块共享内存,零额外协议)
void shm_ring_put_ipc(ShmRing* r, const void* handle, size_t n) {
  if (!r || n > sizeof(r->hdr->ipc_handle)) return;
  memcpy(r->hdr->ipc_handle, handle, n);
  r->hdr->ipc_ready.store(1, std::memory_order_release);
}
bool shm_ring_get_ipc(ShmRing* r, void* handle, size_t n) {
  if (!r || n > sizeof(r->hdr->ipc_handle)) return false;
  for (int t = 0; t < 40000; t++) {
    if (r->hdr->ipc_ready.load(std::memory_order_acquire)) {
      memcpy(handle, r->hdr->ipc_handle, n);
      return true;
    }
    usleep(200);
  }
  return false;
}

void shm_ring_region(ShmRing* r, void** base, size_t* bytes) {
  if (!r) { *base = nullptr; *bytes = 0; return; }
  *base = r->base; *bytes = r->bytes;
}

// ─────────────────────────────────────────────────────────── TCP 连接
// 2026-08-19 清理:TCP **数据面**已删除(slice 只走 verbs/nvshmem;建链失败 =
// 硬错 fail-fast,不再静默降速)。TCP 只承担 ①bootstrap(QP/uid 交换,经
// dconn 的阻塞收发)②控制面(通道 A/B/demand/FIN,MOSAIC_CTRL=tcp 或
// verbs 回退)。
struct Conn {
  int fd = -1;
  int peer = -1;
  bool is_data = false;        // 数据连接现在只用于 bootstrap 阻塞交换
  // rx
  std::vector<uint8_t> rbuf;   // ctrl:消息装配缓冲
  size_t rlen = 0;
  // tx
  std::deque<std::vector<uint8_t>> wq_ctrl;
  size_t wq_ctrl_off = 0;
};

static void set_nonblock(int fd) {
  int f = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, f | O_NONBLOCK);
}
static void set_sockopts(int fd) {
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int sz = 4 << 20;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

// staging 槽归还(取在 engine.cc 的 stage_alloc2)
void stage_release(Comm* c, int slot) { if (slot >= 0) c->stage_free.push_back(slot); }

// ─────────────────────────────────────────────────────────── 建链
static std::vector<std::string> host_ips(int H) {
  std::vector<std::string> v;
  const char* e = getenv("MOSAIC_HOSTS");
  std::string s = e && *e ? e : "127.0.0.1";
  size_t p = 0;
  while (p <= s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    v.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  while ((int)v.size() < H) v.push_back(v.back());
  return v;
}

// R19:同一 SLURM 作业里多次 srun(step)共用一个 MOSAIC_JOB 时,上一步
// 崩溃留下的陈旧 /dev/shm 环(内含死进程的 IPC 句柄)会被下一步打开 ⇒
// E5005 invalid resource handle,上下文被毒化(29345541 的 fast 后端实测)。
// 把 SLURM_STEP_ID 折进 shm 命名;本机无 SLURM ⇒ 行为不变。
static int shm_job_id() {
  int job = getenv("MOSAIC_JOB") ? atoi(getenv("MOSAIC_JOB")) : 0;
  const char* se = getenv("SLURM_STEP_ID");
  if (se && *se) job = (job % 10000000) * 100 + (atoi(se) % 100) + 1;
  return job;
}

int net_setup(Comm* c) {
  const int n = c->cfg.nranks, me = c->cfg.rank;
  const int port_base = getenv("MOSAIC_PORT_BASE") ? atoi(getenv("MOSAIC_PORT_BASE")) : 31000;
  const int job = shm_job_id();
  auto ips = host_ips(c->topo.H);
  c->conn.assign(n, nullptr);
  c->dconn.assign(n, nullptr);
  c->tx_ring.assign(n, nullptr);
  c->rx_ring.assign(n, nullptr);

  c->epfd = epoll_create1(0);
  if (c->epfd < 0) { MLOG_ERR("epoll_create1: %s", strerror(errno)); return -1; }

  // 1) listen
  c->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(c->listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = htons(port_base + me);
  if (bind(c->listen_fd, (sockaddr*)&a, sizeof(a)) != 0) {
    MLOG_ERR("bind(%d): %s", port_base + me, strerror(errno)); return -1;
  }
  listen(c->listen_fd, 128);

  // 需要几条入向连接:来自所有 p > me 的(ctrl 恒有,跨机再加一条 data)
  int want_in = 0, want_out = 0;
  for (int p = 0; p < n; p++) {
    if (p == me) continue;
    int nch = c->topo.same_host(me, p) ? 1 : 2;
    if (p > me) want_in += nch; else want_out += nch;
  }

  // 2) 主动连低 rank
  for (int p = 0; p < me; p++) {
    int nch = c->topo.same_host(me, p) ? 1 : 2;
    for (int ch = 0; ch < nch; ch++) {
      int fd = -1;
      for (int tries = 0; tries < 6000; tries++) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_port = htons(port_base + p);
        inet_pton(AF_INET, ips[c->topo.host(p)].c_str(), &sa.sin_addr);
        if (connect(fd, (sockaddr*)&sa, sizeof(sa)) == 0) break;
        close(fd); fd = -1; usleep(2000);
      }
      if (fd < 0) { MLOG_ERR("连不上 rank %d(%s:%d)", p, ips[c->topo.host(p)].c_str(), port_base + p); return -1; }
      set_sockopts(fd);
      uint8_t hello[4] = {(uint8_t)me, (uint8_t)ch, (uint8_t)job, 0};
      if (write(fd, hello, 4) != 4) { MLOG_ERR("hello 写失败"); return -1; }
      Conn* cn = new Conn();
      cn->fd = fd; cn->peer = p; cn->is_data = (ch == 1);
      (cn->is_data ? c->dconn : c->conn)[p] = cn;
      set_nonblock(fd);
      epoll_event ev{}; ev.events = EPOLLIN; ev.data.ptr = cn;
      epoll_ctl(c->epfd, EPOLL_CTL_ADD, fd, &ev);
      want_out--;
    }
  }

  // 3) 接受高 rank
  while (want_in > 0) {
    sockaddr_in ca{}; socklen_t cl = sizeof(ca);
    int fd = accept(c->listen_fd, (sockaddr*)&ca, &cl);
    if (fd < 0) { if (errno == EINTR) continue; MLOG_ERR("accept: %s", strerror(errno)); return -1; }
    uint8_t hello[4];
    size_t got = 0;
    while (got < 4) {
      ssize_t k = read(fd, hello + got, 4 - got);
      if (k <= 0) { MLOG_ERR("hello 读失败"); return -1; }
      got += k;
    }
    set_sockopts(fd);
    Conn* cn = new Conn();
    cn->fd = fd; cn->peer = hello[0]; cn->is_data = (hello[1] == 1);
    (cn->is_data ? c->dconn : c->conn)[cn->peer] = cn;
    set_nonblock(fd);
    epoll_event ev{}; ev.events = EPOLLIN; ev.data.ptr = cn;
    epoll_ctl(c->epfd, EPOLL_CTL_ADD, fd, &ev);
    want_in--;
  }

  // 4) 机内 SHM 环:每有序 pair 一条,发送方创建、接收方打开
  const int nslots = getenv("MOSAIC_SHM_SLOTS") ? atoi(getenv("MOSAIC_SHM_SLOTS")) : 16;
  for (int p = 0; p < n; p++) {
    if (p == me || !c->topo.same_host(me, p)) continue;
    char nm[128];
    snprintf(nm, sizeof(nm), "/mosaic_%d_%d_%d", job, me, p);
    c->tx_ring[p] = shm_ring_create(nm, nslots, c->cfg.slice_bytes, true, !c->cfg.nv_ipc);
    if (!c->tx_ring[p]) return -1;
  }
  for (int p = 0; p < n; p++) {
    if (p == me || !c->topo.same_host(me, p)) continue;
    char nm[128];
    snprintf(nm, sizeof(nm), "/mosaic_%d_%d_%d", job, p, me);
    c->rx_ring[p] = shm_ring_create(nm, nslots, c->cfg.slice_bytes, false, !c->cfg.nv_ipc);
    if (!c->rx_ring[p]) return -1;
  }
  MLOG_INFO("建链完成:%d ranks, G=%d, H=%d", n, c->topo.G, c->topo.H);
  return 0;
}

void net_teardown(Comm* c) {
  const int job = shm_job_id();      // 必须与 net_setup 同一命名(R19)
  for (auto* v : {&c->conn, &c->dconn})
    for (auto*& cn : *v)
      if (cn) { close(cn->fd); delete cn; cn = nullptr; }
  for (int p = 0; p < (int)c->tx_ring.size(); p++) {
    if (c->tx_ring[p]) {
      char nm[128]; snprintf(nm, sizeof(nm), "/mosaic_%d_%d_%d", job, c->cfg.rank, p);
      shm_ring_destroy(c->tx_ring[p], nm, true); c->tx_ring[p] = nullptr;
    }
    if (c->rx_ring[p]) {
      char nm[128]; snprintf(nm, sizeof(nm), "/mosaic_%d_%d_%d", job, p, c->cfg.rank);
      shm_ring_destroy(c->rx_ring[p], nm, false); c->rx_ring[p] = nullptr;
    }
  }
  if (c->listen_fd >= 0) close(c->listen_fd);
  if (c->epfd >= 0) close(c->epfd);
}

// ─────────────────────────────────────────────────────────── 控制面收发
// 调用前必须持有 c->ctrl_mu
static void conn_flush_ctrl_locked(Comm* c, Conn* cn) {
  while (!cn->wq_ctrl.empty()) {
    auto& m = cn->wq_ctrl.front();
    ssize_t k = write(cn->fd, m.data() + cn->wq_ctrl_off, m.size() - cn->wq_ctrl_off);
    if (k < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) return;
      MLOG_ERR("ctrl write → r%d: %s", cn->peer, strerror(errno));
      return;
    }
    cn->wq_ctrl_off += k;
    if (cn->wq_ctrl_off == m.size()) { cn->wq_ctrl.pop_front(); cn->wq_ctrl_off = 0; }
  }
}

// 已组帧报文直接走 TCP 队列(ctrl_send 的尾段;eRPC 回退路径也用它)
int ctrl_send_tcp(Comm* c, int peer, const void* framed, uint32_t n) {
  Conn* cn = c->conn[peer];
  if (!cn) return -1;
  std::vector<uint8_t> m((const uint8_t*)framed, (const uint8_t*)framed + n);
  std::lock_guard<std::mutex> lk(c->ctrl_mu);
  cn->wq_ctrl.push_back(std::move(m));
  conn_flush_ctrl_locked(c, cn);
  return 0;
}

int ctrl_send(Comm* c, int peer, CtrlType t, uint16_t aux, const void* body, uint32_t n) {
  Conn* cn = c->conn[peer];
  if (!cn) return -1;
  std::vector<uint8_t> m(sizeof(CtrlHdr) + n);
  CtrlHdr h{};
  h.len = (uint32_t)m.size(); h.type = (uint8_t)t;
  h.rank = (uint8_t)c->cfg.rank; h.aux = aux;
  memcpy(m.data(), &h, sizeof(h));
  if (n) memcpy(m.data() + sizeof(h), body, n);
  // O3(MOSAIC_CTRL=verbs):跨机 peer 的控制报文走 RDMA SEND,单程 ~5µs
  // (TCP 内核栈 ~30-60µs)。无槽/超长/建链中回退 TCP —— 报文自带长度分帧,
  // 两条通路交错到达也能正确解析(接收侧按 CtrlHdr 逐条处理,无流内状态)。
  if (c->ctrl_verbs && !c->topo.same_host(c->cfg.rank, peer) &&
      verbs_ctrl_send(c, peer, m.data(), (uint32_t)m.size()) == 0)
    return 0;
  return ctrl_send_tcp(c, peer, m.data(), (uint32_t)m.size());
}

int ctrl_bcast(Comm* c, CtrlType t, uint16_t aux, const void* body, uint32_t n) {
  for (int p = 0; p < c->cfg.nranks; p++)
    if (p != c->cfg.rank) ctrl_send(c, p, t, aux, body, n);
  return 0;
}

static void handle_ctrl_readable(Comm* c, Conn* cn) {
  uint8_t tmp[65536];
  for (;;) {
    ssize_t k = read(cn->fd, tmp, sizeof(tmp));
    if (k < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) break; MLOG_ERR("ctrl read r%d: %s", cn->peer, strerror(errno)); break; }
    if (k == 0) break;
    cn->rbuf.insert(cn->rbuf.end(), tmp, tmp + k);
    for (;;) {
      if (cn->rbuf.size() < sizeof(CtrlHdr)) break;
      CtrlHdr h;
      memcpy(&h, cn->rbuf.data(), sizeof(h));
      if (cn->rbuf.size() < h.len) break;
      on_ctrl_msg(c, cn->peer, h, cn->rbuf.data() + sizeof(CtrlHdr));
      cn->rbuf.erase(cn->rbuf.begin(), cn->rbuf.begin() + h.len);
    }
    if ((size_t)k < sizeof(tmp)) break;
  }
}

// 建链期的阻塞式小报文交换:经已有的 TCP 数据连接(fd 是非阻塞的,这里自旋等)。
// 双方都是「先写全、再读全」,报文 ≤ 几十字节,内核缓冲兜底,无死锁。
int net_blocking_exchange(Comm* c, int peer, const void* mine, void* remote, size_t n) {
  Conn* cn = c->dconn[peer];
  // same_host 的 peer 没有 dconn(net_setup 只给它建 ctrl 通道)—— 这里必须
  // 出声:R13 之前静默 return -1,上层只看到 E5016,查了一整个 job 才定位到
  if (!cn) { MLOG_ERR("exchange r%d:无数据连接(same_host peer 无 dconn)", peer); return -1; }
  size_t w = 0;
  while (w < n) {
    ssize_t k = write(cn->fd, (const uint8_t*)mine + w, n - w);
    if (k < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) { usleep(100); continue; }
                 MLOG_ERR("exchange write r%d: %s", peer, strerror(errno)); return -1; }
    w += k;
  }
  size_t r = 0;
  uint64_t dl = now_ns() + 30ull * 1000000000ull;
  while (r < n) {
    ssize_t k = read(cn->fd, (uint8_t*)remote + r, n - r);
    if (k < 0) { if (errno == EAGAIN || errno == EWOULDBLOCK) {
                   if (now_ns() > dl) { MLOG_ERR("exchange read r%d 超时", peer); return -1; }
                   usleep(100); continue; }
                 MLOG_ERR("exchange read r%d: %s", peer, strerror(errno)); return -1; }
    if (k == 0) { MLOG_ERR("exchange read r%d: 对端关闭", peer); return -1; }
    r += k;
  }
  return 0;
}

int net_poll(Comm* c) {
  int n = 0;
  if (c->verbs) n += verbs_poll(c);      // 数据面或 ctrl-only(O3)都要轮询
  if (c->use_nvsh) n += nvsh_poll(c);
  epoll_event evs[64];
  int k = epoll_wait(c->epfd, evs, 64, 0);
  for (int i = 0; i < k; i++) {
    Conn* cn = (Conn*)evs[i].data.ptr;
    if (evs[i].events & EPOLLIN) {
      // data 连接(bootstrap 专用)在引导后不该再有字节 —— 有就是协议错乱。
      // 必须吞掉(level-triggered epoll,不读会每圈重复触发)。
      if (cn->is_data) {
        uint8_t junk[4096];
        ssize_t got = read(cn->fd, junk, sizeof(junk));
        if (got > 0)
          MLOG_ERR("E1030 bootstrap 连接(r%d)收到运行期数据 %zdB(已丢弃)", cn->peer, got);
        continue;
      }
      handle_ctrl_readable(c, cn);
      n++;
    }
  }
  for (int p = 0; p < c->cfg.nranks; p++) {
    Conn* ct = c->conn[p];
    if (ct) {
      std::lock_guard<std::mutex> lk(c->ctrl_mu);
      if (!ct->wq_ctrl.empty()) { conn_flush_ctrl_locked(c, ct); n++; }
    }
  }
  return n;
}

}  // namespace mosaic
