// test_devpipe —— 多 wave 数据面 A/B/C 决定性微基准(2 PE,跨机,NVSHMEM)
//
// 目标(用户 2026-08-17):**减少多 wave 下的 overhead** —— 已归因每片固定开销
// ~145µs 里 94% 是主机往返(发现/交付/credit),kernel 启动只占 1%,PDL 在
// sm_89/sm_80 上不可用(griddepcontrol 需 sm_90)。本测验证替代路线:
//
//   A host3put : host `nvshmem_putmem_nbi`×3 + fence×2(现状 zerostream 发送侧)
//                + 接收侧 **主机 D2H 镜像轮询** 发现 + 主机回 credit
//                = 现状数据面的忠实缩影,每片每波都有主机在环里
//   C onstream : `nvshmemx_putmem_signal_nbi_on_stream`(默认档的融合发送)
//                + 接收侧设备核 `nvshmem_uint64_wait_until` + 设备回 credit
//                = 主机只在**入队时**出现,执行期零主机
//   B devfused : 设备核 `nvshmemx_putmem_signal_nbi_warp`(FAST 同款)
//                + 设备等待 + 设备 credit = 全程零主机、零流依赖
//
// 波语义:W 个 wave × K 片 × S 字节;wave n+1 的发送核在设备上等 credit>=n
// (接收方收齐本波所有 signal 后 SET credit)。B/C 把**全部 W 波一次性入队**,
// 运行期主机完全退出 —— 这正是「决策下发不阻塞上一波尾巴」的极限形态。
//
// 判据:B/C 在小片(64-256KB)多波下能否逼近 port rate(本机 100GbE ≈ 90+Gbps
// 可达,参照 T6 实测 87.8);A 预期塌陷。任何 size/wave 都行 —— 只看效果。
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define CUCHK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2);} } while (0)

static double now_us() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }

// ── 设备核 ────────────────────────────────────────────────────────────────
// 发送:每 warp 一片,put+signal 融合一笔(FAST flash_internode_p2p 同款原语)
__global__ void k_send_fused(uint8_t* dst, const uint8_t* src, size_t S,
                             uint64_t* sig, uint64_t val, int pe, int K) {
  int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32);
  if (w < K)
    nvshmemx_putmem_signal_nbi_warp(dst + (size_t)w * S, src + (size_t)w * S, S,
                                    sig + w, val, NVSHMEM_SIGNAL_SET, pe);
}
// 等一组 signal(接收侧发现 = 设备自旋;线程各自等不同槽)
__global__ void k_wait_sigs(uint64_t* sig, int K, uint64_t val) {
  for (int w = threadIdx.x; w < K; w += blockDim.x)
    nvshmem_uint64_wait_until(sig + w, NVSHMEM_CMP_GE, val);
}
// 等单个字(发送侧等 credit)
__global__ void k_wait_one(uint64_t* p, uint64_t val) {
  if (threadIdx.x == 0) nvshmem_uint64_wait_until(p, NVSHMEM_CMP_GE, val);
}
// 回 credit(SET 到对端)
__global__ void k_set_remote(uint64_t* p, uint64_t val, int pe) {
  if (threadIdx.x == 0) nvshmemx_signal_op(p, val, NVSHMEM_SIGNAL_SET, pe);
}

// ── TCP:换 UID + 每配置一次的 host barrier ──────────────────────────────
static int g_sock = -1;
static void tcp_setup(int rank, const char* peer_ip, int port) {
  if (rank == 0) {
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr*)&a, sizeof a) || listen(ls, 1)) { perror("bind/listen"); exit(2); }
    g_sock = accept(ls, nullptr, nullptr); close(ls);
  } else {
    g_sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, peer_ip, &a.sin_addr);
    for (int i = 0; i < 100; i++) { if (!connect(g_sock, (sockaddr*)&a, sizeof a)) break;
      usleep(200 * 1000); if (i == 99) { perror("connect"); exit(2); } }
  }
  int one = 1; setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}
static void tcp_xchg(void* buf, size_t n, int rank) {   // rank0 发,rank1 收
  if (rank == 0) { size_t o = 0; while (o < n) o += write(g_sock, (char*)buf + o, n - o); }
  else { size_t o = 0; while (o < n) { ssize_t r = read(g_sock, (char*)buf + o, n - o); if (r <= 0) exit(2); o += r; } }
}
static void tcp_barrier() { char c = 'B'; (void)!write(g_sock, &c, 1); (void)!read(g_sock, &c, 1); }

int main(int argc, char** argv) {
  int rank = -1, port = 29650, K = 32, W = 16, LAG = 0, BIDIR = 0;
  const char* peer_ip = "192.168.100.2";
  std::string sizes = "65536,262144,1048576", modes = "host,onstream,dev";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { return argv[++i]; };
    if (a == "--rank") rank = atoi(next());
    else if (a == "--peer") peer_ip = next();
    else if (a == "--port") port = atoi(next());
    else if (a == "--K") K = atoi(next());
    else if (a == "--W") W = atoi(next());
    else if (a == "--lag") LAG = atoi(next());   // 允许在飞的波数:0=严格串行,>=W=完全不 gate
    else if (a == "--bidir") BIDIR = 1;          // 双向:同一进程内既发又收(= 真实引擎形态,验坑 #49)
    else if (a == "--sizes") sizes = next();
    else if (a == "--modes") modes = next();
  }
  if (rank < 0) { fprintf(stderr, "用法: --rank 0|1 --peer <ip> [--K 32 --W 16 --sizes a,b,c --modes host,onstream,dev]\n"); return 1; }

  CUCHK(cudaSetDevice(0));
  tcp_setup(rank, peer_ip, port);
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  if (rank == 0) nvshmemx_get_uniqueid(&uid);
  tcp_xchg(&uid, sizeof uid, rank);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, 2, &uid, &attr);
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr)) { fprintf(stderr, "nvshmem init 失败\n"); return 2; }
  const int me = nvshmem_my_pe(), pe = 1 - me;

  const size_t SMAX = 1 << 20;
  uint8_t* ring = (uint8_t*)nvshmem_malloc((size_t)K * SMAX);      // 收侧落点(对称)
  uint8_t* src  = (uint8_t*)nvshmem_malloc((size_t)K * SMAX);      // 发侧源(对称=天然注册)
  uint64_t* sig  = (uint64_t*)nvshmem_calloc(2 * K, sizeof(uint64_t)); // 每片一个 signal(双向用 2K)
  uint64_t* cred = (uint64_t*)nvshmem_calloc(1, sizeof(uint64_t)); // 发侧等的 credit
  if (!ring || !src || !sig || !cred) { fprintf(stderr, "对称堆不够\n"); return 2; }
  CUCHK(cudaMemset(src, 0xA5 ^ me, (size_t)K * SMAX));
  uint64_t* sig_shadow; CUCHK(cudaHostAlloc(&sig_shadow, K * sizeof(uint64_t), 0));
  uint64_t* cred_shadow; CUCHK(cudaHostAlloc(&cred_shadow, sizeof(uint64_t), 0));
  cudaStream_t st; CUCHK(cudaStreamCreate(&st));
  nvshmem_barrier_all();

  std::vector<size_t> Ss; { char* d = strdup(sizes.c_str());
    for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) Ss.push_back(strtoull(t, nullptr, 10)); }
  std::vector<std::string> Ms; { char* d = strdup(modes.c_str());
    for (char* t = strtok(d, ","); t; t = strtok(nullptr, ",")) Ms.push_back(t); }

  if (me == 0) printf("# 2PE devpipe:K=%d 片/wave × W=%d wave;lag=允许在飞波数(0=串行);线速参照 T6=87.8Gbps\n", K, W);

  for (size_t S : Ss) {
    for (auto& M : Ms) {
      CUCHK(cudaMemset(sig, 0, 2 * K * sizeof(uint64_t)));
      CUCHK(cudaMemset(cred, 0, sizeof(uint64_t)));
      nvshmem_barrier_all(); tcp_barrier();
      double t0 = now_us();

      if (BIDIR) {
        // ── 双向:同一进程内 [on-stream 融合发送] + [设备自旋等待核] 共存
        // 这正是坑 #38/#49 说"自旋核在场则引擎的流整条楔死"的场景 —— 实测判定。
        // sig 分两半:低 K 个 = 我收的,高 K 个 = 对端收的(各自往对方的低半写)
        uint64_t* my_sig = sig;             // 我等这一半
        uint64_t* peer_sig = sig + K;       // 我写对端这一半(对称地址,写到 pe 上)
        if (me == 1) { my_sig = sig + K; peer_sig = sig; }
        for (int wv = 1; wv <= W; wv++) {
          if (wv - 1 - LAG > 0) k_wait_one<<<1, 32, 0, st>>>(cred, (uint64_t)(wv - 1 - LAG));
          if (M == "dev") {
            int thr = 256, blk = (K * 32 + thr - 1) / thr;
            k_send_fused<<<blk, thr, 0, st>>>(ring, src, S, peer_sig, wv, pe, K);
          } else {
            for (int w = 0; w < K; w++)
              nvshmemx_putmem_signal_nbi_on_stream(ring + (size_t)w * S, src + (size_t)w * S, S,
                                                   peer_sig + w, wv, NVSHMEM_SIGNAL_SET, pe, st);
          }
          k_wait_sigs<<<1, 256, 0, st>>>(my_sig, K, wv);      // 同流上的自旋等待核
          k_set_remote<<<1, 32, 0, st>>>(cred, wv, pe);
        }
        CUCHK(cudaStreamSynchronize(st));
        double dt = now_us() - t0;
        double gbps = (double)K * S * W * 8.0 / dt / 1000.0;
        if (me == 0)
          printf("  S=%7zu W=%2d K=%2d lag=%-2d %-9s %8.1f us (%6.1f us/wave) %6.2f Gbps [双向]\n",
                 S, W, K, LAG, M.c_str(), dt, dt / W, gbps);
        fflush(stdout);
      } else if (me == 0) {                                   // ── 发送方
        if (M == "host") {
          // 现状缩影:每片 3 笔 host put + 2 fence;每波后 quiet;等 credit 靠 D2H 轮询
          for (int wv = 1; wv <= W; wv++) {
            while (true) {                                    // 等上一波 credit(主机轮询)
              CUCHK(cudaMemcpy(cred_shadow, cred, 8, cudaMemcpyDeviceToHost));
              if ((long long)*cred_shadow >= (long long)wv - 1 - LAG) break;
            }
            for (int w = 0; w < K; w++) {
              uint8_t* d = ring + (size_t)w * S; const uint8_t* s = src + (size_t)w * S;
              nvshmem_putmem_nbi(d + 72, s + 72, S - 72, pe);  // ① payload
              nvshmem_fence();
              nvshmem_putmem_nbi(d, s, 64, pe);                // ② "hdr" 64B
              nvshmem_fence();
              nvshmem_uint64_p(sig + w, (uint64_t)wv, pe);     // ③ sig 8B(现状同款三笔)
            }
            nvshmem_quiet();                                   // 每波一次批量 flush(坑 #39 纪律)
          }
        } else if (M == "onstream") {
          // 全部 W 波一次性入队:[等 credit 核][K 笔 on-stream 融合 put] × W
          for (int wv = 1; wv <= W; wv++) {
            if (wv - 1 - LAG > 0) k_wait_one<<<1, 32, 0, st>>>(cred, (uint64_t)(wv - 1 - LAG));
            for (int w = 0; w < K; w++)
              nvshmemx_putmem_signal_nbi_on_stream(ring + (size_t)w * S, src + (size_t)w * S, S,
                                                   sig + w, wv, NVSHMEM_SIGNAL_SET, pe, st);
          }
          CUCHK(cudaStreamSynchronize(st));
        } else {                                              // dev
          for (int wv = 1; wv <= W; wv++) {
            if (wv - 1 - LAG > 0) k_wait_one<<<1, 32, 0, st>>>(cred, (uint64_t)(wv - 1 - LAG));
            int thr = 256, blk = (K * 32 + thr - 1) / thr;
            k_send_fused<<<blk, thr, 0, st>>>(ring, src, S, sig, wv, pe, K);
          }
          CUCHK(cudaStreamSynchronize(st));
        }
        // 终态:等最后一波 credit 到手才算完(计时含收侧确认)
        while (true) { CUCHK(cudaMemcpy(cred_shadow, cred, 8, cudaMemcpyDeviceToHost));
          if (*cred_shadow >= (uint64_t)W) break; }
        double dt = now_us() - t0;
        double gbps = (double)K * S * W * 8.0 / dt / 1000.0;
        printf("  S=%7zu W=%2d K=%2d lag=%-2d %-9s %8.1f us (%6.1f us/wave) %6.2f Gbps\n",
               S, W, K, LAG, M.c_str(), dt, dt / W, gbps);
        fflush(stdout);
      } else {                                                // ── 接收方
        if (M == "host") {
          for (int wv = 1; wv <= W; wv++) {
            while (true) {                                    // 主机 D2H 镜像轮询(现状发现路径)
              CUCHK(cudaMemcpy(sig_shadow, sig, K * sizeof(uint64_t), cudaMemcpyDeviceToHost));
              bool all = true;
              for (int w = 0; w < K; w++) if (sig_shadow[w] < (uint64_t)wv) { all = false; break; }
              if (all) break;
            }
            nvshmem_uint64_p(cred, (uint64_t)wv, pe); nvshmem_quiet();  // 主机回 credit
          }
        } else {
          for (int wv = 1; wv <= W; wv++) {                   // 设备发现 + 设备回 credit
            k_wait_sigs<<<1, 256, 0, st>>>(sig, K, wv);
            k_set_remote<<<1, 32, 0, st>>>(cred, wv, pe);
          }
          CUCHK(cudaStreamSynchronize(st));
        }
      }
      nvshmem_barrier_all(); tcp_barrier();
    }
  }
  // 轻校验:收侧末波内容 = 发侧 pattern
  if (me == 1) {
    std::vector<uint8_t> h(4096);
    CUCHK(cudaMemcpy(h.data(), ring, h.size(), cudaMemcpyDeviceToHost));
    int bad = 0; for (auto b : h) if (b != (0xA5 ^ 0)) bad++;
    printf("  verify(首 4KB): %s(bad=%d)\n", bad ? "FAILED" : "ok", bad);
  }
  nvshmem_barrier_all();
  nvshmem_free(ring); nvshmem_free(src); nvshmem_free(sig); nvshmem_free(cred);
  nvshmem_finalize(); close(g_sock);
  return 0;
}
