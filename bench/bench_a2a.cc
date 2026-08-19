/* bench_a2a.cc —— 三方案共用的 alltoallv 基准 harness(Imp_Plan/04 §3)。
 *
 * 同一份 demand 生成、同一套计时、同一套 bit-exact 校验、同样的预热与重复次数,
 * 只有 --sched 一处分叉 —— 公平性的技术落点就在"harness 只有一份"。
 *
 * demand 语义对齐模拟器 tools/gen_nonuniform_a2a.py:
 *   weight[d] = 1/(rank[d]+1)^alpha,rank 来自种子化的置换;
 *   每 src 总量恒 = (n-1)*msg_bytes(与 uniform 契约逐字节相同)。
 */
#include "mosaic/mosaic.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t xs(uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

// 与 python 版同构的矩阵构造(RNG 不同,但所有 rank 用同一种子 ⇒ 全局一致)
static std::vector<uint64_t> build_demand(int n, uint64_t msg, double alpha, int seed,
                                          const std::string& mode, uint64_t gran) {
  std::vector<uint64_t> m((size_t)n * n, 0);
  // ── mode=flashuniform:**逐字节复刻官方 FAST 的 workload**(公平性铁律的解法)。
  // 背景:三套完整方案对比要求 FAST 用它自己的 libflash 全栈跑,而
  // `flash_tester.py → test_flash_alltoall` 的 demand 是它**内部**生成的
  // (`alltoall_nvshmem.cpp:565` → `uniform_distribution(w, rank_n, pow(2,23))`),
  // 我们**不能改它的源码**去喂自己的 demand。于是反过来:让我们复刻它的分布。
  // 官方那段的语义(alltoall_nvshmem.cpp:111-123 + alltoall_define.h:51):
  //   for i,j:  w[i][j] = mem_512B_align(rand() % 2^23)      // 对角线也消耗 rand()
  //   for i:    w[i][i] = 0                                   // 之后才清对角
  // rank0 生成后 broadcast ⇒ 全员同一份。它**不调 srand** ⇒ 用 glibc 默认种子 1;
  // 我们同机同 libc 调同样的 rand() 序列,得到的矩阵**逐字节相同**。
  // 注意:对角线必须"先取随机数再清零",少一次 rand() 调用整条序列就错位。
  if (mode == "flashuniform") {
    const uint64_t bound = 1ull << 23;            // 官方 pow(2,23)
    srand(1);                                     // glibc 默认种子,显式写出以免依赖隐式初值
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        // 与官方 `mem_512B_align(x) = ((x + 0x1ff) & 0xfff…e00)` 逐位同形(向上取整到 512B)。
        // **陷阱**:官方那个数组叫 `workload_in_bytes`,但它存的是**元素个数**不是字节 ——
        // `alltoall_nvshmem.cpp:569,571` 才 `buff_size = sum × data_size`,`data_size = 8`。
        // 少乘这个 8,两边搬的字节数会差 8 倍(核对:56 对 × 均值 4.19M × 8 ÷ 8 rank
        // ≈ 234.6MB,与官方自报的 per-GPU 229.677MB 吻合)。
        m[(size_t)i * n + j] =
            (((uint64_t)(rand() % bound) + 0x1ffull) & 0xfffffffffffffe00ull) * 8ull;
    for (int i = 0; i < n; i++) m[(size_t)i * n + i] = 0;
    return m;
  }
  std::vector<int> perm(n);
  for (int i = 0; i < n; i++) perm[i] = i;
  uint64_t s = 0x9E3779B97F4A7C15ull ^ (uint64_t)(seed + 1);
  for (int i = n - 1; i > 0; i--) std::swap(perm[i], perm[xs(s) % (i + 1)]);
  std::vector<int> rank_of(n);
  for (int r = 0; r < n; r++) rank_of[perm[r]] = r;

  std::vector<double> w(n);
  auto set_w = [&](const std::vector<int>& rk) {
    for (int d = 0; d < n; d++) {
      if (mode == "khot") w[d] = (rk[d] < std::max(1, (int)alpha)) ? 8.0 : 1.0;
      else if (mode == "uniform") w[d] = 1.0;
      else w[d] = 1.0 / std::pow((double)rk[d] + 1.0, alpha);
    }
  };
  set_w(rank_of);

  const uint64_t total_per_src = (uint64_t)(n - 1) * msg;
  for (int src = 0; src < n; src++) {
    if (mode == "persrc") {                       // 每 src 独立排名
      std::vector<int> sp(n);
      for (int i = 0; i < n; i++) sp[i] = i;
      for (int i = n - 1; i > 0; i--) std::swap(sp[i], sp[xs(s) % (i + 1)]);
      std::vector<int> rk(n);
      for (int r = 0; r < n; r++) rk[sp[r]] = r;
      set_w(rk);
    }
    double wsum = 0;
    for (int d = 0; d < n; d++) if (d != src) wsum += w[d];
    uint64_t assigned = 0;
    int hottest = -1;
    for (int d = 0; d < n; d++) {
      if (d == src) continue;
      double raw = (double)total_per_src * w[d] / wsum;
      uint64_t b = std::max(gran, (uint64_t)(raw / gran) * gran);
      m[(size_t)src * n + d] = b;
      assigned += b;
      if (hottest < 0 || w[d] > w[hottest]) hottest = d;
    }
    if (assigned < total_per_src && hottest >= 0)   // 余数给最热的,保持总量契约
      m[(size_t)src * n + hottest] += total_per_src - assigned;
  }
  return m;
}

// 读模拟器格式的 demand 文件(node_count N / demand s d bytes)——
// 与模拟器对表时必须用同一文件,两边矩阵逐字节一致(T11)
static std::vector<uint64_t> load_demand(const char* path, int n_expect) {
  FILE* f = fopen(path, "r");
  if (!f) { fprintf(stderr, "demand 文件打不开:%s\n", path); exit(2); }
  int n = 0;
  std::vector<uint64_t> m;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#') continue;
    int s, d; long long b;
    if (sscanf(line, "node_count %d", &n) == 1) m.assign((size_t)n * n, 0);
    else if (sscanf(line, "demand %d %d %lld", &s, &d, &b) == 3 && n)
      m[(size_t)s * n + d] = (uint64_t)b;
  }
  fclose(f);
  if (n != n_expect) { fprintf(stderr, "demand n=%d != nranks=%d\n", n, n_expect); exit(2); }
  return m;
}

static uint8_t pat(int src, int dst, uint64_t off) {
  return (uint8_t)(src * 31 + dst * 17 + (off * 7) + ((off >> 8) * 13) + ((off >> 16) * 3));
}

int main(int argc, char** argv) {
  mosaicConfig_t cfg;
  mosaicConfigDefault(&cfg);

  std::string mode = "hotdst", tag = "run", demand_file;
  int dump_demand = 0;
  double alpha = 1.0;
  int seed = 42, iters = 10, warmup = 3, verify = 1;
  uint64_t msg = 1 << 20;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return std::string(argv[++i]); };
    if (a == "--mode") mode = nx();
    else if (a == "--alpha") alpha = atof(nx().c_str());
    else if (a == "--seed") seed = atoi(nx().c_str());
    else if (a == "--msg") msg = strtoull(nx().c_str(), nullptr, 0);
    else if (a == "--iters") iters = atoi(nx().c_str());
    else if (a == "--warmup") warmup = atoi(nx().c_str());
    else if (a == "--verify") verify = atoi(nx().c_str());
    else if (a == "--tag") tag = nx();
    else if (a == "--sched") {
      std::string s = nx();
      cfg.sched = (s == "mosaic") ? MOSAIC_SCHED_MOSAIC
                : (s == "fast") ? MOSAIC_SCHED_FAST
                : (s == "fast_struct") ? MOSAIC_SCHED_FAST_STRUCT
                : MOSAIC_SCHED_DIRECT;
    }
    else if (a == "--corridors") cfg.corridors = atoi(nx().c_str());
    else if (a == "--waves") cfg.waves = atoi(nx().c_str());
    else if (a == "--nv") cfg.nv_gbps = atof(nx().c_str());
    else if (a == "--nic") cfg.nic_gbps = atof(nx().c_str());
    else if (a == "--ledger") cfg.ledger = atoi(nx().c_str());
    else if (a == "--bcast") cfg.bcast = atoi(nx().c_str());
    else if (a == "--dump-demand") dump_demand = 1;
    else if (a == "--demand-file") demand_file = nx();
  }
  // --verify 只控制**端到端字节校验**(it==warmup 那次,对照 pat() 逐字节比对,
  // 这就是 bit-exact 判据)。逐 slice 的 CRC 是独立的 debug 旋钮
  // MOSAIC_SLICE_CRC=1(默认关):它在单事件线程热路径上逐 slice 计算,即便
  // 换了硬件 crc32c 也要 ~3GB/s 的税,开着会把计时口径压低(log.md 坑 #14)。
  cfg.verify = getenv("MOSAIC_SLICE_CRC") ? atoi(getenv("MOSAIC_SLICE_CRC")) : 0;

  const int n = cfg.nranks, me = cfg.rank;
  auto D = demand_file.empty() ? build_demand(n, msg, alpha, seed, mode, 4096)
                               : load_demand(demand_file.c_str(), n);
  if (!demand_file.empty())
    for (int i = 0; i < n; i++) D[(size_t)i * n + i] = 0;   // 对表口径:无 self 流量

  if (dump_demand) {   // 供 tools/hw_bounds.py 算割下界(保证与执行用的矩阵逐字节一致)
    printf("# n=%d G=%d msg=%llu mode=%s alpha=%.2f seed=%d nv=%.3f nic=%.3f\n",
           n, cfg.gpus_per_host, (unsigned long long)msg, mode.c_str(), alpha, seed,
           cfg.nv_gbps, cfg.nic_gbps);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) printf("%llu%s", (unsigned long long)D[(size_t)i * n + j], j + 1 < n ? "," : "\n");
    }
    return 0;
  }

  std::vector<size_t> sc(n), sd(n), rc(n), rd(n);
  size_t so = 0, ro = 0;
  for (int d = 0; d < n; d++) { sc[d] = D[(size_t)me * n + d]; sd[d] = so; so += sc[d]; }
  for (int s = 0; s < n; s++) { rc[s] = D[(size_t)s * n + me]; rd[s] = ro; ro += rc[s]; }
  const size_t sbytes = so, rbytes = ro;

  if (cfg.use_gpu) {
    // 必须与 engine_start 同一选卡逻辑(rank % G,MOSAIC_CUDA_DEV 覆盖):
    // sbuf/rbuf 在 CommInit 之前分配,选错卡就和引擎不同设备(R11)。
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    int dev = me % cfg.gpus_per_host;
    if (ndev > 0) dev %= ndev;   // 卡不够分回退共享(本机 1 卡 G=4 烟测)
    if (const char* e = getenv("MOSAIC_CUDA_DEV")) { if (*e) dev = atoi(e); }
    cudaSetDevice(dev);
  }
  void *sbuf = nullptr, *rbuf = nullptr;
  std::vector<uint8_t> hs(sbytes), hr(rbytes);
  for (int d = 0; d < n; d++)
    for (size_t k = 0; k < sc[d]; k++) hs[sd[d] + k] = pat(me, d, k);
  if (cfg.use_gpu) {
    cudaMalloc(&sbuf, sbytes ? sbytes : 1);
    cudaMalloc(&rbuf, rbytes ? rbytes : 1);
    cudaMemcpy(sbuf, hs.data(), sbytes, cudaMemcpyHostToDevice);
    cudaMemset(rbuf, 0, rbytes);
  } else {
    sbuf = malloc(sbytes ? sbytes : 1);
    rbuf = calloc(1, rbytes ? rbytes : 1);
    memcpy(sbuf, hs.data(), sbytes);
  }

  mosaicComm_t comm;
  if (mosaicCommInit(&comm, &cfg) != mosaicSuccess) { fprintf(stderr, "init 失败\n"); return 1; }

  std::vector<double> times;
  double makespan_best = 1e18;
  for (int it = 0; it < warmup + iters; it++) {
    // cudaMemset 是**异步**的(null stream),不同步就直接 barrier 的话,它可能在
    // 对端数据已经落到 recvbuf 之后才执行 → 把收到的数据清成 0。
    // 症状极具误导性:交付字节数完全正确、内容全 0。必须先同步再进 barrier。
    if (cfg.use_gpu) { cudaMemset(rbuf, 0, rbytes); cudaDeviceSynchronize(); }
    mosaicBarrier(comm);
    mosaicResult_t r = mosaicAlltoallv(comm, sbuf, sc.data(), sd.data(), rbuf, rc.data(), rd.data());
    if (r != mosaicSuccess) {
      fprintf(stderr, "rank %d alltoallv 失败 rc=%d: %s\n", me, r, mosaicGetLastError(comm));
      return 2;
    }
    mosaicStats_t st;
    mosaicStatsSync(comm, &st);
    if (it >= warmup) { times.push_back(st.makespan_us); makespan_best = std::min(makespan_best, st.makespan_us); }

    if (verify && it == warmup) {          // 逐字节校验一次(带校验的那次不进计时统计口径)
      if (cfg.use_gpu) cudaMemcpy(hr.data(), rbuf, rbytes, cudaMemcpyDeviceToHost);
      else memcpy(hr.data(), rbuf, rbytes);
      size_t bad = 0;
      for (int s = 0; s < n; s++)
        for (size_t k = 0; k < rc[s]; k++)
          if (hr[rd[s] + k] != pat(s, me, k)) { if (bad < 4) fprintf(stderr,
              "  E3001 rank%d: (%d→%d) off=%zu 期望 %02x 实得 %02x\n", me, s, me, k,
              pat(s, me, k), hr[rd[s] + k]); bad++; }
      if (bad) { fprintf(stderr, "rank %d 校验 FAILED(%zu 字节错)\n", me, bad); return 3; }
    }
  }

  std::sort(times.begin(), times.end());
  double med = times.empty() ? 0 : times[times.size() / 2];
  double p99 = times.empty() ? 0 : times[(size_t)(times.size() * 0.99) % times.size()];
  mosaicStats_t st;
  mosaicGetStats(comm, &st);

  {   // 资源实际通过率(核验限速器是否真的绑定 —— 见 log.md 坑 #10)
    const char* rn[5] = {"NICup", "NICdn", "NVup", "NVdn", "SPINE"};
    double t = st.completion_us;
    fprintf(stderr, "[rank %d] completion=%.1fus 各资源实测速率(Gbps):", me, t);
    for (int k = 0; k < 5; k++)
      fprintf(stderr, " %s=%.2f", rn[k], t > 0 ? st.res_passed[k] * 8.0 / (t * 1000.0) : 0.0);
    fprintf(stderr, "  额定 nv=%.1f nic=%.1f\n", cfg.nv_gbps, cfg.nic_gbps);
    if (cfg.sched == MOSAIC_SCHED_MOSAIC) {
      const char* cn[6] = {"DIRECT", "RX", "TX", "SP0", "SP1", "SP2"};
      fprintf(stderr, "[rank %d] 入向分配(MB):", me);
      for (int k = 0; k < 6; k++) fprintf(stderr, " %s=%.2f", cn[k], st.ingress_corr[k] / 1e6);
      fprintf(stderr, "\n");
    }
  }
  if (me == 0) {
    printf("tag,sched,n,G,mode,alpha,seed,msg,nv_gbps,nic_gbps,C,W,"
           "med_us,min_us,p99_us,bytes_sent,bytes_relay,plan_us_avg,plan_us_max,waves,"
           "chA_msgs,chB_msgs,chB_bytes,c_direct,c_rx,c_tx,c_sp0,c_sp1,c_sp2,in_direct,in_rx,in_tx,in_sp0,in_sp1,in_sp2,sched_plan_us,chA_bytes\n");
    printf("%s,%s,%d,%d,%s,%.1f,%d,%llu,%.2f,%.2f,%d,%d,"
           "%.1f,%.1f,%.1f,%llu,%llu,%.2f,%.2f,%llu,%llu,%llu,%llu,"
           "%llu,%llu,%llu,%llu,%llu,%llu,"
           "%llu,%llu,%llu,%llu,%llu,%llu,%.1f,%llu\n",
           tag.c_str(), mosaicSchedName(cfg.sched), n, cfg.gpus_per_host, mode.c_str(), alpha, seed,
           (unsigned long long)msg, cfg.nv_gbps, cfg.nic_gbps, cfg.corridors, cfg.waves,
           med, makespan_best, p99,
           (unsigned long long)st.bytes_sent, (unsigned long long)st.bytes_relayed,
           st.plan_calls ? st.plan_us_total / st.plan_calls : 0.0, st.plan_us_max,
           (unsigned long long)st.waves_emitted,
           (unsigned long long)st.chA_msgs, (unsigned long long)st.chB_msgs,
           (unsigned long long)st.chB_bytes,
           (unsigned long long)st.corridor_bytes[0], (unsigned long long)st.corridor_bytes[1],
           (unsigned long long)st.corridor_bytes[2], (unsigned long long)st.corridor_bytes[3],
           (unsigned long long)st.corridor_bytes[4], (unsigned long long)st.corridor_bytes[5],
           (unsigned long long)st.ingress_corr[0], (unsigned long long)st.ingress_corr[1],
           (unsigned long long)st.ingress_corr[2], (unsigned long long)st.ingress_corr[3],
           (unsigned long long)st.ingress_corr[4], (unsigned long long)st.ingress_corr[5],
           st.sched_plan_us, (unsigned long long)st.chA_bytes);
    fflush(stdout);
  }
  mosaicCommDestroy(comm);
  return 0;
}
