/* decider_abc_test —— 决策者/视野/顺序 的离线判别(纯 CPU 决策层,零通信)。
 *
 * 动机(File/log_raven.md `local-0823-decider-abc`):按阶段反转策略决定者之前,
 * 先在离线注水层回答两个问题:①dst 驱动的贪心顺序重不重要;②src 驱动(行侧
 * 注水)对 MoE dispatch 形状(行全等、列深斜)是否结构性更优。
 *
 * 四档(全部单波全额注水,共享同一 demand):
 *   A0  dst 决策,rank 序,**残缺视野** —— 复刻现状 apply_policy:每个 rank 对
 *       发布者 p 的 charges 只看得到「自己是 src」的那部分条目;
 *   A   dst 决策,rank 序,全局视野(任务 1 修复后:policy 广播给所有 rank);
 *   B   dst 决策,热度(列和)降序,全局视野;
 *   C   src 决策,rank 序,全局视野(行侧注水,与 lg_split 逐字同参:24 步、
 *       1024 地板、0.95/1.05 带、带内短链优先)。
 *
 * A0/A/B 走生产代码 engine_test_emit_wave(注水单一来源);C 是行侧的同构贪心。
 * 判据:①makespan 代理 = max_r outstanding_r/cap_r;②公平性 = 各 (s,d) pair
 * 投影完成时间的 max/median 与 CoV(所有片完成时间一致才最公平);③中转膨胀 =
 * Σ len×(nhops-1)。
 *
 * 用法:decider_abc_test <demand_file(sim 格式)> [nv nic spine]
 */
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace mosaic;

static int g_n = 0;

static std::string res_name(const Topo& t, int r) {
  char b[48];
  static const char* K[] = {"NIC^", "NICv", "NV^", "NVv", "SP^"};
  if (r < t.nranks * RES_KINDS)
    snprintf(b, sizeof(b), "r%d.%s", r / RES_KINDS, K[r % RES_KINDS]);
  else if (r < t.nranks * RES_KINDS + t.G)
    snprintf(b, sizeof(b), "fab^%d", r - t.nranks * RES_KINDS);
  else
    snprintf(b, sizeof(b), "fabv%d", r - t.nranks * RES_KINDS - t.G);
  return b;
}

struct TierResult {
  std::vector<uint64_t> total;                        // [res] 全局 charge
  std::map<std::pair<int,int>, std::array<uint64_t, NCORR>> alloc;  // (s,d)→per-corr
};

static void report(const char* name, Comm* c, const TierResult& tr) {
  const Topo& t = c->topo;
  const int nres = t.nres();
  std::vector<double> T(nres, 0);
  for (int r = 0; r < nres; r++) {
    double cap = c->ledger.cap_Bps[r];
    T[r] = cap > 0 ? (double)tr.total[r] / cap * 1e6 : 0;   // µs
  }
  double tmax = 0; int argmax = 0;
  for (int r = 0; r < nres; r++) if (T[r] > tmax) { tmax = T[r]; argmax = r; }
  // 每 pair 投影完成 = 其占用走廊里最忙资源的投影时间的最大值
  std::vector<double> pf;
  uint64_t relay_bytes = 0;
  for (auto& kv : tr.alloc) {
    int s = kv.first.first, d = kv.first.second;
    double f = 0;
    for (int cc = 0; cc < NCORR; cc++) {
      if (!kv.second[cc]) continue;
      Chain ch;
      if (!t.chain_of(s, d, cc, &ch)) continue;
      relay_bytes += kv.second[cc] * (uint64_t)(ch.nhops - 1);
      int res[16]; int nr = t.chain_res(ch, res);
      for (int i = 0; i < nr; i++) f = std::max(f, T[res[i]]);
    }
    pf.push_back(f);
  }
  std::vector<double> sp = pf;
  std::sort(sp.begin(), sp.end());
  double med = sp.empty() ? 0 : sp[sp.size() / 2];
  double mean = 0, var = 0;
  for (double v : pf) mean += v;
  mean /= std::max<size_t>(1, pf.size());
  for (double v : pf) var += (v - mean) * (v - mean);
  double cov = mean > 0 ? std::sqrt(var / std::max<size_t>(1, pf.size())) / mean : 0;
  printf("%-3s makespan_proxy %8.1fus @%-8s pair_finish max/med %5.2f cov %.3f relayMB %6.1f\n",
         name, tmax, res_name(t, argmax).c_str(),
         med > 0 ? sp.back() / med : 0, cov, relay_bytes / 1048576.0);
  // 热列的入口拆分(NICv vs NVv)——「hot dst 是否被欠配」的直接观测
  std::vector<uint64_t> col(g_n, 0);
  for (auto& kv : tr.alloc)
    for (int cc = 0; cc < NCORR; cc++) col[kv.first.second] += kv.second[cc];
  int hot = (int)(std::max_element(col.begin(), col.end()) - col.begin());
  uint64_t nic_in = tr.total[t.res(hot, RES_NIC_DN)], nv_in = tr.total[t.res(hot, RES_NV_DN)];
  printf("    hot=d%d ingress NICv %5.1fMB (%.1fus) NVv %5.1fMB (%.1fus)\n", hot,
         nic_in / 1048576.0, T[t.res(hot, RES_NIC_DN)],
         nv_in / 1048576.0, T[t.res(hot, RES_NV_DN)]);
  // top-5 资源负载
  std::vector<int> idx(nres);
  for (int r = 0; r < nres; r++) idx[r] = r;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return T[a] > T[b]; });
  printf("    top:");
  for (int i = 0; i < 5 && i < nres; i++)
    printf(" %s=%.1f", res_name(t, idx[i]).c_str(), T[idx[i]]);
  printf("\n");
}

// dst 驱动档(A0/A/B):order = 决策序;full_view = 通道 B 是否全局广播
static TierResult run_dst_tier(Comm* c, const std::vector<uint64_t>& D,
                               const std::vector<int>& order, bool full_view) {
  const int n = g_n, nres = c->topo.nres();
  TierResult tr;
  tr.total.assign(nres, 0);
  std::vector<std::vector<uint64_t>> chg_by(n, std::vector<uint64_t>(nres, 0));
  // 残缺视野的账目:发布者 p 的 charges 里、src=q 的那部分(A0 还原现状用)
  std::vector<std::vector<std::vector<uint64_t>>> chg_by_src(
      n, std::vector<std::vector<uint64_t>>(n, std::vector<uint64_t>(nres, 0)));
  std::vector<uint64_t> zero(nres, 0);
  for (int d : order) {
    topo_init(&c->topo, n, c->cfg.gpus_per_host, d);
    c->cfg.rank = d;
    // 装配 d 的账本视图
    c->ledger.charge = chg_by[d];
    std::fill(c->ledger.discharge.begin(), c->ledger.discharge.end(), 0);
    for (int p = 0; p < n; p++)
      c->ledger.set_peer_charge(
          p, p == d ? zero.data()
                    : (full_view ? chg_by[p].data() : chg_by_src[p][d].data()),
          nres);
    c->dst = DstCtrl();
    c->dst.active = true;
    for (int s = 0; s < n; s++)
      if (s != d) c->dst.remaining[s] = D[(size_t)s * n + d];
    engine_test_emit_wave(c);
    chg_by[d] = c->ledger.charge;                 // 含本波新 charges
    for (auto& e : c->dbg_last_ents) {
      Chain ch;
      if (!c->topo.chain_of(e.src, d, e.corr, &ch)) continue;
      int res[16]; int nr = c->topo.chain_res(ch, res);
      for (int i = 0; i < nr; i++) chg_by_src[d][e.src][res[i]] += e.len;
      tr.alloc[{(int)e.src, d}][e.corr] += e.len;
    }
  }
  for (int p = 0; p < n; p++)
    for (int r = 0; r < nres; r++) tr.total[r] += chg_by[p][r];
  return tr;
}

// src 驱动档(C):走生产代码 engine_test_emit_wave_src(注水与 dst 档同一
// 来源 waterfill_pair)。全局视野 = 共享账本(bcast=0 回放语义,rank 序)。
static TierResult run_src_tier(Comm* c, const std::vector<uint64_t>& D) {
  const int n = g_n, nres = c->topo.nres();
  TierResult tr;
  std::vector<uint64_t> zero(nres, 0);
  std::fill(c->ledger.charge.begin(), c->ledger.charge.end(), 0);
  std::fill(c->ledger.discharge.begin(), c->ledger.discharge.end(), 0);
  for (int p = 0; p < n; p++) c->ledger.set_peer_charge(p, zero.data(), nres);
  for (int s = 0; s < n; s++) {
    topo_init(&c->topo, n, c->cfg.gpus_per_host, s);
    c->cfg.rank = s;
    c->srcc = SrcCtrl();
    c->srcc.active = true;
    for (int d = 0; d < n; d++)
      if (d != s) c->srcc.remaining[d] = D[(size_t)s * n + d];
    engine_test_emit_wave_src(c);
    for (auto& e : c->dbg_last_ents)           // src 波条目:e.src 承载 dst 端点
      tr.alloc[{s, (int)e.src}][e.corr] += e.len;
  }
  tr.total = c->ledger.charge;
  return tr;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "用法:decider_abc_test demand [nv nic spine]\n"); return 2; }
  double nv = argc > 2 ? atof(argv[2]) : 400.0;
  double nic = argc > 3 ? atof(argv[3]) : 100.0;
  double spine = argc > 4 ? atof(argv[4]) : 180.0;

  FILE* f = fopen(argv[1], "r");
  if (!f) { perror("demand"); return 2; }
  int n = 0;
  std::vector<uint64_t> D;
  char line[256];
  while (fgets(line, sizeof(line), f)) {
    if (line[0] == '#') continue;
    int s, d; long long b;
    if (sscanf(line, "node_count %d", &n) == 1) { D.assign((size_t)n * n, 0); continue; }
    if (sscanf(line, "demand %d %d %lld", &s, &d, &b) == 3 && n) D[(size_t)s * n + d] = b;
  }
  fclose(f);
  if (!n) { fprintf(stderr, "demand 解析失败\n"); return 2; }
  g_n = n;

  const int G = 4;
  Comm* c = new Comm();
  mosaicConfigDefault(&c->cfg);
  c->cfg.nranks = n; c->cfg.gpus_per_host = G;
  c->cfg.corridors = 6; c->cfg.waves = 1;         // 单波全额:比的是注水结构
  c->cfg.ledger = 1; c->cfg.bcast = 0;
  c->cfg.use_gpu = 0; c->cfg.sched = MOSAIC_SCHED_MOSAIC;
  c->cfg.nv_gbps = nv; c->cfg.nic_gbps = nic; c->cfg.spine_gbps = spine;
  c->cfg.detector = 0;
  c->scounts.assign(n, 0); c->sdispls.assign(n, 0);
  c->rcounts.assign(n, 0); c->rdispls.assign(n, 0);
  c->demand = D;
  topo_init(&c->topo, n, G, 0);
  ledger_init(c);

  std::vector<int> rank_order(n), hot_order(n);
  for (int i = 0; i < n; i++) rank_order[i] = hot_order[i] = i;
  std::vector<uint64_t> col(n, 0);
  for (int s = 0; s < n; s++)
    for (int d = 0; d < n; d++) if (s != d) col[d] += D[(size_t)s * n + d];
  std::sort(hot_order.begin(), hot_order.end(),
            [&](int a, int b) { return col[a] > col[b]; });

  printf("n=%d G=%d nv=%.0f nic=%.0f spine=%.0f cols(MB):", n, G, nv, nic, spine);
  for (int d = 0; d < n; d++) printf(" %llu", (unsigned long long)(col[d] >> 20));
  printf("\n");

  TierResult a0 = run_dst_tier(c, D, rank_order, /*full_view=*/false);
  TierResult a  = run_dst_tier(c, D, rank_order, /*full_view=*/true);
  TierResult b  = run_dst_tier(c, D, hot_order,  /*full_view=*/true);
  TierResult cc = run_src_tier(c, D);
  report("A0", c, a0);
  report("A", c, a);
  report("B", c, b);
  report("C", c, cc);

  // C 与 A 的逐 pair 分配分歧(字节搬移量)——决策差异有多大
  uint64_t moved = 0, total = 0;
  for (auto& kv : a.alloc) {
    auto it = cc.alloc.find(kv.first);
    for (int k = 0; k < NCORR; k++) {
      uint64_t va = kv.second[k], vc = it == cc.alloc.end() ? 0 : it->second[k];
      moved += va > vc ? va - vc : vc - va;
      total += va;
    }
  }
  printf("C-vs-A 分配分歧:%.1f%%(%.1fMB / %.1fMB)\n",
         total ? 100.0 * moved / 2 / total : 0, moved / 2.0 / 1048576.0,
         total / 1048576.0);
  return 0;
}
