/* replay_core_test —— L2 协议核 vs 模拟器的决策轨迹对拍(Imp_Plan/01 §4,M2)。
 *
 * 金标:htsim `-mosaic_decision_trace` 的逐 (wave,dst,src) 走廊切分
 * (Moasic/csg-htsim,同一 demand 文件驱动)。本对拍 V1 覆盖 **wave 0**
 * (t=0 prior 注水):模拟器里 wave 0 的各 dst 依错峰槽位(dst mod 8 × 3µs,
 * 通道 B 延迟 2µs)串行决策,槽位小的 dst 的发布对槽位大的可见 ——
 * 对拍按 dst 升序重放、共享一本账,复现同一可见性链。
 *
 * 声明的覆盖边界(如实):
 *  - wave ≥1 依赖交付时序(discharge 演化 + 75% 触发次序),V1 不注入;
 *    全轨迹对拍并入 T11(需要 fabric 账重构,见 log.md 坑 #18)。
 *  - 模拟器 rail 布线的 fabric 账(cross-rail leaf/spine 段)在 runtime 里
 *    只有聚合 SPINE 一本(本机 flat 交换,spine_gbps=0 → 不设限):wave 0
 *    空账本下 fabric 不构成瓶颈,若某 cell 因此分歧会在输出里逐条列出。
 *
 * 走廊号映射:sim {DIRECT=0,TX=1,RX=2,SP=3..} ↔ rt {DIRECT=0,RX=1,TX=2,SP=3..}。
 *
 * 用法:
 *   ./build/replay_core_test <demand_file(sim 格式)> <trace_file> [nv_gbps nic_gbps [taper [--src]]]
 * --src:src 驱动 dispatch 相(任务二)的对拍 —— 金标由模拟器
 *   `-mosaic_src_driven -mosaic_decision_trace` 生成,重放按 src 升序驱动
 *   engine_test_emit_wave_src(错峰槽位 = s mod 8,n=8 时升序即槽位序)。
 * 退出码 0 = wave0 全 (dst,src) 逐字节一致。
 */
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <tuple>
#include <array>
#include <vector>

using namespace mosaic;

static int sim2rt_corr(int c) { return c == 1 ? CORR_TX : c == 2 ? CORR_RX : c; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "用法:replay_core_test demand trace [nv nic [taper [--src]]]\n"); return 2; }
  double nv = argc > 4 ? atof(argv[3]) : 360.0, nic = argc > 4 ? atof(argv[4]) : 100.0;
  bool src_mode = false;
  for (int i = 5; i < argc; i++)
    if (strcmp(argv[i], "--src") == 0) src_mode = true;

  // ── 读 demand(sim 文本格式:node_count N / demand s d bytes)
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

  // ── 读金标 trace:wave0 的 (d,s) → SH[6]
  std::map<std::pair<int,int>, std::array<long long,6>> gold;
  f = fopen(argv[2], "r");
  if (!f) { perror("trace"); return 2; }
  while (fgets(line, sizeof(line), f)) {
    int w, d, s; long long q, sh[6];
    if (sscanf(line, "W %d D %d S %d Q %lld SH %lld %lld %lld %lld %lld %lld",
               &w, &d, &s, &q, &sh[0], &sh[1], &sh[2], &sh[3], &sh[4], &sh[5]) == 10) {
      if (w != 0) continue;
      auto& g = gold[{d, s}];
      for (int c = 0; c < 6; c++) g[c] = sh[c];
    }
  }
  fclose(f);
  if (gold.empty()) { fprintf(stderr, "trace 里没有 wave0 记录\n"); return 2; }

  // ── 构造无网络、无 GPU 的 Comm,按 dst 升序重放 wave0
  const int G = 4;               // 本机测床形状(与 sim -flex_gpus_per_host 一致)
  Comm* c = new Comm();
  mosaicConfigDefault(&c->cfg);
  c->cfg.nranks = n; c->cfg.gpus_per_host = G;
  c->cfg.corridors = 6; c->cfg.waves = 6;
  c->cfg.ledger = 1; c->cfg.bcast = 0;   // bcast=0:重放不走网络,共享账本即视图
  c->cfg.use_gpu = 0; c->cfg.sched = MOSAIC_SCHED_MOSAIC;
  // spine_gbps = 模拟器 -flex_taper_gbps 同口径(rail leaf 段聚合额定,默认 180)
  c->cfg.nv_gbps = nv; c->cfg.nic_gbps = nic;
  c->cfg.spine_gbps = argc > 5 ? atof(argv[5]) : 180.0;
  c->cfg.detector = 0;
  c->scounts.assign(n, 0); c->sdispls.assign(n, 0);
  c->rcounts.assign(n, 0); c->rdispls.assign(n, 0);
  c->demand = D;
  topo_init(&c->topo, n, G, 0);   // 先有拓扑(nres)才能建账本
  ledger_init(c);

  size_t rows = 0, bad = 0;
  // 对比一个决策者本波的 (对端,corr) 分配与金标行;dst 相 mine 键 = (src,corr),
  // src 相 mine 键 = (dst,corr)(CT_POLICY_SRC 语义:ent.src 承载 dst 端点)
  auto check = [&](int decider, const std::map<std::pair<int,int>, long long>& mine) {
    for (int p = 0; p < n; p++) {
      const int d = src_mode ? p : decider, s = src_mode ? decider : p;
      if (p == decider || D[(size_t)s * n + d] == 0) continue;
      rows++;
      auto it = gold.find({d, s});
      if (it == gold.end()) { printf("MISS 金标无 (d=%d,s=%d)\n", d, s); bad++; continue; }
      bool row_bad = false;
      char msg[256]; int off = 0;
      for (int sc = 0; sc < 6; sc++) {
        long long want = it->second[sc];
        long long got = 0;
        auto jt = mine.find({p, sim2rt_corr(sc)});
        if (jt != mine.end()) got = jt->second;
        off += snprintf(msg + off, sizeof(msg) - off, " c%d:%lld/%lld", sc, got, want);
        if (got != want) row_bad = true;
      }
      if (row_bad) { printf("DIFF d=%d s=%d(got/want)%s\n", d, s, msg); bad++; }
    }
  };
  for (int r = 0; r < n; r++) {          // 升序 = 错峰槽位序(id mod 8, n=8)
    topo_init(&c->topo, n, G, r);
    c->cfg.rank = r;
    if (src_mode) {
      c->srcc = SrcCtrl();
      c->srcc.active = true;
      for (int d = 0; d < n; d++)
        if (d != r) c->srcc.remaining[d] = D[(size_t)r * n + d];
      engine_test_emit_wave_src(c);      // 充入共享账本 + 填 dbg_last_ents
    } else {
      c->dst = DstCtrl();
      c->dst.active = true;
      c->dst.wave_idx = 0;
      for (int s = 0; s < n; s++)
        if (s != r) c->dst.remaining[s] = D[(size_t)s * n + r];
      engine_test_emit_wave(c);
    }
    std::map<std::pair<int,int>, long long> mine;
    for (auto& e : c->dbg_last_ents) mine[{(int)e.src, (int)e.corr}] += e.len;
    check(r, mine);
  }
  printf("replay%s wave0:%zu 行,%zu 分歧 → %s\n", src_mode ? "(src)" : "",
         rows, bad, bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
