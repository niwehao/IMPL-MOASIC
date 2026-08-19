/* test_topo.cc —— L2 拓扑/corridor 的离线单测(不需要 GPU、不需要网络)。
 * 这是 Imp_Plan/06 §3 的 T0 级验证:协议核可以在没有 GPU 的登录节点上跑。 */
#include "internal.h"

#include <cassert>
#include <cstdio>
#include <set>

using namespace mosaic;

static const char* cname(int c) {
  static const char* n[] = {"DIRECT", "RX", "TX", "SPREAD0", "SPREAD1", "SPREAD2"};
  return c < 6 ? n[c] : "?";
}

int main() {
  int fails = 0;
  Topo t;
  topo_init(&t, 8, 4, 0);   // 2 host × 4 vGPU

  printf("=== corridor 族(n=8, G=4, rail wiring)===\n");
  for (int s = 0; s < 8; s++) {
    for (int d = 0; d < 8; d++) {
      if (s == d) continue;
      int cl[NCORR];
      int n = t.corridors_of(s, d, NCORR, cl);
      printf("  %d→%d [%s,%s] :", s, d,
             t.same_host(s, d) ? "同机" : (t.rail(s) == t.rail(d) ? "同轨" : "跨轨"),
             t.same_host(s, d) ? "NV" : "NIC");
      for (int i = 0; i < n; i++) {
        Chain ch;
        t.chain_of(s, d, cl[i], &ch);
        printf(" %s(", cname(cl[i]));
        for (int h = 0; h <= ch.nhops; h++) printf("%d%s", ch.path[h], h < ch.nhops ? "→" : "");
        printf(")");
      }
      printf("\n");

      // 断言 1:同机只有一条固定 NV 单跳
      if (t.same_host(s, d) && n != 1) { printf("  FAIL 同机 corridor 数 %d≠1\n", n); fails++; }
      // 断言 2:同轨跨机 = DIRECT + SPREAD lanes(无 TX/RX,它们会退化)
      if (!t.same_host(s, d) && t.rail(s) == t.rail(d)) {
        for (int i = 0; i < n; i++)
          if (cl[i] == CORR_RX || cl[i] == CORR_TX) { printf("  FAIL 同轨不该有 RX/TX\n"); fails++; }
      }
      // 断言 3:链的两端必须是 s 和 d,且中转点不重复
      for (int i = 0; i < n; i++) {
        Chain ch;
        t.chain_of(s, d, cl[i], &ch);
        if (ch.path[0] != s || ch.path[ch.nhops] != d) { printf("  FAIL 链端点错\n"); fails++; }
        std::set<int> seen;
        for (int h = 0; h <= ch.nhops; h++)
          if (!seen.insert(ch.path[h]).second) { printf("  FAIL 链上有重复节点\n"); fails++; }
        if (ch.nhops > MAX_HOPS) { printf("  FAIL 跳数 %d>3\n", ch.nhops); fails++; }
      }
    }
  }

  printf("\n=== 资源签名(支配剪枝的前提:签名互不包含)===\n");
  {
    int s = 0, d = 5;          // 跨机跨轨:corridor 最全的 pair
    int cl[NCORR];
    int n = t.corridors_of(s, d, NCORR, cl);
    std::vector<std::set<int>> sigs;
    for (int i = 0; i < n; i++) {
      Chain ch;
      t.chain_of(s, d, cl[i], &ch);
      int res[16];
      int nr = t.chain_res(ch, res);
      std::set<int> sg(res, res + nr);
      printf("  %-8s: ", cname(cl[i]));
      for (int r : sg) printf("r%d/%s ", r / RES_KINDS,
                              (const char*[]){"NICu", "NICd", "NVu", "NVd", "SPN"}[r % RES_KINDS]);
      printf("\n");
      sigs.push_back(sg);
    }
    // SPREAD 各 lane 是同形不同 rail:签名不同(否则 lane 没意义)
    for (size_t i = 0; i < sigs.size(); i++)
      for (size_t j = i + 1; j < sigs.size(); j++)
        if (sigs[i] == sigs[j]) { printf("  FAIL corridor %zu 与 %zu 签名完全相同\n", i, j); fails++; }
  }

  printf("\n=== SPREAD lane 轮转(不同 pair 的同号 lane 应落在不同中转轨)===\n");
  {
    std::set<int> lanes;
    for (int s = 0; s < 4; s++) {
      Chain ch;
      if (t.chain_of(s, 4 + ((s + 1) % 4), CORR_SPREAD0, &ch)) {
        printf("  %d→%d SPREAD0 过轨 %d\n", s, 4 + ((s + 1) % 4), t.rail(ch.path[1]));
        lanes.insert(t.rail(ch.path[1]));
      }
    }
    if (lanes.size() < 2) { printf("  FAIL lane 没有轮转开\n"); fails++; }
  }

  printf("\n%s(fails=%d)\n", fails ? "FAILED" : "PASSED", fails);
  return fails ? 1 : 0;
}
