/* fast_native.cc —— FAST 基线:**官方 host 调度器**驱动的 adapter。
 *
 * 诚实边界(这段话就是本文件的契约,勿再夸大):
 *   - 算法 = FAST 官方仓库 FAST/amd/src/fast_alltoall/{alltoall_matrix,
 *     alltoall_algorithm,alltoall_local_scheduler,alltoall_global_scheduler}.cc,
 *     由 Makefile 直接编译**未拷贝、未修改**的官方源文件(含 Birkhoff/SDSM
 *     分解、sender balance、逐步 restore;hip 头用空 shim 注入,见
 *     src/fast_native/hip/hip_runtime.h)。
 *   - 本文件只是 adapter(Imp_Plan/04 §1.2 允许的 glue):把全局 demand 矩阵
 *     投喂给官方 GlobalScheduler,再把它产出的逐步 schedule 翻译成本 runtime
 *     的显式 hop 链。执行底座(限速、staging、hop 依赖、计时、校验)与
 *     MOSAIC 完全共用 —— 公平性的技术落点。
 *   - 官方 GPU 侧 buffer 布局与 NVSHMEM kernel(libflash)在本机模拟形态下
 *     不使用(NVSHMEM 不支持多 PE 共享一卡);它们的部署走 fast-rt/
 *     (2-PE 烟测)与 Raven(rank_n_per_node=4,M8)。
 *
 * 翻译规则(逐字段对照官方语义,行号指官方源码):
 *   - LocalScheduler::data_after_balance[i][D*G+j].sz[from]/.offset[from]
 *     = balance 后落在本机 GPU i 上、属于消息 (S,from)→(D,j) 的字节段
 *     (alltoall_local_scheduler.cc:38-44,200-209)。
 *   - run_scheduler 按 Birkhoff 置换逐步调用 restore_one_server
 *     (alltoall_global_scheduler.cc:134-152),把这些字节段消费进:
 *       steps[t].channel / crossnode_sz     —— 第 t 步跨机 (S,i)→(D,i)
 *       steps[t+1].restore[S][i][j*G+from]  —— 落地后 NV 转发 (D,i)→(D,j)
 *       steps[t+1].direct_cpy[S][i][from]   —— i==j,落地即终点
 *     每条都带 {sz, offset}(offset = 在原 (src,dst) 消息里的字节偏移,
 *     alltoall_local_scheduler.cc:248-254)。
 *   - 因此对 rank me=(S,from):第 t 步经中转 i 到 (D,j) 的字节段的完整链 =
 *       me →NV→ (S,i) →NIC 同轨→ (D,i) →NV→ (D,j)
 *     (i==from / j==i 时对应 hop 塌缩)。把它按 slice_bytes 切片、wave=t
 *     发射,即得与官方一致的路由与步序;步内的包级顺序与官方 GPU 运行时
 *     的差异 = 与 MOSAIC 相同的 min-epoch 近似(File/log_raven.md §6)。
 *
 * 完整性守卫:官方分解在数值上覆盖全部 demand(SDSM 补虚字节只垫置换,
 * restore 用 MIN 截断);adapter 仍逐 (dst) 核对发射字节数,缺口走 DIRECT
 * 补发并 MLOG_ERR 记账 —— 出现即视为 adapter bug,不静默。
 */
#include "internal.h"

#include <vector>

// 官方头文件(未修改;include 根 = FAST/amd/src/include,见 Makefile)
#include "fast_alltoall/alltoall_global_scheduler.h"

namespace mosaic {

void emit_chain_pub(Comm* c, int dst, const Chain& ch, uint64_t off, uint64_t len, int wave);
void emit_range_pub(Comm* c, int dst, int corr, uint64_t off, uint64_t len, int wave);

// 把显式路径压成 Chain(塌缩重复 hop),并断言 ≤MAX_HOPS
static bool make_chain(const int* nodes, int n, Chain* out) {
  int path[MAX_HOPS + 2], m = 0;
  for (int i = 0; i < n; i++) {
    if (m == 0 || path[m - 1] != nodes[i]) path[m++] = nodes[i];
  }
  if (m - 1 > MAX_HOPS || m < 2) return false;
  out->nhops = (uint8_t)(m - 1);
  for (int i = 0; i < m; i++) out->path[i] = (uint8_t)path[i];
  return true;
}

void sched_fast_native_plan(Comm* c) {
  const int n = c->cfg.nranks, me = c->cfg.rank, G = c->cfg.gpus_per_host;
  const int H = n / G, S = c->topo.host(me), from = c->topo.rail(me);

  if (H > MAX_SERVER_NUM || G > MAX_GPU_PER_SERVER) {
    MLOG_ERR("E4002 官方 FAST 上限 server<=%d gpu/srv<=%d(现 %d/%d),退回 structural",
             MAX_SERVER_NUM, MAX_GPU_PER_SERVER, H, G);
    void sched_fast_struct_plan(Comm*);
    sched_fast_struct_plan(c);
    return;
  }

  // ── 官方调度(SPMD:每 rank 用同一全局矩阵确定性重算,与 flash_scheduler 同构)。
  // 跨 collective 复用官方 GlobalScheduler:首个 collective init,之后走官方
  // update_global_scheduler(官方提供的重投喂入口)—— 避免每次 22MB 的
  // malloc/free 抖动把「含 plan」口径不公平地做差。plan 时间两口径见 07 §2.2。
  uint64_t t0 = now_ns();
  static thread_local GlobalScheduler* gs_cache = nullptr;
  static thread_local int gs_H = 0, gs_G = 0;
  if (gs_cache && (gs_H != H || gs_G != G)) {
    free_global_scheduler(gs_cache); delete gs_cache; gs_cache = nullptr;
  }
  if (!gs_cache) {
    gs_cache = new GlobalScheduler();
    memset(gs_cache, 0, sizeof(GlobalScheduler));
    init_global_scheduler(gs_cache, (uint)H, (uint)G, c->demand.data(), (uint)me);
    gs_H = H; gs_G = G;
  } else {
    update_global_scheduler(gs_cache, c->demand.data());
  }
  GlobalScheduler& gs = *gs_cache;
  run_scheduler(&gs);
  c->st.sched_plan_us = (now_ns() - t0) / 1000.0;   // plan 时间单独计量(07 §2.2 两口径)

  std::vector<uint64_t> emitted(n, 0);

  // ── 机内(intrinsic all-to-all):官方直接单跳 NV(prepare_load_balance:132-135)
  for (int d = S * G; d < (S + 1) * G; d++) {
    if (d == me || c->scounts[d] == 0) continue;
    emit_range_pub(c, d, CORR_DIRECT, 0, c->scounts[d], 0);
    emitted[d] += c->scounts[d];
  }

  // ── 跨机:逐步翻译官方 schedule(步 t 的落地转发/直达记录在 steps[t+1])
  const scheduling_result_t* sc = gs.sched;
  (void)sc;
  for (uint t = 0; t + 1 < sc->step_n; t++) {
    int D = (int)sc->steps[t].to_server[S];
    if (D == S) continue;                       // SDSM 垫出的自环置换,无真流量
    const scheduling_step_t& nxt = sc->steps[t + 1];
    for (int i = 0; i < G; i++) {               // 中转 GPU(balance 落点,机内)
      // i==j:落地即终点 (D,i)
      const recv_data_t& dc = nxt.direct_cpy[S][i][from];
      if (dc.sz) {
        int dst = D * G + i;
        int nodes[3] = {me, S * G + i, dst};
        Chain ch;
        if (make_chain(nodes, 3, &ch)) {
          MLOG_DBG("fastN t=%u dcpy i=%d dst=%d off=%llu len=%llu", t, i, dst,
                   (unsigned long long)dc.offset, (unsigned long long)dc.sz);
          emit_chain_pub(c, dst, ch, dc.offset, dc.sz, (int)t);
          emitted[dst] += dc.sz;
        }
      }
      // i!=j:落地 (D,i) 后 NV 转发到 (D,j)
      for (int j = 0; j < G; j++) {
        if (j == i) continue;
        const recv_data_t& r = nxt.restore[S][i][(size_t)j * G + from];
        if (!r.sz) continue;
        int dst = D * G + j;
        int nodes[4] = {me, S * G + i, D * G + i, dst};
        Chain ch;
        if (make_chain(nodes, 4, &ch)) {
          emit_chain_pub(c, dst, ch, r.offset, r.sz, (int)t);
          emitted[dst] += r.sz;
        }
      }
    }
  }

  // ── 完整性守卫(adapter 级;触发 = 翻译 bug,响亮报错不静默)
  for (int d = 0; d < n; d++) {
    if (d == me) continue;
    if (emitted[d] < c->scounts[d]) {
      uint64_t gap = c->scounts[d] - emitted[d];
      MLOG_ERR("E4003 FAST native 翻译缺口 (%d→%d) %llu 字节,DIRECT 补发(adapter bug!)",
               me, d, (unsigned long long)gap);
      emit_range_pub(c, d, CORR_DIRECT, emitted[d], gap, 0);
    } else if (emitted[d] > c->scounts[d]) {
      MLOG_ERR("E4004 FAST native 翻译超发 (%d→%d) %llu>%llu(adapter bug!数据将错位)",
               me, d, (unsigned long long)emitted[d], (unsigned long long)c->scounts[d]);
    }
  }

  // gs_cache 常驻复用,进程退出时由 OS 回收(官方 free_global_scheduler 只在
  // 拓扑形状变化时调用,见上)
}

}  // namespace mosaic
