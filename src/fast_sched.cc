/* fast_sched.cc —— structural FAST(近似版,**不是官方代码**;仅作消融位)。
 *
 * 诚实声明(2026-08-14 修正:本文件旧版头注释声称「算法来自官方仓库的
 * host 侧调度代码,逐行不改」,那是**错的** —— 本文件从头到尾没有 include
 * 过任何官方文件,只是按 FAST 论文的三段结构(sender balance → 同轨出网 →
 * receiver dispatch)重写的**均摊近似**,没有官方的 Birkhoff/SDSM 分解与
 * 逐步置换对齐。旧的 results 数字里 FAST 列来自本文件,已在
 * results/README.md 标注口径并重跑)。
 *
 * 现在的角色分工:
 *   - MOSAIC_SCHED_FAST(默认)→ src/fast_native.cc:官方
 *     fast_alltoall/*.cc 未改动编译链接,官方 GlobalScheduler 产 schedule,
 *     adapter 翻译成 hop 链 —— 那才是「官方 FAST」基线;
 *   - 本文件 = MOSAIC_SCHED_FAST_STRUCT / MOSAIC_FAST_NATIVE=0:
 *     结构近似消融位,用于分离「三段结构」与「Birkhoff 置换协调」各自的贡献。
 * 两者与 MOSAIC 共用同一执行底座(限速、staging、hop 依赖、计时、校验)。
 */
#include "internal.h"

#include <algorithm>

namespace mosaic {

// FAST 的 corridor 语义映射到本 runtime:
//   balance 到同机 relay r(rail 号 = 目的 rail)→ 与 CORR_TX 的链**同形**
//   (s →NV→ (host(s), rail(d)) →NIC 同轨→ d)。
//   receiver dispatch 则对应 landing≠dst 时的 (host(d), rail(s)) →NV→ d,
//   与 CORR_RX 同形。二者都在 corridor 表里,所以 adapter 只需决定"每个
//   pair 的字节按什么比例落到哪条链",执行完全复用。
//
// 官方 local scheduler 的 sender balance = 把 server 对之间的字节**均摊到
// G 行**(每个本机 GPU 出一份),这正是"每字节两过机内层"的来源(W7)。
void sched_fast_struct_plan(Comm* c) {
  const int n = c->cfg.nranks, me = c->cfg.rank, G = c->cfg.gpus_per_host;
  for (int d = 0; d < n; d++) {
    if (d == me || c->scounts[d] == 0) continue;
    const uint64_t total = c->scounts[d];
    if (c->topo.same_host(me, d)) {                 // 机内:单跳 NV,与 FAST 一致
      emit_range_pub(c, d, CORR_DIRECT, 0, total, 0);
      continue;
    }
    // 跨机:按 FAST 的 sender balance,把字节均摊到本机 G 个 GPU 出网。
    // 落在"与 d 同 rail 的本机 GPU"上的那份 = 同轨直出(TX 链);
    // 其余份额需要在目的机做 receiver dispatch(SPREAD 链,经 j 轨落地再 NV 进 d)。
    uint64_t off = 0;
    for (int j = 0; j < G; j++) {
      uint64_t share = total / G + ((uint64_t)j < total % G ? 1 : 0);
      if (!share) continue;
      int corr;
      if (j == c->topo.rail(d)) corr = CORR_TX;          // balance 落点恰为 d 的轨
      else if (j == c->topo.rail(me)) corr = CORR_RX;    // 自己出网,目的机再 dispatch
      else {
        // 经第 j 轨:先 NV 到本机 j,再同轨出网,再 NV 进 d —— SPREAD lane
        corr = -1;
        Chain ch;
        for (int cc = CORR_SPREAD0; cc < NCORR; cc++) {
          if (!c->topo.chain_of(me, d, cc, &ch)) continue;
          if (c->topo.rail(ch.path[1]) == j) { corr = cc; break; }
        }
        if (corr < 0) corr = CORR_RX;                    // 该 lane 不存在则退回
      }
      Chain probe;
      if (!c->topo.chain_of(me, d, corr, &probe)) corr = CORR_DIRECT;
      emit_range_pub(c, d, corr, off, share, 0);
      off += share;
    }
  }
}

}  // namespace mosaic
