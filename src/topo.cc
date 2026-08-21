/* topo.cc —— L2 的拓扑与 corridor 族(报告 §4.1 的剪枝结果,代码固化)。
 *
 * 模拟的 wiring = rail:H 台 host、每台 G 个 vGPU,vGPU 的 rail 号 = host 内序号,
 * 每个 vGPU 有自己的 vNIC。同 rail 跨机 = 直连(不过 spine);跨 rail 跨机 = 过 spine。
 *
 * 剪枝后的 corridor 族(与 mosaic_schedule.h: pair_corridors()/corridor_hops() 同构):
 *   DIRECT   s→d                       入 d 走 NIC 管
 *   TX       s→(host(s),rail(d))→d     发送侧 NV 换轨,避 spine;入 d 走 NIC 管
 *   RX       s→(host(d),rail(s))→d     入口摊到别的 NIC;入 d 走 NV 管
 *   SPREAD_j s→(host(s),j)→(host(d),j)→d   同 rail pair 唯一的摊开手段;入 d 走 NV 管
 * mask 规则:同机 = 固定 NV 单跳;同 rail 跨机 = {DIRECT, SPREAD lanes};跨 rail = 全部。
 */
#include "internal.h"

namespace mosaic {

void topo_init(Topo* t, int nranks, int G, int rank) {
  t->nranks = nranks;
  t->G = G;
  t->H = (nranks + G - 1) / G;
  t->rank = rank;
  for (int r = 0; r < nranks; r++) t->nic_leader[r] = (uint8_t)r;   // 恒等 = 私有 NIC
  // 网络虚拟化档(根 CLAUDE.md「网络虚拟化铁律」):MOSAIC_NET_MODE=rail 时
  // 跨 rail 的跨机对**不存在直连通道**,必须换轨;origin(默认)= 旧行为。
  {
    const char* nm = getenv("MOSAIC_NET_MODE");
    t->net_rail = (nm && *nm && strcmp(nm, "rail") == 0) ? 1 : 0;
    if (t->net_rail && rank == 0)
      MLOG_WARN("网络虚拟化:rail 档 —— 跨 rail 跨机对禁止 DIRECT,必须换轨");
  }
  // T5:LP 落点表(MOSAIC_LP_TABLE=<文件>,格式 "jdist r rd j w",
  // 每 (r,rd) 组内权重降序)。加载失败只告警不致命 —— 回退轮转公式。
  t->lp_j.clear();
  const char* lp = getenv("MOSAIC_LP_TABLE");
  if (lp && *lp) {
    FILE* f = fopen(lp, "r");
    if (!f) {
      MLOG_WARN("E1030 LP 表打不开(%s),落点回退轮转", lp);
    } else {
      t->lp_j.assign(G, std::vector<std::vector<uint8_t>>(G));
      char tag[16]; int r, rd, j; double w; int rows = 0;
      char line[256];
      while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%15s %d %d %d %lf", tag, &r, &rd, &j, &w) == 5 &&
            strcmp(tag, "jdist") == 0 &&
            r >= 0 && r < G && rd >= 0 && rd < G && j >= 0 && j < G) {
          t->lp_j[r][rd].push_back((uint8_t)j);
          rows++;
        }
      }
      fclose(f);
      if (rows == 0) { t->lp_j.clear(); MLOG_WARN("E1031 LP 表无 jdist 行(%s)", lp); }
      else if (rank == 0) MLOG_INFO("LP 落点表加载:%d 行(%s)", rows, lp);
    }
  }
}

// SPREAD lane 的中转 rail 选择 —— **逐字对照模拟器 corridor_hops 的 rail 分支**
// (mosaic_schedule.h:「j = (rd+1+h+r) % G,跳过 r/rd,取第 lane 个合法值」;
// 旧版用 (s*G+d) 轮转,是移植分歧,2026-08-14 决策对拍时修正)。
static int transit_rail_of_lane(const Topo& t, int s, int d, int lane) {
  int r = t.rail(s), rd = t.rail(d), h = t.host(s);
  // LP 表在场时:lane k = jdist[r][rd] 的第 k 项(权重降序;与模拟器
  // emit_spread_jdist 的"全轨按权重摊"在 lane 粒度上对应)。表外 lane
  // 回退轮转。
  if (!t.lp_j.empty()) {
    const auto& v = t.lp_j[r][rd];
    if (lane < (int)v.size()) {
      int j = v[lane];
      if (j != r && j != rd) return j;
    }
  }
  int j = (rd + 1 + h + r) % t.G;
  int picked = 0;
  for (int guard = 0; guard < 4 * t.G; guard++) {
    if (j != r && j != rd) {
      if (picked == lane) return j;
      picked++;
    }
    j = (j + 1) % t.G;
  }
  return -1;
}

bool Topo::chain_of(int s, int d, int c, Chain* out) const {
  if (s == d) return false;
  if (same_host(s, d)) {                 // 机内:固定 NV 单跳
    if (c != CORR_DIRECT) return false;
    out->nhops = 1; out->path[0] = (uint8_t)s; out->path[1] = (uint8_t)d;
    return true;
  }
  const int rs = rail(s), rd = rail(d);
  switch (c) {
    case CORR_DIRECT:
      // 网络虚拟化 rail 档:**跨 rail 没有直连通道**(要求 2)。同 rail 才放行;
      // 跨 rail 只能走 RX/TX(换轨一次)或 SPREAD(换轨两次)——正是下面几个分支。
      // origin 档(net_rail=0)保持旧行为:跨 rail 也可直发,只多记一笔 fabric 账。
      if (net_rail && rs != rd) return false;
      out->nhops = 1; out->path[0] = (uint8_t)s; out->path[1] = (uint8_t)d;
      return true;
    case CORR_RX: {                      // 落在 d 机上与 s 同 rail 的 GPU,再 NV 进 d
      if (rs == rd) return false;        // 退化成 DIRECT
      int r = host(d) * G + rs;
      if (r == d || r >= nranks) return false;
      out->nhops = 2; out->path[0] = (uint8_t)s; out->path[1] = (uint8_t)r; out->path[2] = (uint8_t)d;
      return true;
    }
    case CORR_TX: {                      // 先 NV 到 s 机上与 d 同 rail 的 GPU,再同轨出网
      if (rs == rd) return false;
      int r = host(s) * G + rd;
      if (r == s || r >= nranks) return false;
      out->nhops = 2; out->path[0] = (uint8_t)s; out->path[1] = (uint8_t)r; out->path[2] = (uint8_t)d;
      return true;
    }
    default: {                           // SPREAD lane
      int lane = c - CORR_SPREAD0;
      // lane 可用性与模拟器 pair_corridors 一致:G > 2 + lane
      if (lane < 0 || G <= 2 + lane) return false;
      int j = transit_rail_of_lane(*this, s, d, lane);
      if (j < 0) return false;
      int r1 = host(s) * G + j, r2 = host(d) * G + j;
      if (r1 >= nranks || r2 >= nranks || r1 == s || r2 == d) return false;
      out->nhops = 3;
      out->path[0] = (uint8_t)s; out->path[1] = (uint8_t)r1;
      out->path[2] = (uint8_t)r2; out->path[3] = (uint8_t)d;
      return true;
    }
  }
}

int Topo::corridors_of(int s, int d, int c_limit, int* out) const {
  int n = 0;
  Chain ch;
  for (int c = 0; c < NCORR && c < c_limit; c++)
    if (chain_of(s, d, c, &ch)) out[n++] = c;
  return n;
}

int Topo::hop_res(int a, int b, int* out) const {
  int n = 0;
  if (same_host(a, b)) {
    out[n++] = res(a, RES_NV_UP);
    out[n++] = res(b, RES_NV_DN);
  } else {
    out[n++] = nic_res_up(a);          // NIC 账记物理单元(leader 位),见 Topo 注释
    out[n++] = nic_res_dn(b);
    if (rail(a) != rail(b)) {                 // 跨轨才过 fabric(rail 直连不过)
      // 与模拟器 lg_res_of_hop(rail 布线,spineagg)逐字一致:
      // 源 rail 的 leaf 上行段 + 目的 rail 的 leaf 下行段
      out[n++] = res_fab_up(rail(a));
      out[n++] = res_fab_dn(rail(b));
    }
  }
  return n;
}

int Topo::chain_res(const Chain& ch, int* out) const {
  int n = 0;
  for (int h = 0; h < ch.nhops; h++) n += hop_res(ch.path[h], ch.path[h + 1], out + n);
  return n;
}

// ─────────────────────────────────────────────────────────── Ledger
void Ledger::init(int nres, int nranks) {
  charge.assign(nres, 0);
  discharge.assign(nres, 0);
  cap_Bps.assign(nres, 0);
  peer_charge.assign(nranks, std::vector<uint64_t>(nres, 0));
}

uint64_t Ledger::outstanding(int r) const {
  uint64_t ch = charge[r];
  for (const auto& p : peer_charge) ch += p[r];
  uint64_t di = discharge[r];
  if (!disch_agg.empty())
    for (int e : disch_agg[r]) di += discharge[e];   // NIC 单元:成员的送达也扣
  return ch > di ? ch - di : 0;
}

double Ledger::proj_us(int r) const {
  double cap = cap_Bps[r];
  if (cap <= 0) return 0;
  return (double)outstanding(r) / cap * 1e6;
}

void Ledger::charge_chain(const int* r, int n, uint64_t bytes) {
  for (int i = 0; i < n; i++) charge[r[i]] += bytes;
}

void Ledger::set_peer_charge(int publisher, const uint64_t* v, int nres) {
  // 通道 B 语义:只存该发布者的**最新一份**,整体替换(单写者 ⇒ 不会分叉)
  auto& p = peer_charge[publisher];
  for (int i = 0; i < nres; i++) p[i] = v[i];
}

void Ledger::clear_epoch() {
  std::fill(charge.begin(), charge.end(), 0);
  std::fill(discharge.begin(), discharge.end(), 0);
  for (auto& p : peer_charge) std::fill(p.begin(), p.end(), 0);
}

void ledger_init(Comm* c) {
  Topo& t = c->topo;
  c->ledger.init(t.nres(), t.nranks);
  for (int r = 0; r < t.nranks; r++) {
    c->ledger.cap_Bps[t.res(r, RES_NIC_UP)] = c->cfg.nic_gbps * 1e9 / 8.0;
    c->ledger.cap_Bps[t.res(r, RES_NIC_DN)] = c->cfg.nic_gbps * 1e9 / 8.0;
    c->ledger.cap_Bps[t.res(r, RES_NV_UP)] = c->cfg.nv_gbps * 1e9 / 8.0;
    c->ledger.cap_Bps[t.res(r, RES_NV_DN)] = c->cfg.nv_gbps * 1e9 / 8.0;
    // 每 rank 的 SPINE_UP 位 cap=0(spine 走 per-rail 聚合账,见下;
    // cap<=0 的资源代价函数恒跳过)。该口的计数仍由 buck[RES_SPINE_UP] 承担
    c->ledger.cap_Bps[t.res(r, RES_SPINE_UP)] = 0;
  }
  // fabric 聚合账(per-rail 上/下行):额定 = spine_gbps(= 模拟器 taper 口径,
  // 每 rail leaf 段的聚合速率);<=0 = 不设限(flat 交换,cost 跳过该账)
  for (int j = 0; j < t.G; j++) {
    double cap = c->cfg.spine_gbps > 0 ? c->cfg.spine_gbps * 1e9 / 8.0 : 0;
    c->ledger.cap_Bps[t.res_fab_up(j)] = cap;
    c->ledger.cap_Bps[t.res_fab_dn(j)] = cap;
  }
}

}  // namespace mosaic
