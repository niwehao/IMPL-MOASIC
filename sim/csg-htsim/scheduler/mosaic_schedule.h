// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MOSAIC_SCHEDULE_H
#define MOSAIC_SCHEDULE_H

// MOSAIC prototype (Note/idea_moe_subschedule_assembly.md): multiple
// SUB-SCHEDULES (corridors) built greedily -- no LP, no solver -- with
// PER-DESTINATION-GPU mixing ratios that are ADJUSTED DYNAMICALLY from
// measured feedback while the collective executes.
//
// Corridors for a pair s=(h,r) -> d=(n,r') (cross-server):
//   DIRECT : s -> d               r==r': same-rail NIC; r!=r': tapered spine
//   TX     : s -> (h,r') -> d     sender NVLink rail-align + same-rail NIC
//                                 (ingress still on d's NIC)      [r!=r']
//   RX     : s -> (n,r)  -> d     same-rail NIC lands on d's server spread
//                                 by SOURCE rail + receiver NVLink [r!=r']
//   SPREAD3: s -> (h,j) -> (n,j) -> d   NVLink + rail j + NVLink, rotating
//                                 j: the ingress spreader for same-rail
//                                 pairs (and a spine-free alternative).
// Same-server pairs ride their single NVLink path outside the mixer.
//
// Execution: every destination GPU runs its own control loop. Its traffic
// is emitted in W waves; wave w splits every remaining pair quota across
// the applicable corridors by the GPU's ratio vector lambda. Once a wave
// crosses a delivery threshold (default 75%), each corridor's EFFECTIVE
// BANDWIDTH so far (delivered bytes / corridor busy time) is measured,
// lambda moves toward the measured rate shares (exponential smoothing +
// exploration floor -- rate-proportional allocation equalizes corridor
// finish times, i.e. divisible-load waterfilling learned online), and the
// next wave is emitted immediately (overlap hides the wave boundary).
// Per-GPU loops mean a hot expert converges to an RX/SPREAD-heavy blend
// while cold GPUs stay DIRECT-heavy, with no global barrier anywhere.
//
// Node-level send behavior reuses the fast/teccl min-epoch path-follower
// (epoch = wave*4 + hop phase), so scheduler.h maps mosaic_schedule onto
// SmartFastScheduleScheduler. A lagging GPU's earlier-wave chunks thereby
// automatically win ports over already-ahead GPUs' later waves.

#include "config.h"
#include "common.h"
#include "fast_schedule.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef SMART_GRAPH_DEFINED
#ifndef MOSAIC_SCHEDULE_RUNTIME_H
#define MOSAIC_SCHEDULE_RUNTIME_H

class SmartMosaicRuntime : public SmartDeliveryObserver,
                           public EventSource {
    struct DstState;
public:
    // Corridor id space. 0-3 are the pruned rail-wiring classes (§4.1 of
    // the report); 4-5 are EXTRA spread lanes (same SPREAD3 shape, distinct
    // transit rail j per pair) used only by the corridor-count ablation:
    // they add lambda granularity but no new resource signature.
    static const uint32_t MAX_CORRIDORS = 6;
    static const uint32_t CORRIDORS = MAX_CORRIDORS;  // array sizing
    enum Corridor {DIRECT = 0, TX = 1, RX = 2, SPREAD3 = 3};
    // Wiring family: corridors are DERIVED per wiring (X2 thesis).
    // RAIL: {DIRECT,TX,RX,SPREAD3,+spread lanes}. POD: {DIRECT,
    // RX-lane m}: landing GPU j on the dst host relays over NV; j
    // selects BOTH the ingress NIC and (via the (src+j)%S pin) the
    // spine -- one lever spreads both bottlenecks.
    // RAIL2P (dual-plane rail): DIRECT is SPLIT per plane (c0=DIRECT_A
    // even pairs, c1=DIRECT_B odd pairs) so lambda controls each plane's
    // NIC pipe separately -- the dst ingress becomes a true 3-pipe system
    // (NV / NIC-A / NIC-B) and the prior is K-pipe waterfilling.
    // CUBE (flex_cube: intra-host cube-mesh, heterogeneous link speeds):
    // same corridor ID layout as RAIL (nested sets and ramp reuse), but
    //   RX      lands on (n,r) and cube-routes to d;
    //   TX      cube-routes r->rd on the src host, then same-rail NIC;
    //   SP_k    lands on the k-th FASTEST cube neighbor j of rd (j != r),
    //           via same-rail NIC j and the single link j->d -- lane k
    //           maps to a physical ingress link of d, so per-corridor
    //           feedback measures per-link effective bandwidth.
    // The dst "NV pipe" becomes B_in(d) = sum of incident link speeds
    // (per-dst heterogeneous); the prior is K-pipe waterfilling over
    // {NIC, incident links} with same-host inflow pinned to its path's
    // last link.
    enum Wiring {WIRING_RAIL = 0, WIRING_POD = 1, WIRING_RAIL2P = 2,
                 WIRING_CUBE = 3};
    enum R2PCorridor {R2P_DIRECT_A = 0, R2P_DIRECT_B = 1, R2P_TX = 2,
                      R2P_RX = 3, R2P_SPREAD = 4};

    SmartMosaicRuntime(SmartGraph& graph, uint32_t node_count,
                       uint32_t gpus_per_server,
                       const vector<vector<mem_b> >& demand,
                       uint32_t waves, double gamma, double floor_ratio,
                       double overlap_threshold, bool adapt,
                       double nv_gbps, double nic_gbps,
                       uint32_t corridor_count = 4,
                       uint32_t wiring = WIRING_RAIL,
                       uint32_t spine_count = 0,
                       double planeb_factor = 1.0)
        : EventSource("mosaic_lg_sync"),
          _graph(graph), _node_count(node_count), _gpus(gpus_per_server),
          _demand(demand), _waves(waves), _gamma(gamma),
          _floor(floor_ratio), _threshold(overlap_threshold),
          _adapt(adapt), _nv_gbps(nv_gbps), _nic_gbps(nic_gbps),
          _corridor_count(corridor_count), _wiring(wiring),
          _spine_count(spine_count), _planeb_factor(planeb_factor) {
        if (_gpus == 0 || _node_count % _gpus != 0) {
            throw runtime_error("mosaic: gpus_per_server must divide nodes");
        }
        if (_corridor_count < 1 || _corridor_count > MAX_CORRIDORS) {
            throw runtime_error("mosaic: corridor_count must be in [1,6]");
        }
        if (_waves == 0) {
            _waves = 1;
        }
        _servers = _node_count / _gpus;
        _remaining = demand;
        _expected_app_bytes = 0;
        for (uint32_t s = 0; s < _node_count; s++) {
            for (uint32_t d = 0; d < _node_count; d++) {
                if (s != d) {
                    _expected_app_bytes += demand[s][d];
                }
            }
        }
        _dst_state.resize(_node_count);
        for (uint32_t d = 0; d < _node_count; d++) {
            set_uniform_lambda(_dst_state[d]);
        }
        _dst_expected.assign(_node_count, 0);
        _dst_delivered.assign(_node_count, 0);
        _dst_done.assign(_node_count, false);
        for (uint32_t d = 0; d < _node_count; d++) {
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s != d) {
                    _dst_expected[d] += demand[s][d];
                }
            }
        }
        _prior_demand = &_demand;
        if (_nv_gbps > 0.0) {
            init_lambda_from_demand();
        }
    }

    // Nested corridor sets for the count ablation: 1={DIRECT},
    // 2=+RX, 3=+TX, 4=+SPREAD3 (default), 5/6=+extra spread lanes.
    static bool rail_nested(uint32_t c, uint32_t count) {
        switch (c) {
        case DIRECT: return true;
        case RX: return count >= 2;
        case TX: return count >= 3;
        default: return count >= c + 1;  // spread lane c-3
        }
    }

    bool corridor_enabled(uint32_t c) const {
        if (_wiring == WIRING_POD || _wiring == WIRING_RAIL2P) {
            return c < _corridor_count;  // nested plain count
        }
        return rail_nested(c, _corridor_count);  // rail AND cube
    }

    // CUBE wiring: install the intra-host ground truth (same matrices the
    // topology executes). local[a][b] = link Gbps (0 = no link);
    // paths[a][b] = pinned local node sequence (used to pin same-host
    // inflow to its LAST link in the prior's fixed loads). Re-derives the
    // waterfilling prior; call before the E3/F3b setters so their
    // ingress bounds see per-dst B_in.
    void set_cube_profile(const vector<vector<double> >& local,
                          const vector<vector<vector<uint32_t> > >& paths) {
        if (_wiring != WIRING_CUBE) {
            throw runtime_error("mosaic: cube profile needs cube wiring");
        }
        _cube_local = local;
        _cube_paths = paths;
        _cube_bin.assign(_gpus, 0.0);
        _cube_nbrs.assign(_gpus, vector<uint32_t>());
        for (uint32_t g = 0; g < _gpus; g++) {
            for (uint32_t v = 0; v < _gpus; v++) {
                if (local[g][v] > 0.0) {
                    _cube_bin[g] += local[g][v];
                    _cube_nbrs[g].push_back(v);
                }
            }
            // Sort neighbors by link speed desc, tie -> lower id: lane k
            // of dst d rides the k-th fastest ingress link.
            for (uint32_t a = 0; a + 1 < _cube_nbrs[g].size(); a++) {
                for (uint32_t b = a + 1; b < _cube_nbrs[g].size(); b++) {
                    uint32_t x = _cube_nbrs[g][a], y = _cube_nbrs[g][b];
                    if (local[g][y] > local[g][x]
                        || (local[g][y] == local[g][x] && y < x)) {
                        _cube_nbrs[g][a] = y;
                        _cube_nbrs[g][b] = x;
                    }
                }
            }
        }
        if (_nv_gbps > 0.0) {
            init_lambda_from_demand();
        }
    }

    // CUBE prior refinement (x29 finding): relay bytes reach the cube
    // links THROUGH the landing NICs, so a relay corridor's usable rate
    // is min(nic, link) -- weighting by raw link capacity over-feeds the
    // fast 400G links whose landing NIC (180G) is the real constraint
    // (symptom: capacity-weighted prior lost to uniform init, and C=4's
    // single over-weighted lane regressed to 7233us).
    // MOSAIC-LP (fixed topology => LP, unknown workload => lambda):
    // per-(j, d) intra-host path decompositions from the workload-free
    // min-congestion LP (tools/mosaic_lp_intra.py, TE-CCL-style mcf core
    // with the time dimension collapsed). At emit time the intra segment
    // of RX/TX (and same-host chains if `local` is set) is split across
    // the LP paths at fixed ratios; lambda keeps steering the corridor
    // mix, so the LP fixes SPACE (where bytes flow inside the host) and
    // the runtime fixes TIME (how much per corridor per wave).
    struct LpPath {
        double weight;
        vector<uint32_t> verts;  // local ids, from j to d inclusive
    };

    void set_lp_intra(const vector<vector<vector<LpPath> > >& table,
                      bool local_too) {
        if (_wiring != WIRING_CUBE) {
            throw runtime_error("mosaic: lp_intra needs cube wiring");
        }
        _lp_intra = table;
        _lp_local = local_too;
    }

    // MOSAIC-LP v2 (spread unification): jdist[r][rd] = LP marginal of
    // the TRANSIT rail j for a relay forced through a third rail (the
    // two-host relay LP with per-commodity bans on j in {r, rd}). When
    // installed, the spread lanes' emission collapses into ONE class
    // split across ALL transit rails by these weights: on a homogeneous
    // host the LP returns the uniform (Latin-like) spread that FAST's
    // balance step realizes structurally; on a heterogeneous host it
    // returns capacity-aware weights -- one mechanism, no homogeneity
    // detection anywhere. Rail (NVSwitch star) and cube use the same
    // table format; lambda still owns the corridor-level mix.
    void set_lp_spread(
            const vector<vector<vector<pair<double, uint32_t> > > >& jd) {
        if (_wiring != WIRING_CUBE && _wiring != WIRING_RAIL) {
            throw runtime_error("mosaic: lp_spread needs rail/cube wiring");
        }
        // Same skew gate as E3/F3b (demand is known at collective start;
        // this is workload structure, not topology detection): the
        // hot-column jdist trades last-mile locality for landing-NIC
        // balance, which only pays under concentrated ingress. Uniform
        // demand => gate closed => keep the hand lanes (measured: -4%
        // locality cost on uniform without the gate).
        if (_nv_gbps > 0.0 && _nic_gbps > 0.0) {
            vector<double> t;
            double t_max = ingress_bounds(t);
            if (!skew_gate(t, t_max)) {
                cout << "mosaic_lp_spread off gate skew" << endl;
                return;
            }
        }
        _lp_jdist = jd;
    }

    // CUBE lane rotation ablation: lane k picks the (k + h + r)-th
    // neighbor instead of the k-th fastest, so different sources spread
    // their lane bytes over different landing NICs (trades the clean
    // "lane = physical link" feedback signal for landing-NIC balance).
    void set_cube_lane_rot(bool on) {
        _cube_lane_rot = on;
    }

    void set_cube_nic_cap(bool on) {
        _cube_nic_cap = on;
        if (_wiring == WIRING_CUBE && !_cube_bin.empty()
            && _nv_gbps > 0.0) {
            init_lambda_from_demand();
        }
    }

    // Per-dst scale-up ingress capacity: B_in on cube, nv elsewhere.
    double nv_of(uint32_t d) const {
        if (_wiring == WIRING_CUBE && !_cube_bin.empty()) {
            return _cube_bin[d % _gpus];
        }
        return _nv_gbps;
    }

    // Compute the waterfilling prior from a DIFFERENT (e.g. stale) matrix:
    // models MoE distribution drift where the ratios were derived from the
    // previous iteration's traffic.
    // Diagnostic tracing (print-only, no behavior change): per-dst
    // completion events with the residual state of every unfinished dst.
    void set_trace(bool on) {
        _trace = on;
    }

    // E5 censoring fix: fed-but-undelivered corridors whose relay hop
    // advanced keep their belief instead of being zeroed.
    void set_hop_progress_fix(bool on) {
        _hop_progress_fix = on;
    }

    // E1 window donation: each dst-completion event grants the most-behind
    // unfinished dst one wave emission ahead of its delivery threshold.
    void set_donate_window(bool on) {
        _donate_window = on;
    }

    // E2 reactive re-waterfill: each dst-completion event re-derives every
    // unfinished dst's two-pipe waterfilling split from its RESIDUAL
    // un-emitted bytes and sets lambda directly (re-anchors feedback away
    // from contention-phase measurement artifacts). Rail wiring only.
    void set_donate_rewaterfill(bool on) {
        _donate_rewaterfill = on;
    }

    // E3 skew-aware epoch precedence: dsts with cut-bound slack are
    // demoted by up to K waves in the epoch encoding, so the predicted
    // straggler's chunks win shared ports from t=0 (generation-time
    // static priority derived from the known demand matrix).
    void set_skew_epoch(uint32_t k) {
        _skew_epoch = k;
        init_epoch_offsets();
    }

    // O1 pair coloring (34_opt exploration): sub-slot the epoch encoding
    // by the diagonal (d - s) mod K so that, at equal (wave, hop), every
    // source PREFERS a distinct destination (a Latin-square permutation
    // hint, the soft form of FAST's stage coordination). Min-epoch is a
    // preference, not a gate, so the fallback stays work-conserving and
    // skew robustness is untouched. K = 0 disables (default; epoch
    // encoding then stays bit-identical to the historical layout).
    void set_pair_color(uint32_t k) {
        _pair_color = k;
    }

    // O1 auto gate. The coloring serves the demand along the n-1 cyclic
    // diagonals D_c = {(s, s+c mod n)} -- the Cayley-table (BvN) family
    // of Z_n -- giving each diagonal an equal-priority time slice. That
    // slice assignment is load-aligned iff the diagonal sums are equal:
    //   sum(D_c) = const  <=>  the demand decomposes as an equal-weight
    //   BvN mixture of the n cyclic permutations.
    // Hot-dst demand satisfies this almost exactly (a column contributes
    // one entry to EVERY diagonal, so column skew cancels: measured diag
    // CoV 0.01-0.03), and uniform trivially (0). persrc demand does not
    // (0.24): big diagonals head-of-line-block their slice and the
    // measured effect flips negative (+4% at nv2400). Gate at CoV < 0.1.
    void set_pair_color_auto() {
        double mean = 0.0;
        vector<double> diag(_node_count, 0.0);
        for (uint32_t c = 1; c < _node_count; c++) {
            for (uint32_t s = 0; s < _node_count; s++) {
                diag[c] += (double)_demand[s][(s + c) % _node_count];
            }
            mean += diag[c];
        }
        uint32_t nd = _node_count - 1;
        mean /= nd;
        double var = 0.0;
        for (uint32_t c = 1; c < _node_count; c++) {
            var += (diag[c] - mean) * (diag[c] - mean);
        }
        double cov = mean > 0.0 ? sqrt(var / nd) / mean : 0.0;
        _pair_color = (cov < 0.1) ? _node_count : 0;
        cout << "mosaic_pair_color_auto diag_cov " << cov
             << " K " << _pair_color << endl;
    }

    // O3 lambda guard (34_opt exploration): before moving lambda toward
    // the measured rate shares, predict the relative wave-time gain of
    // rebalancing, 1 - 1/max_c(lambda_c/share_c); below `eps` the prior
    // is already near-waterfilling and the update is frozen (kills the
    // exploration-noise drift that makes feedback neutral-to-negative on
    // accurate priors). Never freezes while a fed corridor delivered
    // nothing (congestion evidence must keep flowing). 0 = off.
    void set_lambda_guard(double eps) {
        _guard_eps = eps;
    }

    // x41 MOSAIC-L (-mosaic_ledger): replace the per-dst lambda share
    // split with a GLOBAL RESOURCE LEDGER. Every resource the fabric
    // has (per-GPU NV up/down, per-GPU NIC up/down, per-(leaf,spine)
    // fabric segments) carries `outstanding` = bytes assigned to chains
    // whose path crosses it and not yet past it. A pair's wave quota is
    // waterfilled across its corridor mask so the resulting max
    // projected finish (outstanding/rate) over touched resources is
    // minimized -- the online, cross-dst generalization of the closed-
    // form prior (Prop 1 is the empty-ledger special case). Feedback
    // shrinks to per-resource effective-rate correction from busy-period
    // delivery spacing (the exogenous axis: degrade, interference);
    // lambda/censoring/trust/anti-windup/exploration-floor are unused on
    // this path. Wirings: RAIL and POD (others fall back to lambda).
    void set_ledger(double taper_gbps, uint32_t fab_spines,
                    bool rate_correct = true) {
        if (_wiring != WIRING_RAIL && _wiring != WIRING_POD) {
            cout << "mosaic_ledger off (wiring not rail/pod)" << endl;
            return;
        }
        _ledger = true;
        _lg_correct = rate_correct;
        _lg_spines = fab_spines == 0 ? 1 : fab_spines;
        _lg_leaves = (_wiring == WIRING_POD) ? _servers : _gpus;
        _lg_fab0 = 4 * _node_count;
        _lg_nres = _lg_fab0 + 2 * _lg_leaves * _lg_spines;
        double spine_gbps = taper_gbps / _lg_spines;
        _lg_chg.assign(_lg_nres, 0.0);
        _lg_dis.assign(_lg_nres, 0.0);
        _lg_dis_snap.assign(_lg_nres, 0.0);
        _lg_arr.assign(_lg_nres, 0);
        _lg_prev.assign(_lg_nres, 0);
        _lg_busyafter.assign(_lg_nres, 0);
        _lg_bacc.assign(_lg_nres, 0.0);
        _lg_tacc.assign(_lg_nres, 0.0);
        _lg_streak.assign(_lg_nres, 0);
        _lg_nom.assign(_lg_nres, _nic_gbps);
        for (uint32_t g = 0; g < _node_count; g++) {
            _lg_nom[g] = _nv_gbps;
            _lg_nom[_node_count + g] = _nv_gbps;
        }
        for (uint32_t i = _lg_fab0; i < _lg_nres; i++) {
            _lg_nom[i] = spine_gbps;
        }
        _lg_rate = _lg_nom;
        cout << "mosaic_ledger on nres " << _lg_nres
             << " leaves " << _lg_leaves << " spines " << _lg_spines
             << " spine_gbps " << spine_gbps << endl;
    }

    // E4 phase plan: generation-time progressive-filling fluid simulation
    // (NIC-ingress lanes + NV-ingress pipes, max-min shared) predicts the
    // completion order and a per-phase lambda table; at run time the k-th
    // dst completion switches every unfinished dst to the phase-k entry.
    // Rail wiring only.
    void set_phase_plan(bool on) {
        _phase_plan = on;
        if (on) {
            build_phase_plan();
        }
    }

    // F1 fixed wave SIZE (28_wave_ramp exploration): the wave becomes a
    // fixed ~`bytes` quantum at the dst ingress and the COUNT floats with
    // demand: W_d = max(wmin, ceil(col_bytes/bytes)). Hot dsts get many
    // waves (finer feedback + smaller relay chunks), cold dsts few.
    // 0 = off (legacy fixed-count `_waves`).
    void set_wave_bytes(uint64_t bytes, uint32_t wmin) {
        _wave_bytes = bytes;
        if (bytes == 0) {
            _dst_waves.clear();
            return;
        }
        if (wmin == 0) {
            wmin = 1;
        }
        _dst_waves.assign(_node_count, wmin);
        uint32_t w_max = wmin;
        for (uint32_t d = 0; d < _node_count; d++) {
            uint64_t col = _dst_expected[d];
            uint32_t w = (uint32_t)((col + bytes - 1) / bytes);
            _dst_waves[d] = max(wmin, max(w, 1u));
            w_max = max(w_max, _dst_waves[d]);
        }
        cout << "mosaic_wave_size bytes " << bytes
             << " wmin " << wmin << " w_max " << w_max << endl;
    }

    // F2 deadline-aligned epochs: dst d's wave w is encoded at slot
    // w + (W_max - W_d), i.e. every dst is scheduled to FINISH at the
    // same slot. Under min-epoch a small-demand dst automatically yields
    // ports to a large-demand dst at equal byte progress -- the adaptive,
    // demand-proportional form of E3 (self-gating: equal cols => offset 0).
    // Needs set_wave_bytes() first (fixed-count waves are all equal).
    void set_wave_align(bool on) {
        _align_offset.clear();
        if (!on || _dst_waves.empty()) {
            return;
        }
        uint32_t w_max = 0;
        for (uint32_t d = 0; d < _node_count; d++) {
            w_max = max(w_max, _dst_waves[d]);
        }
        _align_offset.assign(_node_count, 0);
        for (uint32_t d = 0; d < _node_count; d++) {
            _align_offset[d] = w_max - _dst_waves[d];
        }
    }

    // F3 corridor ramp (rail only): start every dst at `start` corridors
    // (nested sets) and widen by 1 per own emitted wave -- or per
    // dst-completion event if `event_driven` (ports freed by finished
    // dsts unlock extra spread lanes).
    void set_ramp(uint32_t start, bool event_driven) {
        _ramp_start = start;
        _ramp_event = event_driven;
    }

    // F3b cold-restrict ramp: only dsts with >=2x ingress-bound slack are
    // capped (their relay corridors stop consuming sibling NIC/NV ingress
    // during the contention phase); the predicted straggler keeps the full
    // corridor set from t=0. TWO safety gates, both from generation-time
    // arithmetic on the known demand matrix:
    //   gate 1 (skew, same as E3): no clearly dominant dst => noise, off.
    //   gate 2 (spine feasibility): capping cold to DIRECT forces its
    //   cross-rail bytes through the tapered spine; if the max per-rail
    //   drain time exceeds the straggler's ingress bound, the cap would
    //   mint a NEW binding cut (seen at nv2400 / taper90), so off.
    //   taper_gbps<=0 or force skips gate 2.
    void set_ramp_cold(uint32_t start, bool event_driven,
                       double taper_gbps, bool force) {
        _ramp_cold = start;
        _ramp_event = event_driven;
        _ramp_is_cold.clear();
        if (start == 0
            || (_wiring != WIRING_RAIL && _wiring != WIRING_CUBE)
            || _nv_gbps <= 0.0 || _nic_gbps <= 0.0) {
            _ramp_cold = 0;
            return;
        }
        vector<double> t;
        double t_max = ingress_bounds(t);
        if (!skew_gate(t, t_max)) {
            _ramp_cold = 0;
            cout << "mosaic_ramp_cold off gate skew" << endl;
            return;
        }
        _ramp_is_cold.assign(_node_count, 0);
        for (uint32_t d = 0; d < _node_count; d++) {
            _ramp_is_cold[d] = (t[d] * 2.0 <= t_max) ? 1 : 0;
        }
        if (force || taper_gbps <= 0.0) {
            return;
        }
        vector<double> rail_load(_gpus, 0.0);
        for (uint32_t d = 0; d < _node_count; d++) {
            if (!_ramp_is_cold[d]) {
                continue;
            }
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d || s / _gpus == d / _gpus
                    || s % _gpus == d % _gpus) {
                    continue;  // only cross-server cross-rail rides spine
                }
                rail_load[d % _gpus] += (double)_demand[s][d];
            }
        }
        double t_spine = 0.0;
        for (uint32_t r = 0; r < _gpus; r++) {
            t_spine = max(t_spine, rail_load[r] / taper_gbps);
        }
        cout << "mosaic_ramp_cold gate spine t_spine " << t_spine
             << " t_hot " << t_max
             << (t_spine > t_max ? " -> off" : " -> on") << endl;
        if (t_spine > t_max) {
            _ramp_cold = 0;
            _ramp_is_cold.clear();
        }
    }

    void set_prior_demand(const vector<vector<mem_b> >& prior) {
        _prior_demand = &prior;
        if (_nv_gbps > 0.0) {
            init_lambda_from_demand();
        }
    }

    // F6 (32 组): admission PACING of slack destinations. E3 lowers a cold
    // dst's port priority and F3b narrows its corridor set, but neither
    // limits its RATE: priority acts only on the local chunk-pool pick
    // (fabric queues are FIFO, enqueued cold bytes cannot be preempted)
    // and is work-conserving (cold fills every hot gap and gives it back
    // a store-and-forward chunk time later). Pacing withholds a cold
    // dst's next wave until the predicted straggler advances, so cold
    // bytes spread over the whole makespan instead of bursting into the
    // contention phase. THREE generation-time gates on known inputs:
    //   gate 1 (skew, same as E3/F3b): no clearly dominant dst => the
    //     slack ranking is noise, off.
    //   gate 2 (slack margin): only dsts with >= 2x margin on their
    //     isolated ingress bound are paced, so running them at the
    //     straggler's cadence cannot mint a new binding cut.
    //   gate 3 (ingress-bound model validity): the slack computation
    //     assumes ingress-bound completion. When the per-rail fabric
    //     uplink is thinner than one NIC (taper < nic), the DIRECT class
    //     is fabric-bound and t(d) underestimates real finish times, so
    //     the cold/hot classification is invalid => off.
    //     taper_gbps <= 0 (unknown) skips the gate.
    // 决策轨迹 dump(-mosaic_decision_trace <file>):每波每 (dst,src) 的
    // quota 与走廊切分,供真机实现 mosaic-rt 做移植对拍(Imp_Plan/01 §4 的
    // 金标输入)。只增诊断输出,不改任何决策路径;引入记录见
    // Impl/mosaic-rt/log.md「T5 决策对拍」。
    void set_decision_trace(const char* path) {
        _dtrace = fopen(path, "w");
        if (!_dtrace) {
            cout << "mosaic_decision_trace open fail " << path << endl;
        }
    }
    FILE* _dtrace = nullptr;

    void set_cold_pace(uint32_t k, double taper_gbps) {
        _cold_pace = k;
        _pace_is_cold.clear();
        if (k == 0 || _nv_gbps <= 0.0 || _nic_gbps <= 0.0) {
            _cold_pace = 0;
            return;
        }
        if (taper_gbps > 0.0 && taper_gbps < _nic_gbps) {
            _cold_pace = 0;
            cout << "mosaic_cold_pace off gate fabric taper " << taper_gbps
                 << " nic " << _nic_gbps << endl;
            return;
        }
        vector<double> t;
        double t_max = ingress_bounds(t);
        if (!skew_gate(t, t_max)) {
            _cold_pace = 0;
            cout << "mosaic_cold_pace off gate skew" << endl;
            return;
        }
        _pace_is_cold.assign(_node_count, 0);
        uint32_t cold = 0;
        for (uint32_t d = 0; d < _node_count; d++) {
            if (t[d] >= t_max) {
                _pace_hot = d;
            }
            _pace_is_cold[d] = (t[d] * 2.0 <= t_max) ? 1 : 0;
            cold += _pace_is_cold[d];
        }
        cout << "mosaic_cold_pace on k " << k << " hot " << _pace_hot
             << " cold_dsts " << cold << endl;
    }

    // F7 (32 组): split lambda state by SOURCE HOST. The corridor classes
    // are symmetric across source hosts only while the hardware is; one
    // degraded relay NIC breaks the symmetry INSIDE a class (host 0's TX
    // relay is slow, host 1/2's is not) and a per-dst lambda cannot
    // express that -- it can only tax or reward the whole class for every
    // source at once. Splitting the state gives the feedback the same
    // granularity as the fault. Default off; clean topologies measured
    // neutral.
    void set_host_lambda(bool on) {
        _host_lambda = on;
    }

    // Fixed "all-in on the spreader" heuristic init (naive strong
    // baseline): RX takes everything a pair allows, the rest floored.
    void set_rx_init() {
        if (!corridor_enabled(RX)) {
            throw runtime_error("mosaic: rx_init needs corridor RX enabled");
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            DstState& state = _dst_state[d];
            double others = 0.0;
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                if (c != RX && corridor_enabled(c)) {
                    state.lambda[c] = _floor;
                    others += _floor;
                } else if (c != RX) {
                    state.lambda[c] = 0.0;
                }
            }
            state.lambda[RX] = 1.0 - others;
        }
    }

    // Greedy waterfilling prior (no LP): for each destination, split its
    // cross-server inflow between the NIC-ingress corridors (DIRECT/TX)
    // and the NVLink-ingress corridors (RX/SPREAD3) so both ingress pipes
    // of the GPU would finish together:
    //   (same_server_in + x) / nv  =  (cross_in - x) / nic
    // Execution feedback then corrects for everything this one-liner
    // cannot see (shared fabric, other GPUs' interference, latency).
    void init_lambda_from_demand() {
        if (_nv_gbps <= 0.0 || _nic_gbps <= 0.0) {
            return;
        }
        const vector<vector<mem_b> >& prior = *_prior_demand;
        if (_wiring == WIRING_POD) {
            init_lambda_pod(prior);
            return;
        }
        if (_wiring == WIRING_RAIL2P) {
            init_lambda_rail2p(prior);
            return;
        }
        if (_wiring == WIRING_CUBE && !_cube_bin.empty()) {
            init_lambda_cube(prior);
            return;
        }
        bool tx_on = corridor_enabled(TX);
        bool rx_on = corridor_enabled(RX);
        uint32_t lanes = 0;
        for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                lanes++;
            }
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            double cross = 0.0, same = 0.0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d) {
                    continue;
                }
                if (s / _gpus == d / _gpus) {
                    same += (double)prior[s][d];
                } else {
                    cross += (double)prior[s][d];
                }
            }
            if (cross <= 0.0) {
                continue;
            }
            double x = (_nv_gbps * cross - _nic_gbps * same)
                / (_nv_gbps + _nic_gbps);
            x = max(0.0, min(x, cross));
            double nv_share = x / cross;      // via RX/spread lanes
            rail_shares_to_lambda(_dst_state[d], nv_share);
        }
    }

    // Distribute a dst's (nv_share, 1-nv_share) ingress-pipe split over
    // the enabled rail corridors (DIRECT/TX on the NIC pipe, RX/spread
    // lanes on the NV pipe), floor + normalize. Shared by the prior, the
    // completion-event re-waterfill and the phase plan.
    void rail_shares_to_lambda(DstState& state, double nv_share) {
        bool tx_on = corridor_enabled(TX);
        bool rx_on = corridor_enabled(RX);
        uint32_t lanes = 0;
        for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                lanes++;
            }
        }
        double nic_share = 1.0 - nv_share;
        if (!rx_on && lanes == 0) {         // NIC-only corridor set
            nic_share = 1.0;
            nv_share = 0.0;
        }
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            state.lambda[c] = 0.0;
        }
        state.lambda[DIRECT] = nic_share * (tx_on ? 2.0 / 3.0 : 1.0);
        if (tx_on) {
            state.lambda[TX] = nic_share / 3.0;
        }
        double rx_part = 0.0;
        if (rx_on) {
            rx_part = nv_share * (lanes > 0 ? 2.0 / 3.0 : 1.0);
            state.lambda[RX] = rx_part;
        }
        if (lanes > 0) {
            double lane_part = (nv_share - rx_part) / lanes;
            for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] = lane_part;
                }
            }
        }
        double sum = 0.0;
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                state.lambda[c] = max(_floor, state.lambda[c]);
                sum += state.lambda[c];
            }
        }
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                state.lambda[c] /= sum;
            }
        }
    }

    // CUBE prior: K-pipe waterfilling per dst over {NIC} + {each incident
    // cube link}. Same-host inflow is a FIXED load on its pinned path's
    // last link (it cannot ride the NIC); cross inflow is divisible.
    // The per-link takes then weight the relay corridors (lane k = k-th
    // fastest link), so a slow 100G link starts with a small share
    // instead of an equal one.
    void init_lambda_cube(const vector<vector<mem_b> >& prior) {
        for (uint32_t d = 0; d < _node_count; d++) {
            uint32_t rd = d % _gpus;
            uint32_t host = d / _gpus;
            const vector<uint32_t>& nbrs = _cube_nbrs[rd];
            double cross = 0.0;
            vector<double> cap, fixed;
            cap.push_back(_nic_gbps);
            fixed.push_back(0.0);
            for (uint32_t k = 0; k < nbrs.size(); k++) {
                double c = _cube_local[nbrs[k]][rd];
                cap.push_back(_cube_nic_cap ? min(c, _nic_gbps) : c);
                fixed.push_back(0.0);
            }
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d) {
                    continue;
                }
                if (s / _gpus == host) {
                    const vector<uint32_t>& path =
                        _cube_paths[s % _gpus][rd];
                    uint32_t last = path[path.size() - 2];
                    for (uint32_t k = 0; k < nbrs.size(); k++) {
                        if (nbrs[k] == last) {
                            fixed[1 + k] += (double)prior[s][d];
                            break;
                        }
                    }
                } else {
                    cross += (double)prior[s][d];
                }
            }
            if (cross <= 0.0) {
                continue;
            }
            vector<double> take = waterfill_pipes(cap, fixed, cross);
            vector<double> link_take(take.begin() + 1, take.end());
            cube_shares_to_lambda(_dst_state[d], rd, take[0] / cross,
                                  &link_take);
        }
    }

    // Map a dst's (nic_share, 1-nic_share) ingress split onto the cube
    // corridors. Relay corridors are weighted by landing-link capacity
    // (or by the prior's per-link waterfill takes when given): lane k =
    // k-th fastest incident link of rd; RX's landing link varies per
    // src, so it gets the mean weight. Shared by the prior and the E2
    // completion-event re-waterfill.
    void cube_shares_to_lambda(DstState& state, uint32_t rd,
                               double nic_share,
                               const vector<double>* link_take) {
        bool tx_on = corridor_enabled(TX);
        bool rx_on = corridor_enabled(RX);
        const vector<uint32_t>& nbrs = _cube_nbrs[rd];
        double w_lane[MAX_CORRIDORS];
        double w_sum = 0.0, w_mean = 0.0;
        uint32_t lanes = 0;
        for (uint32_t k = 0; k < nbrs.size(); k++) {
            double w = link_take ? (*link_take)[k]
                                 : _cube_local[nbrs[k]][rd];
            if (_cube_nic_cap) {
                w = min(w, _nic_gbps);
            }
            w_mean += w;
        }
        w_mean /= max((size_t)1, nbrs.size());
        for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
            uint32_t k = c - SPREAD3;
            w_lane[c] = 0.0;
            if (corridor_enabled(c) && k < nbrs.size()) {
                w_lane[c] = link_take ? (*link_take)[k]
                                      : _cube_local[nbrs[k]][rd];
                if (_cube_nic_cap) {
                    w_lane[c] = min(w_lane[c], _nic_gbps);
                }
                w_sum += w_lane[c];
                lanes++;
            }
        }
        double w_rx = rx_on ? w_mean : 0.0;
        w_sum += w_rx;
        double nv_share = 1.0 - nic_share;
        if (!rx_on && lanes == 0) {
            nic_share = 1.0;
            nv_share = 0.0;
        }
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            state.lambda[c] = 0.0;
        }
        state.lambda[DIRECT] = nic_share * (tx_on ? 2.0 / 3.0 : 1.0);
        if (tx_on) {
            state.lambda[TX] = nic_share / 3.0;
        }
        if (w_sum > 0.0) {
            if (rx_on) {
                state.lambda[RX] = nv_share * w_rx / w_sum;
            }
            for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
                if (w_lane[c] > 0.0) {
                    state.lambda[c] = nv_share * w_lane[c] / w_sum;
                }
            }
        } else if (rx_on) {
            state.lambda[RX] = nv_share;
        }
        double sum = 0.0;
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                state.lambda[c] = max(_floor, state.lambda[c]);
                sum += state.lambda[c];
            }
        }
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                state.lambda[c] /= sum;
            }
        }
    }

    // POD prior: SAME two-pipe waterfilling as rail (§4.2) -- the dst
    // still has exactly an NV pipe (all RX lanes) and its own NIC pipe
    // (DIRECT); the split WITHIN the NV group is uniform across lanes
    // (feedback differentiates spines/NICs at run time).
    void init_lambda_pod(const vector<vector<mem_b> >& prior) {
        uint32_t lanes = 0;
        for (uint32_t c = 1; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                lanes++;
            }
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            double cross = 0.0, same = 0.0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d) {
                    continue;
                }
                if (s / _gpus == d / _gpus) {
                    same += (double)prior[s][d];
                } else {
                    cross += (double)prior[s][d];
                }
            }
            if (cross <= 0.0) {
                continue;
            }
            double x = (_nv_gbps * cross - _nic_gbps * same)
                / (_nv_gbps + _nic_gbps);
            x = max(0.0, min(x, cross));
            double nv_share = lanes > 0 ? x / cross : 0.0;
            DstState& state = _dst_state[d];
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                state.lambda[c] = 0.0;
            }
            state.lambda[DIRECT] = 1.0 - nv_share;
            for (uint32_t c = 1; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] = nv_share / lanes;
                }
            }
            double sum = 0.0;
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] = max(_floor, state.lambda[c]);
                    sum += state.lambda[c];
                }
            }
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] /= sum;
                }
            }
        }
    }

    // RAIL2P prior: TRUE 3-pipe waterfilling. dst ingress pipes =
    // {NV (all relayed corridors), NIC-A (even-pair DIRECT), NIC-B
    // (odd-pair DIRECT, possibly a slower hardware plane if
    // planeb_factor>1 is KNOWN static heterogeneity)}. Uses the generic
    // waterfill_pipes(); with symmetric planes this reduces to the
    // two-pipe closed form.
    void init_lambda_rail2p(const vector<vector<mem_b> >& prior) {
        double beta_a = _nic_gbps / 2.0;
        double beta_b = _nic_gbps / 2.0 / _planeb_factor;
        uint32_t lanes = 0;
        for (uint32_t c = R2P_SPREAD; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                lanes++;
            }
        }
        bool tx_on = corridor_enabled(R2P_TX);
        bool rx_on = corridor_enabled(R2P_RX);
        for (uint32_t d = 0; d < _node_count; d++) {
            double cross = 0.0, same = 0.0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d) {
                    continue;
                }
                if (s / _gpus == d / _gpus) {
                    same += (double)prior[s][d];
                } else {
                    cross += (double)prior[s][d];
                }
            }
            if (cross <= 0.0) {
                continue;
            }
            vector<double> cap, fixed;
            cap.push_back(_nv_gbps);  fixed.push_back(same);
            cap.push_back(beta_a);    fixed.push_back(0.0);
            cap.push_back(beta_b);    fixed.push_back(0.0);
            vector<double> take = waterfill_pipes(cap, fixed, cross);
            double nv_share = take[0] / cross;
            DstState& state = _dst_state[d];
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                state.lambda[c] = 0.0;
            }
            state.lambda[R2P_DIRECT_A] = take[1] / cross;
            state.lambda[R2P_DIRECT_B] = take[2] / cross;
            double rx_part = 0.0;
            if (rx_on) {
                rx_part = nv_share
                    * ((tx_on || lanes > 0) ? 2.0 / 3.0 : 1.0);
                state.lambda[R2P_RX] = rx_part;
            }
            double rest = nv_share - rx_part;
            uint32_t others = (tx_on ? 1 : 0) + lanes;
            if (others > 0) {
                if (tx_on) {
                    state.lambda[R2P_TX] = rest / others;
                }
                for (uint32_t c = R2P_SPREAD; c < MAX_CORRIDORS; c++) {
                    if (corridor_enabled(c)) {
                        state.lambda[c] = rest / others;
                    }
                }
            }
            double sum = 0.0;
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] = max(_floor, state.lambda[c]);
                    sum += state.lambda[c];
                }
            }
            for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
                if (corridor_enabled(c)) {
                    state.lambda[c] /= sum;
                }
            }
        }
    }

    // src-driven dispatch phase (mirrors the runtime's MOSAIC_SRC_DRIVEN +
    // phase=1 path; enable with -mosaic_src_driven). Requires the ledger:
    // the src allocator is ledger waterfilling; the lambda/pacing/ramp
    // machinery is dst-wave-centric and stays off in this mode.
    void set_src_driven(bool on) {
        _src_driven = on;
        if (on) {
            _src_state.resize(_node_count);
        }
    }

    void start() {
        if (_src_driven) {
            if (!_ledger) {
                throw runtime_error("mosaic: src_driven requires the ledger");
            }
            for (uint32_t s = 0; s < _node_count; s++) {
                emit_src_wave(s);
            }
            return;
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            emit_wave(d);
        }
    }

    bool complete() const {
        return _delivered_app_bytes >= _expected_app_bytes;
    }

    uint64_t expected_app_bytes() const {
        return _expected_app_bytes;
    }

    uint64_t app_chains() const {
        return _chains.size();
    }

    uint64_t emitted_hop_chunks() const {
        return _hops_emitted;
    }

    uint64_t emitted_hop_bytes() const {
        return _hop_bytes_emitted;
    }

    void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                         uint32_t dst_node, mem_b bytes) override {
        (void)bytes;
        if (app_flow_id >= _chains.size()) {
            throw runtime_error("mosaic: delivery for unknown chain");
        }
        ChainState& chain = *_chains[app_flow_id];
        if (chunk_id != chain.next_hop
            || chain.hops[chain.next_hop].dst != dst_node) {
            throw runtime_error("mosaic: out-of-order hop delivery");
        }
        if (_ledger) {
            const Hop& done = chain.hops[chain.next_hop];
            lg_hop_delivered(chain.hops[0].src, done.src, done.dst,
                             chain.bytes);
        }
        // src-driven: hop-0 delivery of a cross-server chain = the sender's
        // ACK signal (rt: RC send CQ completion). Feed the src's wave gate
        // and advance its wave once the threshold is crossed.
        if (_src_driven && chain.next_hop == 0
            && server_of(chain.hops[0].src) != server_of(chain.hops[0].dst)) {
            uint32_t s = chain.hops[0].src;
            SrcState& sst = _src_state[s];
            if (chain.wave < sst.stats.size()) {
                SrcWaveStats& sw = sst.stats[chain.wave];
                sw.acked += chain.bytes;
                if (chain.wave == sst.emit_wave - 1 && !sst.next_triggered
                    && sw.emitted_nic > 0
                    && sw.acked * 100 >= (uint64_t)(_threshold * 100)
                                          * sw.emitted_nic) {
                    sst.next_triggered = true;
                    emit_src_wave(s);
                }
            }
        }

        chain.next_hop++;
        if (chain.next_hop < chain.hops.size()) {
            if (chain.mixed && !_src_driven) {
                // Relay hop advanced: the corridor is congested-but-alive,
                // which the E5 censoring fix distinguishes from dead.
                _dst_state[chain.dst].stats[chain.wave]
                    .hop_progress[chain.corridor] = true;
            }
            inject_hop(chain);
            return;
        }

        // Chain complete at the application endpoint.
        _delivered_app_bytes += chain.bytes;
        _dst_delivered[chain.dst] += chain.bytes;
        if (_dst_delivered[chain.dst] >= _dst_expected[chain.dst]
            && _dst_expected[chain.dst] > 0
            && !_dst_done[chain.dst]) {
            _dst_done[chain.dst] = true;
            _done_count++;
            if (_trace) {
                trace_dst_done(chain.dst);
            }
            on_dst_complete();
        }
        if (_src_driven) {
            return;   // dst wave stats are never sized in src-driven mode
        }
        DstState& state = _dst_state[chain.dst];
        WaveStats& wave = state.stats[chain.wave];
        wave.delivered += chain.bytes;
        if (chain.mixed) {
            wave.corridor_bytes[chain.corridor] += chain.bytes;
            wave.corridor_last[chain.corridor] = EventList::now();
            if (!wave.hbytes.empty()) {
                wave.hbytes[chain.src_host][chain.corridor] += chain.bytes;
                wave.hlast[chain.src_host][chain.corridor] =
                    EventList::now();
            }
        }
        if (_ledger && _lg_spineagg) {
            if (chain.fab) {
                wave.fab_bytes += chain.bytes;
                wave.fab_last = EventList::now();
            } else {
                wave.nofab_bytes += chain.bytes;
                wave.nofab_last = EventList::now();
            }
        }

        if (chain.wave == state.emit_wave - 1
            && !state.next_triggered
            && wave.delivered * 100 >= (uint64_t)(_threshold * 100)
                                        * (uint64_t)wave.emitted) {
            state.next_triggered = true;
            update_lambda(chain.dst, wave);
            if (_ledger && _lg_spineagg) {
                update_phi(chain.dst, wave);
            }
            print_wave(chain.dst, chain.wave, wave);
            emit_wave(chain.dst);
        }
    }

    void print_report() const {
        cout << "mosaic_report gpus_per_server " << _gpus
             << " wiring " << (_wiring == WIRING_POD ? "pod"
                               : (_wiring == WIRING_RAIL2P ? "rail2p"
                                                           : "rail"))
             << " corridors " << _corridor_count
             << " waves " << _waves
             << " adapt " << (_adapt ? "on" : "off")
             << " gamma " << _gamma
             << " floor " << _floor
             << " threshold " << _threshold
             << " chains " << _chains.size()
             << " hop_chunks " << _hops_emitted
             << " hop_bytes " << _hop_bytes_emitted
             << " app_bytes " << _delivered_app_bytes
             << endl;
        mem_b ctot[CORRIDORS] = {0, 0, 0, 0};
        for (uint32_t d = 0; d < _node_count; d++) {
            for (uint32_t w = 0; w < _dst_state[d].stats.size(); w++) {
                for (uint32_t c = 0; c < CORRIDORS; c++) {
                    ctot[c] += _dst_state[d].stats[w].corridor_emitted[c];
                }
            }
        }
        cout << "mosaic_corridor_bytes";
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            cout << " c" << c << " " << ctot[c];
        }
        cout << endl;
        if (_ledger && (_lg_bcast || _lg_forkprobe)) {
            // channel B volume: one policy row = (src id, corridor,
            // bytes) ~ 8B, plus a ~16B header per publication
            double avg = _lg_pub_cnt > 0
                ? (double)_lg_pub_entries / (double)_lg_pub_cnt : 0.0;
            cout << "mosaic_lg_proto pubs " << _lg_pub_cnt
                 << " entries " << _lg_pub_entries
                 << " avg_bytes_per_pub " << (16.0 + 8.0 * avg)
                 << " fork_frac "
                 << (_lg_fork_total > 0.0
                     ? _lg_fork_bytes / _lg_fork_total : 0.0)
                 << endl;
        }
    }

private:
    struct Hop {
        uint32_t src;
        uint32_t dst;
    };

    struct ChainState {
        ChainState() : packet_flow(NULL) {
        }

        uint32_t id;
        vector<Hop> hops;
        mem_b bytes;
        uint32_t corridor;
        bool mixed;  // participates in lambda feedback
        uint32_t dst;
        uint32_t wave;
        uint32_t src_host;   // F7: which source host emitted this chain
        bool fab = false;    // x44: crosses the switching fabric
        uint32_t next_hop;
        packetid_t next_packet_id;
        PacketFlow packet_flow;
    };

    struct WaveStats {
        uint64_t emitted = 0;
        uint64_t delivered = 0;
        simtime_picosec start = 0;
        mem_b corridor_emitted[CORRIDORS] = {0, 0, 0, 0};
        mem_b corridor_bytes[CORRIDORS] = {0, 0, 0, 0};
        simtime_picosec corridor_last[CORRIDORS] = {0, 0, 0, 0};
        bool hop_progress[CORRIDORS] = {false, false, false, false};
        // x44 spineagg: fabric-vs-not split for the outer loop (phi)
        mem_b fab_emitted = 0, nofab_emitted = 0;
        mem_b fab_bytes = 0, nofab_bytes = 0;
        simtime_picosec fab_last = 0, nofab_last = 0;
        // F7 per-source-host slices (allocated only under -mosaic_host_lambda)
        vector<vector<mem_b> > hemitted;
        vector<vector<mem_b> > hbytes;
        vector<vector<simtime_picosec> > hlast;
    };

    struct DstState {
        DstState() {
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                lambda[c] = 1.0 / CORRIDORS;
            }
        }

        double lambda[CORRIDORS];
        vector<vector<double> > hlambda;   // F7: [src host][corridor]
        uint32_t emit_wave = 0;      // next wave index to emit
        bool next_triggered = false;
        bool paced = false;          // F6: wave withheld, waiting on hot
        vector<WaveStats> stats;
    };

    // src-driven dispatch (task 2, File/design-global-view-src-driven-0823.md):
    // decision goes to the side with EQUAL totals. In the dispatch phase every
    // src emits the same volume, so each src waterfills its own ROW; the wave
    // becomes the src's pipelining unit, advanced when the previous wave's
    // cross-server hop-0 deliveries (the sim analogue of RC ACK: the byte has
    // left the sender and landed at the next hop) pass `_threshold`.
    struct SrcWaveStats {
        uint64_t emitted = 0;        // all bytes emitted this wave
        uint64_t emitted_nic = 0;    // bytes whose hop 0 crosses servers
        uint64_t acked = 0;          // of those, hop 0 delivered
    };
    struct SrcState {
        uint32_t emit_wave = 0;
        bool next_triggered = false;
        vector<SrcWaveStats> stats;
    };

    void set_uniform_lambda(DstState& state) {
        uint32_t enabled = 0;
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            if (corridor_enabled(c)) {
                enabled++;
            }
        }
        for (uint32_t c = 0; c < MAX_CORRIDORS; c++) {
            state.lambda[c] = corridor_enabled(c) ? 1.0 / enabled : 0.0;
        }
    }

    // Per-dst wave budget: fixed count (`_waves`) unless the fixed-SIZE
    // mode (F1) computed a demand-proportional count.
    uint32_t dst_wave_limit(uint32_t d) const {
        return _wave_bytes > 0 ? _dst_waves[d] : _waves;
    }

    // F3 dynamic corridor cap for (dst, wave): nested-set prefix of size
    // `cap`, growing by 1 per own wave (or per completion event).
    uint32_t ramp_cap(uint32_t d, uint32_t w) const {
        if (_wiring != WIRING_RAIL && _wiring != WIRING_CUBE) {
            return _corridor_count;
        }
        uint32_t growth = _ramp_event ? _done_count : w;
        if (_ramp_cold > 0) {
            if (!_ramp_is_cold.empty() && _ramp_is_cold[d]) {
                return min(_corridor_count, _ramp_cold + growth);
            }
            return _corridor_count;
        }
        if (_ramp_start > 0) {
            return min(_corridor_count, _ramp_start + growth);
        }
        return _corridor_count;
    }

    uint32_t server_of(uint32_t gpu) const {
        return gpu / _gpus;
    }

    uint32_t rail_of(uint32_t gpu) const {
        return gpu % _gpus;
    }

    // CUBE spread lanes: lane k of pair (r -> rd) is the k-th fastest
    // cube neighbor of rd, skipping r (r's link is RX's landing).
    uint32_t cube_lane_count(uint32_t r, uint32_t rd) const {
        uint32_t count = 0;
        for (uint32_t k = 0; k < _cube_nbrs[rd].size(); k++) {
            if (_cube_nbrs[rd][k] != r) {
                count++;
            }
        }
        return count;
    }

    uint32_t cube_lane_pick(uint32_t r, uint32_t rd, uint32_t lane,
                            uint32_t src_host) const {
        uint32_t navail = cube_lane_count(r, rd);
        uint32_t want = lane;
        if (_cube_lane_rot) {
            want = (lane + src_host + r) % navail;
        }
        uint32_t seen = 0;
        for (uint32_t k = 0; k < _cube_nbrs[rd].size(); k++) {
            if (_cube_nbrs[rd][k] == r) {
                continue;
            }
            if (seen == want) {
                return _cube_nbrs[rd][k];
            }
            seen++;
        }
        throw runtime_error("mosaic: cube lane out of range");
    }

    // Landing GPUs j on dst host whose pinned spine (s + host(d)*G + j)
    // % S equals k (explicit per-spine lanes, X2-full).
    vector<uint32_t> pod_spine_candidates(uint32_t s, uint32_t d,
                                          uint32_t k) const {
        vector<uint32_t> cand;
        uint32_t n = server_of(d);
        for (uint32_t j = 0; j < _gpus; j++) {
            if (j == rail_of(d)) {
                continue;
            }
            if ((s + n * _gpus + j) % _spine_count == k) {
                cand.push_back(j);
            }
        }
        return cand;
    }

    // Generic K-pipe greedy waterfilling (§4.2 fallback made concrete):
    // pipe i has capacity cap[i] and a fixed load fixed[i] that can only
    // ride pipe i; `divisible` bytes may be split freely. Returns the
    // divisible share per pipe that equalizes finish times (equal-time
    // waterfilling; pipes whose fixed load already exceeds the water
    // level get zero).
    static vector<double> waterfill_pipes(const vector<double>& cap,
                                          const vector<double>& fixed,
                                          double divisible) {
        uint32_t k = (uint32_t)cap.size();
        vector<double> take(k, 0.0);
        // Binary search the common finish time T.
        double lo = 0.0, hi = 0.0;
        double cap_sum = 0.0;
        for (uint32_t i = 0; i < k; i++) {
            lo = max(lo, fixed[i] / cap[i]);
            cap_sum += cap[i];
        }
        hi = lo + divisible / max(cap_sum, 1e-9) + 1e-9;
        for (int it = 0; it < 60; it++) {
            double mid = (lo + hi) / 2.0;
            double room = 0.0;
            for (uint32_t i = 0; i < k; i++) {
                room += max(0.0, mid * cap[i] - fixed[i]);
            }
            if (room < divisible) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double total = 0.0;
        for (uint32_t i = 0; i < k; i++) {
            take[i] = max(0.0, hi * cap[i] - fixed[i]);
            total += take[i];
        }
        if (total > 0.0) {
            for (uint32_t i = 0; i < k; i++) {
                take[i] *= divisible / total;
            }
        }
        return take;
    }

    // Applicable corridor mask for a cross-server pair.
    void pair_corridors(uint32_t s, uint32_t d, bool* usable) const {
        if (_wiring == WIRING_POD) {
            usable[DIRECT] = true;
            for (uint32_t c = 1; c < MAX_CORRIDORS; c++) {
                if (_spine_count > 0) {
                    // Explicit per-spine lane: lane m pins spine m%S; needs
                    // a landing j != rail(d) on the dst host with
                    // (s+host(d)*G+j) % S == m%S.
                    usable[c] = corridor_enabled(c)
                        && !pod_spine_candidates(s, d, (c - 1) % _spine_count)
                               .empty();
                } else {
                    // Implicit: lane m = m-th landing (spine follows).
                    usable[c] = corridor_enabled(c) && (c - 1 < _gpus - 1);
                }
            }
            return;
        }
        if (_wiring == WIRING_RAIL2P) {
            bool even = ((s + d) % 2) == 0;
            bool cross_rail = rail_of(s) != rail_of(d);
            usable[R2P_DIRECT_A] = even;
            usable[R2P_DIRECT_B] = !even && corridor_enabled(R2P_DIRECT_B);
            usable[R2P_TX] = cross_rail && corridor_enabled(R2P_TX);
            usable[R2P_RX] = cross_rail && corridor_enabled(R2P_RX);
            for (uint32_t c = R2P_SPREAD; c < MAX_CORRIDORS; c++) {
                usable[c] = corridor_enabled(c)
                    && _gpus > 2 + (c - R2P_SPREAD);
            }
            // Odd pair with DIRECT_B disabled still needs an outlet.
            if (!even && !usable[R2P_DIRECT_B] && !usable[R2P_TX]
                && !usable[R2P_RX]) {
                usable[R2P_DIRECT_B] = true;
            }
            return;
        }
        bool cross_rail = rail_of(s) != rail_of(d);
        usable[DIRECT] = true;
        usable[TX] = cross_rail && corridor_enabled(TX);
        usable[RX] = cross_rail && corridor_enabled(RX);
        if (_wiring == WIRING_CUBE) {
            if (!_lp_jdist.empty()) {
                bool any = !_lp_jdist[rail_of(s)][rail_of(d)].empty();
                for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
                    usable[c] = corridor_enabled(c) && any;
                }
                return;
            }
            uint32_t navail = cube_lane_count(rail_of(s), rail_of(d));
            for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
                usable[c] = corridor_enabled(c)
                    && (c - SPREAD3) < navail;
            }
            return;
        }
        for (uint32_t c = SPREAD3; c < MAX_CORRIDORS; c++) {
            // Spread lane k needs k+1 distinct transit rails j
            // (j != r, j != rd).
            usable[c] = corridor_enabled(c) && _gpus > 2 + (c - SPREAD3);
        }
    }

    vector<Hop> corridor_hops(uint32_t s, uint32_t d,
                              uint32_t corridor) const {
        vector<Hop> hops;
        uint32_t h = server_of(s), n = server_of(d);
        uint32_t r = rail_of(s), rd = rail_of(d);
        if (_wiring == WIRING_RAIL2P) {
            if (corridor == R2P_DIRECT_A || corridor == R2P_DIRECT_B) {
                hops.push_back(Hop{s, d});
            } else if (corridor == R2P_TX) {
                uint32_t relay = h * _gpus + rd;
                hops.push_back(Hop{s, relay});
                hops.push_back(Hop{relay, d});
            } else if (corridor == R2P_RX) {
                uint32_t landing = n * _gpus + r;
                hops.push_back(Hop{s, landing});
                hops.push_back(Hop{landing, d});
            } else {
                uint32_t lane = corridor - R2P_SPREAD;
                uint32_t j = (rd + 1 + h + r) % _gpus;
                uint32_t picked = 0;
                while (j == r || j == rd || picked < lane) {
                    if (j != r && j != rd) {
                        picked++;
                    }
                    j = (j + 1) % _gpus;
                }
                hops.push_back(Hop{s, h * _gpus + j});
                hops.push_back(Hop{h * _gpus + j, n * _gpus + j});
                hops.push_back(Hop{n * _gpus + j, d});
            }
            return hops;
        }
        if (_wiring == WIRING_POD) {
            if (corridor == DIRECT) {
                hops.push_back(Hop{s, d});
                return hops;
            }
            if (_spine_count > 0) {
                // Explicit per-spine lane: rotate within the spine's
                // landing set by (s+d+lane) for load balance.
                uint32_t lane = corridor - 1;
                vector<uint32_t> cand =
                    pod_spine_candidates(s, d, lane % _spine_count);
                uint32_t j = cand[(s + d + lane / _spine_count)
                                  % cand.size()];
                uint32_t landing = n * _gpus + j;
                hops.push_back(Hop{s, landing});
                hops.push_back(Hop{landing, d});
                return hops;
            }
            // RX lane m: the m-th landing candidate on the dst host,
            // start offset rotated by the source so different sources
            // spread over landings (and thus NICs AND spines).
            uint32_t lane = corridor - 1;
            uint32_t pick = (r + 1 + lane) % _gpus;
            while (pick == rd) {
                pick = (pick + 1) % _gpus;
            }
            // ensure lane-distinctness after the rd skip
            uint32_t seen = 0;
            uint32_t j = (r + 1) % _gpus;
            while (true) {
                if (j != rd) {
                    if (seen == lane) {
                        break;
                    }
                    seen++;
                }
                j = (j + 1) % _gpus;
            }
            uint32_t landing = n * _gpus + j;
            hops.push_back(Hop{s, landing});
            hops.push_back(Hop{landing, d});
            return hops;
        }
        if (_wiring == WIRING_CUBE && corridor >= SPREAD3) {
            // Lane k: land on the k-th fastest cube neighbor j of rd
            // (skipping r): intra cube path r->j, same-rail NIC j, then
            // the single link j->d.
            uint32_t j = cube_lane_pick(r, rd, corridor - SPREAD3, h);
            hops.push_back(Hop{s, h * _gpus + j});
            hops.push_back(Hop{h * _gpus + j, n * _gpus + j});
            hops.push_back(Hop{n * _gpus + j, d});
            return hops;
        }
        if (corridor == DIRECT) {
            hops.push_back(Hop{s, d});
        } else if (corridor == TX) {
            uint32_t relay = h * _gpus + rd;
            hops.push_back(Hop{s, relay});
            hops.push_back(Hop{relay, d});
        } else if (corridor == RX) {
            uint32_t landing = n * _gpus + r;
            hops.push_back(Hop{s, landing});
            hops.push_back(Hop{landing, d});
        } else {
            // Rotate the transit rail j over sources; j != r and j != rd
            // so it adds a genuinely different rail and no spine. Extra
            // spread lanes (corridor > SPREAD3) take the lane-th valid j,
            // so co-enabled lanes of one pair ride distinct rails.
            uint32_t lane = corridor - SPREAD3;
            uint32_t j = (rd + 1 + h + r) % _gpus;
            uint32_t picked = 0;
            while (j == r || j == rd || picked < lane) {
                if (j != r && j != rd) {
                    picked++;
                }
                j = (j + 1) % _gpus;
            }
            hops.push_back(Hop{s, h * _gpus + j});
            hops.push_back(Hop{h * _gpus + j, n * _gpus + j});
            hops.push_back(Hop{n * _gpus + j, d});
        }
        return hops;
    }

    // MOSAIC-LP splitter: returns false when LP splitting does not apply
    // (no table / not a corridor with an intra segment / tiny quota), in
    // which case the caller falls back to the single pinned-route chain.
    bool lp_split_emit(uint32_t s, uint32_t d, uint32_t corridor,
                       bool mixed, mem_b bytes, uint32_t wave) {
        if (_lp_intra.empty() || _wiring != WIRING_CUBE
            || bytes < 2048) {
            return false;
        }
        uint32_t r = rail_of(s), rd = rail_of(d);
        uint32_t h = server_of(s), n = server_of(d);
        bool same_host = (h == n);
        if (same_host) {
            if (!_lp_local || r == rd) {
                return false;
            }
        } else if (corridor != RX && corridor != TX) {
            return false;
        } else if (r == rd) {
            return false;
        }
        const vector<LpPath>& paths = _lp_intra[r][rd];
        if (paths.empty()) {
            return false;
        }
        mem_b assigned = 0;
        for (uint32_t p = 0; p < paths.size(); p++) {
            mem_b share = (p + 1 == paths.size())
                ? bytes - assigned
                : ((mem_b)(paths[p].weight * bytes) / 1024) * 1024;
            share = min(share, bytes - assigned);
            assigned += share;
            if (share <= 0) {
                continue;
            }
            const vector<uint32_t>& v = paths[p].verts;
            vector<Hop> hops;
            if (same_host) {
                for (uint32_t i = 0; i + 1 < v.size(); i++) {
                    hops.push_back(Hop{h * _gpus + v[i],
                                       h * _gpus + v[i + 1]});
                }
            } else if (corridor == RX) {
                hops.push_back(Hop{s, n * _gpus + v[0]});
                for (uint32_t i = 0; i + 1 < v.size(); i++) {
                    hops.push_back(Hop{n * _gpus + v[i],
                                       n * _gpus + v[i + 1]});
                }
            } else {  // TX: intra path on the src host, then rail NIC
                for (uint32_t i = 0; i + 1 < v.size(); i++) {
                    hops.push_back(Hop{h * _gpus + v[i],
                                       h * _gpus + v[i + 1]});
                }
                hops.push_back(Hop{h * _gpus + rd, d});
            }
            emit_chain_hops(s, d, corridor, mixed, share, wave, hops);
        }
        return true;
    }

    void emit_chain(uint32_t s, uint32_t d, uint32_t corridor, bool mixed,
                    mem_b bytes, uint32_t wave) {
        if (lp_split_emit(s, d, corridor, mixed, bytes, wave)) {
            return;
        }
        emit_chain_hops(s, d, corridor, mixed, bytes, wave,
                        corridor_hops(s, d, corridor));
    }

    void emit_chain_hops(uint32_t s, uint32_t d, uint32_t corridor,
                         bool mixed, mem_b bytes, uint32_t wave,
                         const vector<Hop>& hops) {
        ChainState* chain = new ChainState();
        chain->id = (uint32_t)_chains.size();
        chain->src_host = server_of(s);
        chain->hops = hops;
        chain->bytes = bytes;
        chain->corridor = corridor;
        chain->mixed = mixed;
        chain->dst = d;
        chain->wave = wave;
        chain->next_hop = 0;
        chain->next_packet_id = 1;
        _chains.push_back(chain);
        if (_src_driven) {
            // wave stats live on the deciding SRC; dst wave stats are never
            // sized in this mode (indexing them would be out of bounds)
            SrcWaveStats& ss = _src_state[s].stats[wave];
            ss.emitted += bytes;
            if (server_of(hops[0].src) != server_of(hops[0].dst)) {
                ss.emitted_nic += bytes;
            }
        } else {
            WaveStats& ws = _dst_state[d].stats[wave];
            ws.emitted += bytes;
            if (mixed) {
                ws.corridor_emitted[corridor] += bytes;
                if (!ws.hemitted.empty()) {
                    ws.hemitted[chain->src_host][corridor] += bytes;
                }
            }
        }
        if (_ledger && _lg_spineagg) {
            chain->fab = false;
            for (uint32_t h = 0; h < hops.size(); h++) {
                if (lg_hop_fab(hops[h].src, hops[h].dst)) {
                    chain->fab = true;
                    break;
                }
            }
            if (!_src_driven) {   // phi outer loop is dst-wave feedback
                WaveStats& ws = _dst_state[d].stats[wave];
                (chain->fab ? ws.fab_emitted : ws.nofab_emitted) += bytes;
            }
        }
        if (_ledger) {
            // charge attribution goes to the DECIDER (single writer of the
            // channel-B entry): the dst whose wave allocated the chain, or
            // the src in src-driven dispatch.
            lg_charge_chain(chain->hops[0].src, _src_driven ? s : d,
                            chain->hops, bytes);   // assigned, all hops
        }
        inject_hop(*chain);
    }

    void inject_hop(ChainState& chain) {
        const Hop& hop = chain.hops[chain.next_hop];
        if (_ledger) {
            lg_hop_injected(hop.src, hop.dst, chain.bytes);
        }
        vector<uint32_t> path;
        path.push_back(hop.src);
        path.push_back(hop.dst);
        uint32_t wave_slot = chain.wave;
        if (!_epoch_offset.empty()) {
            wave_slot += _epoch_offset[chain.dst];
        }
        if (!_align_offset.empty()) {
            wave_slot += _align_offset[chain.dst];
        }
        uint32_t epoch;
        if (_pair_color > 0) {
            // O1: keep (wave, hop) strictly ordered, order pairs inside by
            // the (d - s) diagonal so equal-priority sends form a rotating
            // permutation across destinations.
            uint32_t s = chain.hops[0].src;
            uint32_t sub = ((chain.dst % _pair_color) + _pair_color
                            - (s % _pair_color)) % _pair_color;
            epoch = (wave_slot * 4 + chain.next_hop) * _pair_color + sub;
        } else {
            epoch = wave_slot * 4 + chain.next_hop;
        }
        bool final_hop = (chain.next_hop + 1 == chain.hops.size());
        _graph.nodes[hop.src]->add_scheduled_chunk(
            chain.packet_flow, hop.src, hop.dst, epoch, chain.id,
            chain.next_hop, chain.bytes, 8, chain.next_packet_id, path,
            final_hop /* count_delivery only at the app-level endpoint */);
        _hops_emitted++;
        _hop_bytes_emitted += chain.bytes;
    }

    // ---- x41 ledger internals ---------------------------------------

    // Resources touched by hop (a, b); returns count (<= 4).
    uint32_t lg_res_of_hop(uint32_t a, uint32_t b, uint32_t* res) const {
        uint32_t n = 0;
        if (server_of(a) == server_of(b)) {
            res[n++] = a;                          // NV up of a
            res[n++] = _node_count + b;            // NV down of b
            return n;
        }
        res[n++] = 2 * _node_count + a;            // NIC up
        res[n++] = 3 * _node_count + b;            // NIC down
        uint32_t leaf_a, leaf_b;
        bool fab;
        if (_wiring == WIRING_POD) {
            leaf_a = server_of(a);
            leaf_b = server_of(b);
            fab = true;                            // every x-host hop
        } else {
            leaf_a = rail_of(a);
            leaf_b = rail_of(b);
            fab = leaf_a != leaf_b;                // only cross-rail
        }
        // x44: the per-(leaf,spine) entries assume the sim's
        // deterministic hash (topo/flex_pod_graph.h:290) -- real ECMP
        // hashing is invisible and a spine segment has no owner to
        // measure its discharge. spineagg keeps what IS owned: the
        // leaf/rail switch's aggregate up/down counters (INT/HPCC-style
        // channel-A telemetry), i.e. one entry per leaf per direction,
        // hash-free; the residual per-spine imbalance is covered by the
        // per-dst outer feedback loop. fabmask (ablation floor) drops
        // fabric entries entirely.
        if (fab && !_lg_fabmask) {
            if (_lg_spineagg) {
                res[n++] = _lg_fab0 + leaf_a * _lg_spines;
                res[n++] = _lg_fab0 + _lg_leaves * _lg_spines
                    + leaf_b * _lg_spines;
            } else {
                uint32_t sp = (a + b) % _lg_spines;
                res[n++] = _lg_fab0 + leaf_a * _lg_spines + sp;
                res[n++] = _lg_fab0 + _lg_leaves * _lg_spines
                    + leaf_b * _lg_spines + sp;
            }
        }
        return n;
    }

    // Does hop (a, b) cross the switching fabric (spine layer)?
    bool lg_hop_fab(uint32_t a, uint32_t b) const {
        if (server_of(a) == server_of(b)) {
            return false;
        }
        if (_wiring == WIRING_POD) {
            return true;
        }
        return rail_of(a) != rail_of(b);
    }

    // x42 synchronization models. The ledger splits into CHARGES
    // (assignment declarations: with deterministic replay every rank can
    // compute them from shared inputs, so they are visible live in every
    // mode except NONE) and DISCHARGES + effective rates (physical
    // observations at the resource owner: they travel by telemetry, so
    // allocators see them through a mode-dependent view):
    //   ORACLE  live global state, zero cost -- NOT a deployable claim,
    //           kept only as the upper-bound reference;
    //   STRONG  blocking allgather per wave: emission delayed sync_us,
    //           uses trigger-time discharge/rate state;
    //   SNAP    SSP-style bounded staleness: a periodic snapshot every
    //           sync_us, emissions immediate (allocators read charges
    //           live + discharges up to sync_us old);
    //   NONE    no exchange at all: each SENDER sees only its own
    //           chains' charges/discharges and nominal rates.
    enum {LG_ORACLE = 0, LG_STRONG = 1, LG_SNAP = 2, LG_NONE = 3};

public:
    void set_ledger_sync(const string& mode, double sync_us) {
        if (!_ledger) {
            return;
        }
        _lg_sync_ps = (simtime_picosec)(sync_us * 1e6);
        if (mode == "strong") {
            _lg_sync = LG_STRONG;
        } else if (mode == "snap") {
            _lg_sync = LG_SNAP;
            _lg_next_snap = EventList::now() + _lg_sync_ps;
            EventList::sourceIsPendingRel(*this, _lg_sync_ps);
        } else if (mode == "none") {
            _lg_sync = LG_NONE;
            _lg_vchg.assign(_node_count,
                            vector<double>(_lg_nres, 0.0));
            _lg_vdis.assign(_node_count,
                            vector<double>(_lg_nres, 0.0));
        } else {
            _lg_sync = LG_ORACLE;
        }
        _lg_dis_snap.assign(_lg_nres, 0.0);
        _lg_rate_snap = _lg_nom;
        cout << "mosaic_ledger_sync " << mode
             << " sync_us " << sync_us << endl;
    }

    // x44 protocol: drop the deterministic-replay assumption. Charges
    // travel on their own channel -- each dst PUBLISHES its wave's
    // corridor allocation at its 75% trigger (channel B), receivers keep
    // only the latest publication per dst; discharges/rates keep the
    // periodic telemetry snapshot (channel A). blag_us models the
    // broadcast network latency; blagw adds a lag in units of the
    // publisher's own inter-publish interval (0.5 / 1 / 2 waves).
    // spineagg drops the per-(leaf,spine) entries -- invisible to real
    // ECMP hashing and ownerless -- and controls the fabric share by a
    // per-dst outer feedback loop instead (two-level allocation:
    // bookkeep what has an owner, learn what does not). fabmask is the
    // ablation floor: fabric entries dropped, no outer loop.
    void set_ledger_proto(bool bcast, double blag_us, double blagw,
                          bool spineagg, bool fabmask, bool forkprobe,
                          double stagger_us = 0.0) {
        if (!_ledger) {
            return;
        }
        _lg_bcast = bcast;
        _lg_blag_ps = (simtime_picosec)(blag_us * 1e6);
        _lg_blagw = blagw;
        _lg_stagger_ps = (simtime_picosec)(stagger_us * 1e6);
        _lg_spineagg = spineagg;
        _lg_fabmask = fabmask;
        _lg_forkprobe = forkprobe;
        if (spineagg) {
            // leaf-aggregate entries carry the whole uplink capacity
            for (uint32_t l = 0; l < 2 * _lg_leaves; l++) {
                uint32_t r = _lg_fab0 + l * _lg_spines;
                _lg_nom[r] *= (double)_lg_spines;
                _lg_rate[r] = _lg_nom[r];
            }
            _lg_rate_snap = _lg_nom;
        }
        if (bcast) {
            _lg_chg_by.assign(_node_count,
                              vector<double>(_lg_nres, 0.0));
            _lg_pub.assign(_node_count, vector<double>(_lg_nres, 0.0));
            _lg_pubsum.assign(_lg_nres, 0.0);
            _lg_prevpub.assign(_node_count, 0);
        }
        if (spineagg) {
            _lg_phi.assign(_node_count, -1.0);
        }
        if (forkprobe) {
            _lg_dis_snap_prev.assign(_lg_nres, 0.0);
        }
        cout << "mosaic_lg_proto bcast " << (bcast ? 1 : 0)
             << " blag_us " << blag_us << " blagw " << blagw
             << " spineagg " << (spineagg ? 1 : 0)
             << " fabmask " << (_lg_fabmask ? 1 : 0) << endl;
    }

private:
    void lg_publish(uint32_t d) {
        simtime_picosec now = EventList::now();
        simtime_picosec delay = _lg_blag_ps;
        if (_lg_blagw > 0.0 && _lg_prevpub[d] > 0) {
            delay += (simtime_picosec)(_lg_blagw
                                       * (double)(now - _lg_prevpub[d]));
        }
        _lg_prevpub[d] = now;
        _lg_pub_cnt++;
        if (delay == 0) {
            vector<double>& by = _lg_chg_by[d];
            vector<double>& pub = _lg_pub[d];
            for (uint32_t r = 0; r < _lg_nres; r++) {
                _lg_pubsum[r] += by[r] - pub[r];
                pub[r] = by[r];
            }
            return;
        }
        LgPub e;
        e.due = now + delay;
        e.d = d;
        e.chg = _lg_chg_by[d];
        _lg_pubq.push_back(e);
        EventList::sourceIsPending(*this, e.due);
    }

public:
    void doNextEvent() override {
        simtime_picosec now = EventList::now();
        uint32_t p = 0;                  // channel B publications land
        while (p < _lg_pubq.size()) {    // (in every sync mode)
            if (_lg_pubq[p].due <= now) {
                LgPub e = _lg_pubq[p];
                _lg_pubq.erase(_lg_pubq.begin() + p);
                vector<double>& pub = _lg_pub[e.d];
                for (uint32_t r = 0; r < _lg_nres; r++) {
                    _lg_pubsum[r] += e.chg[r] - pub[r];
                    pub[r] = e.chg[r];
                }
            } else {
                p++;
            }
        }
        // Delayed wave emissions: STRONG entries carry their allgather's
        // trigger-time snapshot; x44 stagger entries (empty dis) decide
        // on the LIVE views at their slot time -- that is their point.
        size_t i = 0;
        while (i < _lg_emitq.size()) {
            if (_lg_emitq[i].due <= now) {
                LgEmit e = _lg_emitq[i];
                _lg_emitq.erase(_lg_emitq.begin() + i);
                if (!e.dis.empty()) {
                    _lg_dis_snap = e.dis;
                    _lg_rate_snap = e.rate;
                }
                if (_src_driven) {
                    emit_src_wave_body(e.d);   // e.d carries the src id
                } else {
                    emit_wave_body(e.d);
                }
                i = 0;                   // queue may have changed
            } else {
                i++;
            }
        }
        if (_lg_sync == LG_SNAP) {
            // A publication/stagger wakeup between ticks must NOT flip
            // the snapshot early (that would quietly shrink the
            // declared staleness); only the periodic tick does.
            if (now >= _lg_next_snap) {
                if (_lg_forkprobe) {
                    _lg_dis_snap_prev = _lg_dis_snap;
                }
                _lg_dis_snap = _lg_dis;  // the periodic allgather lands
                _lg_rate_snap = _lg_rate;
                _lg_next_snap = now + _lg_sync_ps;
                EventList::sourceIsPendingRel(*this, _lg_sync_ps);
            }
        }
    }

private:
    void lg_charge_chain(uint32_t origin, uint32_t dst,
                         const vector<Hop>& hops, mem_b bytes) {
        uint32_t res[4];
        for (uint32_t h = 0; h < hops.size(); h++) {
            uint32_t n = lg_res_of_hop(hops[h].src, hops[h].dst, res);
            for (uint32_t i = 0; i < n; i++) {
                _lg_chg[res[i]] += (double)bytes;
                if (_lg_sync == LG_NONE) {
                    _lg_vchg[origin][res[i]] += (double)bytes;
                }
                if (_lg_bcast) {
                    // single writer: the dst whose wave allocated this
                    // chain owns the entry (channel B publishes it)
                    _lg_chg_by[dst][res[i]] += (double)bytes;
                }
            }
        }
    }

    void lg_hop_injected(uint32_t a, uint32_t b, mem_b bytes) {
        uint32_t res[4];
        uint32_t n = lg_res_of_hop(a, b, res);
        for (uint32_t i = 0; i < n; i++) {
            _lg_arr[res[i]] += bytes;
        }
    }

    // Hop completed: discharge assigned + arrived work, and update the
    // effective-rate estimate from busy-period delivery spacing (the
    // interval since the previous delivery on this resource was fully
    // busy iff work remained queued after that delivery -- arrivals only
    // add in between, so the resource never idled).
    void lg_hop_delivered(uint32_t origin, uint32_t a, uint32_t b,
                          mem_b bytes) {
        uint32_t res[4];
        uint32_t n = lg_res_of_hop(a, b, res);
        simtime_picosec now = EventList::now();
        for (uint32_t i = 0; i < n; i++) {
            uint32_t r = res[i];
            _lg_dis[r] += (double)bytes;
            if (_lg_sync == LG_NONE) {
                _lg_vdis[origin][r] += (double)bytes;
            }
            // Detector scope = NIC ports only: they are per-GPU edge
            // resources whose busy throughput tracks capacity, and the
            // exogenous threat model (unannounced degrade, neighbor
            // interference, 19-group) physically lives there. NV/fabric
            // segments are the heavily-multiplexed shared stages whose
            // busy throughput dips endogenously under backpressure --
            // trusting those dips re-creates the censoring spiral
            // (+15-18% at nv90 off-seed, measured twice).
            bool nic_r = r >= 2 * _node_count && r < 4 * _node_count;
            if (_lg_correct && nic_r && _lg_prev[r] > 0
                && now > _lg_prev[r] && _lg_busyafter[r]) {
                // Interleaved packet service means one chunk's delivery
                // spacing is NOT its service time; aggregate bytes over
                // busy time and fold periodically instead (a ratio-of-
                // sums, not an EWMA of ratios).
                _lg_bacc[r] += (double)bytes;
                _lg_tacc[r] += (double)(now - _lg_prev[r]);
                if (_lg_tacc[r] >= 5e8) {            // 500 us busy
                    double inst = _lg_bacc[r] * 8000.0 / _lg_tacc[r];
                    if (inst > _lg_nom[r]) {
                        inst = _lg_nom[r];
                    }
                    // Detector, not estimator: congestion coupling makes
                    // busy throughput dip below capacity transiently
                    // (backpressure stalls with work queued); feeding
                    // that back is the censoring spiral in a new coat
                    // (+15-18% at nv90 off-seed, measured). Only a
                    // PERSISTENT large deficit -- the exogenous
                    // signature (degraded NIC, interference) -- switches
                    // the rate off nominal; recovery is immediate.
                    if (inst < 0.45 * _lg_nom[r]) {
                        if (_lg_streak[r] < 250) {
                            _lg_streak[r]++;
                        }
                        // Quorum gate: the exogenous threat model is an
                        // ISOLATED anomaly (one GPU's port pair);
                        // correlated deficits across the class are the
                        // congestion regime -- freeze rather than lock,
                        // or the spiral returns (seed44@90, measured).
                        if (_lg_streak[r] >= 3
                            && (_lg_rate[r] < _lg_nom[r]
                                || _lg_locked < 2)) {
                            if (_lg_rate[r] >= _lg_nom[r]) {
                                _lg_locked++;
                            }
                            double est = _lg_rate[r] < _lg_nom[r]
                                ? 0.5 * _lg_rate[r] + 0.5 * inst : inst;
                            _lg_rate[r] = est < _lg_nom[r] / 16.0
                                ? _lg_nom[r] / 16.0 : est;
                        }
                    } else {
                        _lg_streak[r] = 0;
                        if (_lg_rate[r] < _lg_nom[r] && _lg_locked > 0) {
                            _lg_locked--;
                        }
                        _lg_rate[r] = _lg_nom[r];
                    }
                    _lg_bacc[r] = 0.0;
                    _lg_tacc[r] = 0.0;
                }
            }
            _lg_arr[r] -= bytes;
            if (_lg_arr[r] < 0) {
                _lg_arr[r] = 0;
            }
            _lg_busyafter[r] = _lg_arr[r] > 0 ? 1 : 0;
            _lg_prev[r] = now;
        }
    }

    // Waterfill a pair's quota across its usable corridors: repeatedly
    // give the next slice to the corridor whose resulting bottleneck
    // projected-finish is smallest (ties -> fewer hops, i.e. DIRECT
    // first: the alpha argument). The slices are pending-tracked so one
    // pair's own allocation is self-consistent; the REAL charge happens
    // in emit_chain_hops when the chains are emitted right after.
    // Allocator's view of a resource's outstanding bytes and rate,
    // by synchronization mode (origin = the allocating sender).
    double lg_view_out(uint32_t origin, uint32_t r) const {
        double v;
        const vector<double>& ds =
            (_lg_probe && !_lg_dis_snap_prev.empty()) ? _lg_dis_snap_prev
                                                      : _lg_dis_snap;
        if (_lg_bcast) {
            // channel B: everyone else's charges as of their latest
            // publication, own charges live (origin = allocating dst)
            v = _lg_pubsum[r] - _lg_pub[origin][r]
                + _lg_chg_by[origin][r]
                - (_lg_sync == LG_ORACLE ? _lg_dis[r] : ds[r]);
        } else if (_lg_sync == LG_NONE) {
            v = _lg_vchg[origin][r] - _lg_vdis[origin][r];
        } else if (_lg_sync == LG_ORACLE) {
            v = _lg_chg[r] - _lg_dis[r];
        } else {
            v = _lg_chg[r] - ds[r];
        }
        return v < 0.0 ? 0.0 : v;
    }
    double lg_view_rate(uint32_t r) const {
        if (_lg_sync == LG_NONE) {
            return _lg_nom[r];           // no telemetry, no detector
        }
        if (_lg_sync == LG_ORACLE) {
            return _lg_rate[r];
        }
        return _lg_rate_snap[r];
    }

    void lg_split(uint32_t s, uint32_t d, const bool* usable,
                  mem_b quota, mem_b* share) {
        uint32_t cn = 0;
        uint32_t cand[CORRIDORS];
        uint32_t cres[CORRIDORS][12];
        uint32_t cnres[CORRIDORS];
        uint32_t chops[CORRIDORS];
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            share[c] = 0;
            if (!usable[c]) {
                continue;
            }
            vector<Hop> hops = corridor_hops(s, d, c);
            uint32_t nr = 0;
            uint32_t res[4];
            for (uint32_t h = 0; h < hops.size(); h++) {
                uint32_t k = lg_res_of_hop(hops[h].src, hops[h].dst, res);
                for (uint32_t i = 0; i < k && nr < 12; i++) {
                    cres[cn][nr++] = res[i];
                }
            }
            cand[cn] = c;
            cnres[cn] = nr;
            chops[cn] = (uint32_t)hops.size();
            cn++;
        }
        if (cn == 0) {
            share[DIRECT] = quota;
            return;
        }
        // pending additions per unique resource id (small linear map)
        uint32_t uid[CORRIDORS * 12];
        double upend[CORRIDORS * 12];
        uint32_t un = 0;
        mem_b left = quota;
        mem_b step = ((quota / 24) / 1024) * 1024;
        if (step < 1024) {
            step = 1024;
        }
        while (left > 0) {
            mem_b delta = min(left, step);
            uint32_t best = 0;
            double best_lvl = 0.0;
            for (uint32_t j = 0; j < cn; j++) {
                double lvl = 0.0;
                for (uint32_t i = 0; i < cnres[j]; i++) {
                    uint32_t r = cres[j][i];
                    double pend = 0.0;
                    for (uint32_t u = 0; u < un; u++) {
                        if (uid[u] == r) {
                            pend = upend[u];
                            break;
                        }
                    }
                    // origin = the DECIDER whose own charges are read live
                // (dst-wave: d; src-driven dispatch: s); non-bcast modes
                // keep the sender-private view.
                double lv = (lg_view_out(_lg_bcast ? (_src_driven ? s : d)
                                                   : s, r) + pend
                                 + (double)delta) / lg_view_rate(r);
                    if (lv > lvl) {
                        lvl = lv;
                    }
                }
                // Lexicographic (bottleneck level, then hop count): a
                // relay byte consumes 2-3 resources' time, so within a
                // level band the shorter corridor wins -- min-max alone
                // is blind to total-work inflation (the alpha argument,
                // and the ledger form of the prior's in-pipe 2:1 bias).
                if (j == 0) {
                    best = 0;
                    best_lvl = lvl;
                } else if (lvl < best_lvl * 0.95
                           || (lvl < best_lvl * 1.05
                               && chops[j] < chops[best])) {
                    best = j;
                    if (lvl < best_lvl) {
                        best_lvl = lvl;
                    }
                }
            }
            share[cand[best]] += delta;
            left -= delta;
            for (uint32_t i = 0; i < cnres[best]; i++) {
                uint32_t r = cres[best][i];
                bool found = false;
                for (uint32_t u = 0; u < un; u++) {
                    if (uid[u] == r) {
                        upend[u] += (double)delta;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    uid[un] = r;
                    upend[un] = (double)delta;
                    un++;
                }
            }
        }
    }

    // Does corridor c of pair (s, d) cross the fabric anywhere?
    bool lg_corridor_fab(uint32_t s, uint32_t d, uint32_t c) {
        vector<Hop> hops = corridor_hops(s, d, c);
        for (uint32_t h = 0; h < hops.size(); h++) {
            if (lg_hop_fab(hops[h].src, hops[h].dst)) {
                return true;
            }
        }
        return false;
    }

    // x44 two-level allocation: the fabric share is set by the per-dst
    // OUTER feedback loop (phi, learned from receiver-measured rates --
    // the spine is unobservable to bookkeeping), the INNER waterfill
    // uses the precise port ledger within each side. Wave 0 runs
    // unconstrained (port ledger only) and phi initializes from its
    // measured outcome -- no new constants.
    void lg_split_2level(uint32_t s, uint32_t d, const bool* usable,
                         mem_b quota, mem_b* share) {
        bool ufab[CORRIDORS], unf[CORRIDORS];
        uint32_t nfab = 0, nnf = 0;
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            ufab[c] = unf[c] = false;
            if (!usable[c]) {
                continue;
            }
            if (lg_corridor_fab(s, d, c)) {
                ufab[c] = true;
                nfab++;
            } else {
                unf[c] = true;
                nnf++;
            }
        }
        double phi = _lg_phi[d];
        if (nfab == 0 || nnf == 0 || phi < 0.0) {
            lg_split(s, d, usable, quota, share);
            return;
        }
        mem_b qf = ((mem_b)(phi * (double)quota) / 1024) * 1024;
        if (qf > quota) {
            qf = quota;
        }
        mem_b qn = quota - qf;
        mem_b shf[CORRIDORS] = {0, 0, 0, 0};
        mem_b shn[CORRIDORS] = {0, 0, 0, 0};
        if (qf > 0) {
            lg_split(s, d, ufab, qf, shf);
        }
        if (qn > 0) {
            lg_split(s, d, unf, qn, shn);
        }
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            share[c] = shf[c] + shn[c];
        }
    }

    // Outer loop: the EPS lambda law verbatim on the 2-way {fabric,
    // non-fabric} split, from receiver-local per-wave rates.
    void update_phi(uint32_t d, const WaveStats& wave) {
        double rf = 0.0, rn = 0.0;
        if (wave.fab_bytes > 0 && wave.fab_last > wave.start) {
            rf = (double)wave.fab_bytes
                / ((double)(wave.fab_last - wave.start) / 1e6);
        }
        if (wave.nofab_bytes > 0 && wave.nofab_last > wave.start) {
            rn = (double)wave.nofab_bytes
                / ((double)(wave.nofab_last - wave.start) / 1e6);
        }
        double tot = rf + rn;
        if (tot <= 0.0) {
            return;
        }
        double share;
        if (rf > 0.0) {
            share = rf / tot;
        } else if (wave.fab_emitted > 0) {
            share = 0.0;                 // fed, delivered nothing
        } else if (_lg_phi[d] >= 0.0) {
            share = _lg_phi[d];          // starved: keep belief
        } else {
            return;
        }
        if (_lg_phi[d] < 0.0) {          // wave-0 outcome initializes
            _lg_phi[d] = min(max(share, (double)_floor),
                             1.0 - (double)_floor);
            return;
        }
        double upd = (1.0 - _gamma) * _lg_phi[d] + _gamma * share;
        upd = min(upd, _lg_phi[d] * 1.5 + 0.01);   // trust region
        _lg_phi[d] = min(max(upd, (double)_floor), 1.0 - (double)_floor);
    }

    // Emit one wave of traffic toward destination GPU d. Under STRONG
    // sync the trigger only LAUNCHES the allgather: the emission itself
    // lands sync_us later carrying trigger-time discharge/rate state
    // (doNextEvent fires it). All other modes emit inline.
    void emit_wave(uint32_t d) {
        // x44 decision staggering: a wave front triggers many dsts
        // within nanoseconds, and any real broadcast latency blinds the
        // whole cohort to each other's policies (measured +6-12% at
        // nv90). Waiting one's slot (d mod 8 spaced stagger_us apart)
        // before deciding lets later slots read the earlier slots'
        // publications -- a real, implementable protocol (slotting),
        // costing at most 7*stagger_us of wave latency.
        if (_ledger && _lg_bcast && _lg_stagger_ps > 0) {
            LgEmit e;
            e.due = EventList::now() + (simtime_picosec)(d % 8)
                * _lg_stagger_ps;
            e.d = d;
            if (e.due == EventList::now()) {
                emit_wave_body(d);
                return;
            }
            _lg_emitq.push_back(e);
            EventList::sourceIsPending(*this, e.due);
            return;
        }
        if (_ledger && _lg_sync == LG_STRONG) {
            LgEmit e;
            e.due = EventList::now() + _lg_sync_ps;
            e.d = d;
            e.dis = _lg_dis;
            e.rate = _lg_rate;
            _lg_emitq.push_back(e);
            EventList::sourceIsPending(*this, e.due);
            return;
        }
        emit_wave_body(d);
    }

    void emit_wave_body(uint32_t d) {
        DstState& state = _dst_state[d];
        uint32_t wave_limit = dst_wave_limit(d);
        if (state.emit_wave >= wave_limit) {
            return;
        }
        // F6 pacing: a cold dst may run at most (k-1) waves ahead of the
        // predicted straggler's wave count; the withheld wave is released
        // by release_paced() when the straggler advances or finishes.
        if (_cold_pace > 0 && !_pace_is_cold.empty() && _pace_is_cold[d]
            && !_dst_done[_pace_hot]
            && state.emit_wave
               > _dst_state[_pace_hot].emit_wave + _cold_pace - 1) {
            state.paced = true;
            return;
        }
        state.paced = false;
        uint32_t w = state.emit_wave;
        state.emit_wave++;
        state.next_triggered = false;
        state.stats.resize(state.emit_wave);
        state.stats[w].start = EventList::now();

        uint32_t waves_left = wave_limit - w;
        bool last_wave = (waves_left <= 1);
        bool emitted = false;

        // Stale backlog: bytes emitted into a corridor two or more waves
        // ago and still undelivered. One wave in flight is healthy
        // pipelining; older residue means the corridor was over-allocated,
        // so its share of THIS wave is docked by it (queue-level control,
        // not just rate-level).
        double eff_lambda[CORRIDORS];
        uint64_t wave_total = 0;
        for (uint32_t s = 0; s < _node_count; s++) {
            if (s != d && _remaining[s][d] > 0) {
                mem_b rem = _remaining[s][d];
                wave_total += last_wave
                    ? rem : (rem + waves_left - 1) / waves_left;
            }
        }
        double eff_sum = 0.0;
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            mem_b stale = 0;
            for (uint32_t v = 0; v + 1 < w; v++) {
                stale += state.stats[v].corridor_emitted[c]
                    - state.stats[v].corridor_bytes[c];
            }
            eff_lambda[c] = state.lambda[c];
            if (wave_total > 0 && stale > 0) {
                eff_lambda[c] = max(_floor,
                                    state.lambda[c]
                                    - (double)stale / (double)wave_total);
            }
            eff_sum += eff_lambda[c];
        }
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            eff_lambda[c] /= eff_sum;
        }
        // F7: per-source-host effective lambda (same stale-backlog dock,
        // applied to each host's slice; the slice's share of wave_total is
        // approximated as 1/_servers).
        vector<vector<double> > host_eff;
        if (_host_lambda) {
            if (state.hlambda.empty()) {
                state.hlambda.assign(_servers,
                                     vector<double>(CORRIDORS, 0.0));
                for (uint32_t h = 0; h < _servers; h++) {
                    for (uint32_t c = 0; c < CORRIDORS; c++) {
                        state.hlambda[h][c] = state.lambda[c];
                    }
                }
            }
            state.stats[w].hemitted.assign(_servers,
                                           vector<mem_b>(CORRIDORS, 0));
            state.stats[w].hbytes.assign(_servers,
                                         vector<mem_b>(CORRIDORS, 0));
            state.stats[w].hlast.assign(
                _servers, vector<simtime_picosec>(CORRIDORS, 0));
            host_eff.assign(_servers, vector<double>(CORRIDORS, 0.0));
            for (uint32_t h = 0; h < _servers; h++) {
                double sum = 0.0;
                for (uint32_t c = 0; c < CORRIDORS; c++) {
                    mem_b stale = 0;
                    for (uint32_t v = 0; v + 1 < w; v++) {
                        if (state.stats[v].hemitted.empty()) {
                            continue;
                        }
                        stale += state.stats[v].hemitted[h][c]
                            - state.stats[v].hbytes[h][c];
                    }
                    host_eff[h][c] = state.hlambda[h][c];
                    if (wave_total > 0 && stale > 0) {
                        host_eff[h][c] =
                            max(_floor,
                                state.hlambda[h][c]
                                - (double)stale * _servers
                                  / (double)wave_total);
                    }
                    sum += host_eff[h][c];
                }
                for (uint32_t c = 0; c < CORRIDORS; c++) {
                    host_eff[h][c] /= sum;
                }
            }
        }
        uint32_t cap = ramp_cap(d, w);

        for (uint32_t s = 0; s < _node_count; s++) {
            mem_b rem = (s == d) ? 0 : _remaining[s][d];
            if (rem <= 0) {
                continue;
            }
            mem_b quota = last_wave
                ? rem : (rem + waves_left - 1) / waves_left;
            quota = min(quota, rem);
            _remaining[s][d] -= quota;
            emitted = true;

            if (server_of(s) == server_of(d)) {
                if (_dtrace) {
                    fprintf(_dtrace,
                            "W %u D %u S %u Q %lld SH %lld 0 0 0 0 0\n",
                            w, d, s, (long long)quota, (long long)quota);
                }
                emit_chain(s, d, DIRECT, false, quota, w);
                continue;
            }

            bool usable[CORRIDORS];
            pair_corridors(s, d, usable);
            if (cap < _corridor_count) {
                for (uint32_t c = 0; c < CORRIDORS; c++) {
                    if (!rail_nested(c, cap)) {
                        usable[c] = false;
                    }
                }
            }
            const double* lam = _host_lambda
                ? &host_eff[server_of(s)][0] : eff_lambda;
            double mass = 0.0;
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (usable[c]) {
                    mass += lam[c];
                }
            }

            mem_b share[CORRIDORS] = {0, 0, 0, 0};
            if (_ledger && _lg_spineagg) {
                lg_split_2level(s, d, usable, quota, share);
            } else if (_ledger) {
                lg_split(s, d, usable, quota, share);
                if (_lg_forkprobe && _lg_sync == LG_SNAP
                    && !_lg_dis_snap_prev.empty()) {
                    // Replay-fork sensitivity: a rank whose telemetry is
                    // one snap period behind recomputes this split with
                    // the previous snapshot; differing bytes measure how
                    // fast replicated-replay charges would diverge.
                    mem_b sh2[CORRIDORS] = {0, 0, 0, 0};
                    _lg_probe = true;
                    lg_split(s, d, usable, quota, sh2);
                    _lg_probe = false;
                    mem_b diff = 0;
                    for (uint32_t c = 0; c < CORRIDORS; c++) {
                        diff += share[c] > sh2[c] ? share[c] - sh2[c]
                                                  : sh2[c] - share[c];
                    }
                    _lg_fork_bytes += (double)diff / 2.0;
                    _lg_fork_total += (double)quota;
                }
            } else {
                mem_b assigned = 0;
                uint32_t heavy = DIRECT;
                double heavy_lambda = -1.0;
                for (uint32_t c = 0; c < CORRIDORS; c++) {
                    if (!usable[c]) {
                        continue;
                    }
                    if (mass <= 0.0) {   // e.g. forced fallback corridor
                        heavy = c;       // with lambda 0: whole quota
                        break;
                    }
                    share[c] = ((mem_b)(lam[c] / mass * quota)
                                / 1024) * 1024;
                    assigned += share[c];
                    if (lam[c] > heavy_lambda) {
                        heavy_lambda = lam[c];
                        heavy = c;
                    }
                }
                share[heavy] += quota - assigned;
            }
            if (_dtrace) {
                fprintf(_dtrace,
                        "W %u D %u S %u Q %lld SH %lld %lld %lld %lld %lld"
                        " %lld\n",
                        w, d, s, (long long)quota, (long long)share[0],
                        (long long)share[1], (long long)share[2],
                        (long long)share[3], (long long)share[4],
                        (long long)share[5]);
            }
            // LP-v2: collapse the spread lanes into one class routed by
            // the jdist marginal (all transit rails, LP weights).
            mem_b spread_total = 0;
            if (!_lp_jdist.empty()
                && (_wiring == WIRING_CUBE || _wiring == WIRING_RAIL)) {
                for (uint32_t c = SPREAD3; c < CORRIDORS; c++) {
                    spread_total += share[c];
                    share[c] = 0;
                }
            }
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (share[c] <= 0) {
                    continue;
                }
                if (_lg_bcast) {
                    _lg_pub_entries++;   // one (src, corridor) policy row
                }
                if (_ledger) {
                    // A chain is the store-and-forward unit: the lambda
                    // split's per-corridor slivers gave relays a fine
                    // pipeline for free; a whole-quota ledger chain
                    // would store-forward megabytes per hop. Slice to
                    // keep the relay pipeline granularity.
                    mem_b left2 = share[c];
                    while (left2 > 0) {
                        mem_b piece = min(left2, _lg_chain_cap);
                        emit_chain(s, d, c, true, piece, w);
                        left2 -= piece;
                    }
                } else {
                    emit_chain(s, d, c, true, share[c], w);
                }
            }
            if (spread_total > 0) {
                emit_spread_jdist(s, d, spread_total, w);
            }
        }

        if (_ledger && _lg_bcast && emitted) {
            // channel B: publish this wave's allocation (the charges it
            // implies) -- receivers hold each dst's latest publication
            lg_publish(d);
        }
        if (!emitted && state.emit_wave < wave_limit) {
            // Nothing left for this GPU; fast-forward remaining waves.
            state.emit_wave = wave_limit;
        }
        if (_cold_pace > 0 && d == _pace_hot) {
            release_paced();
        }
    }

    // src-driven wave emission (task 2). Same stagger discipline as
    // emit_wave (slot = s mod 8): n srcs trigger within nanoseconds of each
    // other and must get a chance to read earlier slots' channel-B
    // publications before deciding.
    void emit_src_wave(uint32_t s) {
        if (_ledger && _lg_bcast && _lg_stagger_ps > 0) {
            LgEmit e;
            e.due = EventList::now() + (simtime_picosec)(s % 8)
                * _lg_stagger_ps;
            e.d = s;
            if (e.due == EventList::now()) {
                emit_src_wave_body(s);
                return;
            }
            _lg_emitq.push_back(e);
            EventList::sourceIsPending(*this, e.due);
            return;
        }
        emit_src_wave_body(s);
    }

    void emit_src_wave_body(uint32_t s) {
        SrcState& state = _src_state[s];
        if (state.emit_wave >= _waves) {
            return;
        }
        uint32_t w = state.emit_wave;
        state.emit_wave++;
        state.next_triggered = false;
        state.stats.resize(state.emit_wave);

        uint32_t waves_left = _waves - w;
        bool last_wave = (waves_left <= 1);
        bool emitted = false;

        for (uint32_t d = 0; d < _node_count; d++) {
            mem_b rem = (d == s) ? 0 : _remaining[s][d];
            if (rem <= 0) {
                continue;
            }
            mem_b quota = last_wave
                ? rem : (rem + waves_left - 1) / waves_left;
            quota = min(quota, rem);
            _remaining[s][d] -= quota;
            emitted = true;

            if (server_of(s) == server_of(d)) {
                if (_dtrace) {
                    fprintf(_dtrace,
                            "W %u D %u S %u Q %lld SH %lld 0 0 0 0 0\n",
                            w, d, s, (long long)quota, (long long)quota);
                }
                emit_chain(s, d, DIRECT, false, quota, w);
                continue;
            }

            bool usable[CORRIDORS];
            pair_corridors(s, d, usable);
            mem_b share[CORRIDORS] = {0, 0, 0, 0};
            if (_lg_spineagg) {
                lg_split_2level(s, d, usable, quota, share);
            } else {
                lg_split(s, d, usable, quota, share);
            }
            if (_dtrace) {
                fprintf(_dtrace,
                        "W %u D %u S %u Q %lld SH %lld %lld %lld %lld %lld"
                        " %lld\n",
                        w, d, s, (long long)quota, (long long)share[0],
                        (long long)share[1], (long long)share[2],
                        (long long)share[3], (long long)share[4],
                        (long long)share[5]);
            }
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (share[c] <= 0) {
                    continue;
                }
                if (_lg_bcast) {
                    _lg_pub_entries++;
                }
                mem_b left2 = share[c];
                while (left2 > 0) {
                    mem_b piece = min(left2, _lg_chain_cap);
                    emit_chain(s, d, c, true, piece, w);
                    left2 -= piece;
                }
            }
        }

        if (_ledger && _lg_bcast && emitted) {
            lg_publish(s);   // channel B, attributed to the deciding src
        }
        if (!emitted && state.emit_wave < _waves) {
            state.emit_wave = _waves;
        }
        // A wave with no cross-server hop-0 bytes has no ACK events to
        // trigger on (rt: immediate advance) -- move on right away.
        if (emitted && state.emit_wave < _waves
            && state.stats[w].emitted_nic == 0) {
            state.next_triggered = true;
            emit_src_wave(s);
        }
    }

    // F6: hand out the waves withheld by pacing (the straggler advanced
    // or finished). Re-entrancy guarded because emit_wave() calls this at
    // its own tail.
    void release_paced() {
        if (_cold_pace == 0 || _releasing) {
            return;
        }
        _releasing = true;
        for (uint32_t d = 0; d < _node_count; d++) {
            if (!_dst_done[d] && _dst_state[d].paced) {
                _dst_state[d].paced = false;
                emit_wave(d);
            }
        }
        _releasing = false;
    }

    // LP-v2 spread emission: split the aggregated spread share across
    // ALL transit rails j by the jdist weights. Chains are tagged
    // SPREAD3 so wave feedback measures the class as one corridor.
    void emit_spread_jdist(uint32_t s, uint32_t d, mem_b bytes,
                           uint32_t wave) {
        uint32_t r = rail_of(s), rd = rail_of(d);
        uint32_t h = server_of(s), n = server_of(d);
        const vector<pair<double, uint32_t> >& jd = _lp_jdist[r][rd];
        if (jd.empty()) {
            emit_chain(s, d, SPREAD3, true, bytes, wave);
            return;
        }
        mem_b assigned = 0;
        for (uint32_t k = 0; k < jd.size(); k++) {
            mem_b part = (k + 1 == jd.size())
                ? bytes - assigned
                : ((mem_b)(jd[k].first * bytes) / 1024) * 1024;
            part = min(part, bytes - assigned);
            assigned += part;
            if (part <= 0) {
                continue;
            }
            uint32_t j = jd[k].second;
            vector<Hop> hops;
            hops.push_back(Hop{s, h * _gpus + j});
            hops.push_back(Hop{h * _gpus + j, n * _gpus + j});
            hops.push_back(Hop{n * _gpus + j, d});
            emit_chain_hops(s, d, SPREAD3, true, part, wave, hops);
        }
    }

    // Feedback: move lambda toward each corridor's measured effective
    // bandwidth share in the finished (thresholded) wave.
    void update_lambda(uint32_t d, const WaveStats& wave) {
        if (!_adapt) {
            return;
        }
        DstState& state = _dst_state[d];
        if (_host_lambda && !wave.hbytes.empty() && !state.hlambda.empty()) {
            update_host_lambda(state, wave);
        }
        double rate[CORRIDORS];
        double total_rate = 0.0;
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            if (wave.corridor_bytes[c] > 0
                && wave.corridor_last[c] > wave.start) {
                double us = (double)(wave.corridor_last[c] - wave.start)
                    / 1e6;
                rate[c] = (double)wave.corridor_bytes[c] / us;
            } else {
                rate[c] = 0.0;
            }
            total_rate += rate[c];
        }
        if (total_rate <= 0.0) {
            return;
        }

        if (_guard_eps > 0.0) {
            bool congested = false;   // fed but delivered nothing
            double worst = 0.0;
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (!corridor_enabled(c)) {
                    continue;
                }
                if (rate[c] > 0.0) {
                    worst = max(worst, state.lambda[c] * total_rate
                                       / rate[c]);
                } else if (wave.corridor_emitted[c] > 0) {
                    congested = true;
                }
            }
            if (!congested && worst > 0.0
                && 1.0 - 1.0 / worst < _guard_eps) {
                return;   // prior already near-waterfilling: freeze
            }
        }

        double sum = 0.0;
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            if (!corridor_enabled(c)) {
                state.lambda[c] = 0.0;
                continue;
            }
            double share;
            if (rate[c] > 0.0) {
                share = rate[c] / total_rate;
            } else if (wave.corridor_emitted[c] > 0) {
                // Fed bytes but delivered nothing by the threshold: the
                // strongest possible evidence of congestion. Treating it
                // as "no data, keep belief" is censoring bias -- slow
                // corridors are unmeasured precisely because they are
                // slow -- and keeps feeding the bottleneck.
                // E5 refinement (flag-gated): a multi-hop corridor whose
                // relay hop DID advance is congested-but-alive (the tail
                // hop just started late); zeroing it starves the NV pipe
                // the straggler needs. Keep belief in that case.
                share = (_hop_progress_fix && wave.hop_progress[c])
                    ? state.lambda[c] : 0.0;
            } else {
                share = state.lambda[c];  // starved corridor: no evidence
            }
            double updated = (1.0 - _gamma) * state.lambda[c]
                + _gamma * share;
            // Trust region: at most 1.5x growth per wave, so a noisy
            // single-wave estimate cannot slam a corridor from floor to
            // dominance and re-congest it (oscillation damping).
            updated = min(updated, state.lambda[c] * 1.5 + 0.01);
            state.lambda[c] = max(updated, _floor);
            sum += state.lambda[c];
        }
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            state.lambda[c] /= sum;
        }
    }

    // F7: the same update rule as update_lambda(), applied independently
    // to each source host's slice of the wave. A class-asymmetric fault
    // (one relay NIC of three) shows up in exactly one slice, so only that
    // host's sources move off the corridor; symmetric slices keep moving
    // together and reproduce the per-dst behavior.
    void update_host_lambda(DstState& state, const WaveStats& wave) {
        for (uint32_t h = 0; h < _servers; h++) {
            double rate[CORRIDORS];
            double total_rate = 0.0;
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (wave.hbytes[h][c] > 0 && wave.hlast[h][c] > wave.start) {
                    double us = (double)(wave.hlast[h][c] - wave.start)
                        / 1e6;
                    rate[c] = (double)wave.hbytes[h][c] / us;
                } else {
                    rate[c] = 0.0;
                }
                total_rate += rate[c];
            }
            if (total_rate <= 0.0) {
                continue;
            }
            double sum = 0.0;
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                if (!corridor_enabled(c)) {
                    state.hlambda[h][c] = 0.0;
                    continue;
                }
                double share;
                if (rate[c] > 0.0) {
                    share = rate[c] / total_rate;
                } else if (wave.hemitted[h][c] > 0) {
                    // Same censoring-bias stance as the per-dst rule.
                    share = (_hop_progress_fix && wave.hop_progress[c])
                        ? state.hlambda[h][c] : 0.0;
                } else {
                    share = state.hlambda[h][c];
                }
                double updated = (1.0 - _gamma) * state.hlambda[h][c]
                    + _gamma * share;
                updated = min(updated, state.hlambda[h][c] * 1.5 + 0.01);
                state.hlambda[h][c] = max(updated, _floor);
                sum += state.hlambda[h][c];
            }
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                state.hlambda[h][c] /= sum;
            }
        }
    }

    // F7: event-driven re-anchors (E2 rewaterfill / E4 phase plan) rewrite
    // the per-dst lambda underneath the host slices. Re-align each slice
    // multiplicatively (slice keeps its learned deviation RATIO from the
    // old per-dst value, applied to the new one) instead of resetting, so
    // a discovered per-host fault survives the re-anchor.
    void sync_host_lambda(DstState& state, const double* old_lam) {
        if (!_host_lambda || state.hlambda.empty()) {
            return;
        }
        for (uint32_t h = 0; h < _servers; h++) {
            double sum = 0.0;
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                double ratio = old_lam[c] > 1e-9
                    ? state.hlambda[h][c] / old_lam[c] : 1.0;
                state.hlambda[h][c] =
                    max(_floor, ratio * state.lambda[c]);
                if (!corridor_enabled(c)) {
                    state.hlambda[h][c] = 0.0;
                }
                sum += state.hlambda[h][c];
            }
            if (sum <= 0.0) {
                continue;
            }
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                state.hlambda[h][c] /= sum;
            }
        }
    }

    // E4 fluid model. Links 0..N-1 are per-GPU NIC-ingress lanes, links
    // N..2N-1 are per-dst NV-ingress pipes. A same-host pair rides
    // {NV(d)}; a cross pair has one direct subflow {NIC(d)} plus one
    // relay subflow {NIC(j), NV(d)} per host peer j of d. Max-min rates
    // via standard progressive filling; pairs drain as fluid; each dst
    // completion opens a new phase whose per-dst nv_share is snapshotted.
    struct PlanFlow {
        uint32_t pair;
        int link1;   // -1 = none
        int link2;
        bool nv;     // counts toward the dst's NV-pipe share
        double rate;
    };

    void build_phase_plan() {
        if (_wiring != WIRING_RAIL || _nv_gbps <= 0.0
            || _nic_gbps <= 0.0) {
            _phase_plan = false;
            return;
        }
        uint32_t n = _node_count;
        vector<uint32_t> ps, pd;
        vector<double> prem;
        vector<uint32_t> dst_pairs(n, 0);
        for (uint32_t s = 0; s < n; s++) {
            for (uint32_t d = 0; d < n; d++) {
                if (s != d && _demand[s][d] > 0) {
                    ps.push_back(s);
                    pd.push_back(d);
                    prem.push_back((double)_demand[s][d]);
                    dst_pairs[d]++;
                }
            }
        }
        uint32_t np = (uint32_t)ps.size();
        vector<PlanFlow> flows;
        vector<vector<uint32_t> > pair_flows(np);
        for (uint32_t p = 0; p < np; p++) {
            uint32_t s = ps[p], d = pd[p];
            if (server_of(s) == server_of(d)) {
                pair_flows[p].push_back((uint32_t)flows.size());
                flows.push_back(PlanFlow{p, (int)(n + d), -1, true, 0.0});
                continue;
            }
            pair_flows[p].push_back((uint32_t)flows.size());
            flows.push_back(PlanFlow{p, (int)d, -1, false, 0.0});
            uint32_t host = server_of(d);
            for (uint32_t j = host * _gpus; j < (host + 1) * _gpus; j++) {
                if (j == d) {
                    continue;
                }
                pair_flows[p].push_back((uint32_t)flows.size());
                flows.push_back(PlanFlow{p, (int)j, (int)(n + d),
                                         true, 0.0});
            }
        }
        vector<char> pair_done(np, 0);
        _plan_nv_share.assign(n, vector<double>(n, -1.0));
        _plan_order.clear();
        uint32_t phase = 0;
        bool snapshot_due = true;
        uint32_t pairs_left = np;
        int guard = (int)np + (int)n + 8;
        while (pairs_left > 0 && guard-- > 0) {
            // Max-min rates for active subflows (progressive filling).
            vector<double> cap(2 * n, 0.0);
            vector<uint32_t> cnt(2 * n, 0);
            for (uint32_t d = 0; d < n; d++) {
                cap[d] = _nic_gbps;
                cap[n + d] = _nv_gbps;
            }
            vector<char> fixed(flows.size(), 0);
            uint32_t unfixed = 0;
            for (uint32_t f = 0; f < flows.size(); f++) {
                flows[f].rate = 0.0;
                if (pair_done[flows[f].pair]) {
                    fixed[f] = 1;
                    continue;
                }
                unfixed++;
                if (flows[f].link1 >= 0) {
                    cnt[flows[f].link1]++;
                }
                if (flows[f].link2 >= 0) {
                    cnt[flows[f].link2]++;
                }
            }
            while (unfixed > 0) {
                int bl = -1;
                double share = 0.0;
                for (uint32_t l = 0; l < 2 * n; l++) {
                    if (cnt[l] == 0) {
                        continue;
                    }
                    double sh = cap[l] / cnt[l];
                    if (bl < 0 || sh < share) {
                        bl = (int)l;
                        share = sh;
                    }
                }
                if (bl < 0) {
                    break;
                }
                for (uint32_t f = 0; f < flows.size(); f++) {
                    if (fixed[f]
                        || (flows[f].link1 != bl
                            && flows[f].link2 != bl)) {
                        continue;
                    }
                    flows[f].rate = share;
                    fixed[f] = 1;
                    unfixed--;
                    int other = flows[f].link1 == bl ? flows[f].link2
                                                     : flows[f].link1;
                    if (other >= 0) {
                        cap[other] -= share;
                        cnt[other]--;
                    }
                }
                cap[bl] = 0.0;
                cnt[bl] = 0;
            }
            if (snapshot_due && phase < n) {
                for (uint32_t d = 0; d < n; d++) {
                    if (dst_pairs[d] == 0) {
                        continue;
                    }
                    double nv_rate = 0.0, total = 0.0;
                    for (uint32_t f = 0; f < flows.size(); f++) {
                        if (pair_done[flows[f].pair]
                            || pd[flows[f].pair] != d
                            || server_of(ps[flows[f].pair])
                                   == server_of(d)) {
                            continue;  // mixer splits cross bytes only
                        }
                        total += flows[f].rate;
                        if (flows[f].nv) {
                            nv_rate += flows[f].rate;
                        }
                    }
                    if (total > 0.0) {
                        _plan_nv_share[phase][d] = nv_rate / total;
                    }
                }
                snapshot_due = false;
            }
            // Time to the next pair completion.
            double dt = -1.0;
            for (uint32_t p = 0; p < np; p++) {
                if (pair_done[p]) {
                    continue;
                }
                double rate = 0.0;
                for (uint32_t i = 0; i < pair_flows[p].size(); i++) {
                    rate += flows[pair_flows[p][i]].rate;
                }
                if (rate > 1e-12) {
                    double t = prem[p] / rate;
                    if (dt < 0.0 || t < dt) {
                        dt = t;
                    }
                }
            }
            if (dt < 0.0) {
                break;
            }
            for (uint32_t p = 0; p < np; p++) {
                if (pair_done[p]) {
                    continue;
                }
                double rate = 0.0;
                for (uint32_t i = 0; i < pair_flows[p].size(); i++) {
                    rate += flows[pair_flows[p][i]].rate;
                }
                prem[p] -= rate * dt;
                if (prem[p] <= 1e-6 * (double)_demand[ps[p]][pd[p]] + 1.0) {
                    pair_done[p] = 1;
                    pairs_left--;
                    if (--dst_pairs[pd[p]] == 0) {
                        _plan_order.push_back(pd[p]);
                        phase++;
                        snapshot_due = true;
                    }
                }
            }
        }
        cout << "mosaic_phase_plan phases " << phase << " order";
        for (uint32_t i = 0; i < _plan_order.size(); i++) {
            cout << " " << _plan_order[i];
        }
        cout << endl;
        apply_phase(0);
    }

    void apply_phase(uint32_t k) {
        if (k >= _plan_nv_share.size()) {
            return;
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            if (_dst_done[d] || _plan_nv_share[k][d] < 0.0) {
                continue;
            }
            double old_lam[CORRIDORS];
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                old_lam[c] = _dst_state[d].lambda[c];
            }
            rail_shares_to_lambda(_dst_state[d], _plan_nv_share[k][d]);
            sync_host_lambda(_dst_state[d], old_lam);
        }
    }

    // Completion-event dispatcher for the donation/plan mechanisms.
    // Lambda-table updates run first so an early wave (E1) is cut with
    // the post-donation split.
    void on_dst_complete() {
        if (_phase_plan && _done_count < _plan_nv_share.size()) {
            apply_phase(_done_count);
        }
        if (_donate_rewaterfill) {
            rewaterfill_residual();
        }
        if (_donate_window) {
            donate_window_slot();
        }
        if (_cold_pace > 0) {
            // The straggler may have just finished; never strand a paced
            // dst.
            release_paced();
        }
    }

    // E2: isolated two-pipe waterfilling on each unfinished dst's
    // un-emitted residual, applied directly (no trust region: this is an
    // event-triggered re-anchor, not a noisy single-wave measurement).
    void rewaterfill_residual() {
        if ((_wiring != WIRING_RAIL && _wiring != WIRING_CUBE)
            || _nv_gbps <= 0.0 || _nic_gbps <= 0.0) {
            return;
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            if (_dst_done[d]) {
                continue;
            }
            double cross = 0.0, same = 0.0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d || _remaining[s][d] <= 0) {
                    continue;
                }
                if (s / _gpus == d / _gpus) {
                    same += (double)_remaining[s][d];
                } else {
                    cross += (double)_remaining[s][d];
                }
            }
            if (cross <= 0.0) {
                continue;
            }
            double nv = nv_of(d);
            double x = (nv * cross - _nic_gbps * same)
                / (nv + _nic_gbps);
            x = max(0.0, min(x, cross));
            double old_lam[CORRIDORS];
            for (uint32_t c = 0; c < CORRIDORS; c++) {
                old_lam[c] = _dst_state[d].lambda[c];
            }
            if (_wiring == WIRING_CUBE) {
                cube_shares_to_lambda(_dst_state[d], d % _gpus,
                                      1.0 - x / cross, NULL);
            } else {
                rail_shares_to_lambda(_dst_state[d], x / cross);
            }
            sync_host_lambda(_dst_state[d], old_lam);
        }
    }

    // E1: grant the most-behind unfinished dst its next wave immediately.
    void donate_window_slot() {
        uint32_t straggler = _node_count;
        uint64_t worst = 0;
        for (uint32_t d = 0; d < _node_count; d++) {
            if (_dst_done[d]) {
                continue;
            }
            uint64_t und = _dst_expected[d] - _dst_delivered[d];
            if (und > worst) {
                worst = und;
                straggler = d;
            }
        }
        if (straggler < _node_count
            && _dst_state[straggler].emit_wave < dst_wave_limit(straggler)) {
            emit_wave(straggler);
        }
    }

    // Isolated two-pipe ingress bound t(d) per dst (shared by E3 and the
    // F3b cold classifier); returns t_max.
    double ingress_bounds(vector<double>& t) const {
        t.assign(_node_count, 0.0);
        double t_max = 0.0;
        for (uint32_t d = 0; d < _node_count; d++) {
            double cross = 0.0, same = 0.0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s == d) {
                    continue;
                }
                if (s / _gpus == d / _gpus) {
                    same += (double)_demand[s][d];
                } else {
                    cross += (double)_demand[s][d];
                }
            }
            t[d] = max(same / nv_of(d),
                       (same + cross) / (nv_of(d) + _nic_gbps));
            t_max = max(t_max, t[d]);
        }
        return t_max;
    }

    // Safety gate: skew mechanisms pay only when one dst's bound clearly
    // dominates (hot-dst structure). Without it (uniform, persrc) the
    // slack ranking is noise and demoting dsts can only hurt, so the
    // mechanism self-disables.
    bool skew_gate(const vector<double>& t, double t_max) const {
        if (t_max <= 0.0) {
            return false;
        }
        vector<double> sorted_t = t;
        sort(sorted_t.begin(), sorted_t.end());
        double median = sorted_t[_node_count / 2];
        return median > 0.0 && t_max >= 2.0 * median;
    }

    // E3: per-dst wave offsets from cut-bound slack. t(d) is the isolated
    // two-pipe ingress bound; the predicted straggler gets offset 0, a
    // dst with full slack gets K.
    void init_epoch_offsets() {
        _epoch_offset.assign(_node_count, 0);
        if (_skew_epoch == 0 || _nv_gbps <= 0.0 || _nic_gbps <= 0.0) {
            return;
        }
        vector<double> t;
        double t_max = ingress_bounds(t);
        if (!skew_gate(t, t_max)) {
            return;
        }
        for (uint32_t d = 0; d < _node_count; d++) {
            _epoch_offset[d] = (uint32_t)
                (_skew_epoch * (1.0 - t[d] / t_max) + 0.5);
        }
    }

    // One line per completion event + one line per still-unfinished dst:
    // un-emitted residual (lambda can still steer it), in-flight bytes
    // (already committed to a corridor), and wave cursor.
    void trace_dst_done(uint32_t done_dst) const {
        cout << "mosaic_dst_done k " << _done_count
             << " dst " << done_dst
             << " t_us " << (double)EventList::now() / 1e6 << endl;
        for (uint32_t d = 0; d < _node_count; d++) {
            if (_dst_done[d] || _dst_expected[d] == 0) {
                continue;
            }
            uint64_t unemitted = 0;
            for (uint32_t s = 0; s < _node_count; s++) {
                if (s != d && _remaining[s][d] > 0) {
                    unemitted += _remaining[s][d];
                }
            }
            uint64_t undelivered = _dst_expected[d] - _dst_delivered[d];
            cout << "mosaic_dst_open k " << _done_count
                 << " dst " << d
                 << " unemitted " << unemitted
                 << " inflight " << (undelivered - unemitted)
                 << " undelivered " << undelivered
                 << " wave " << _dst_state[d].emit_wave
                 << endl;
        }
    }

    void print_wave(uint32_t d, uint32_t w, const WaveStats& wave) const {
        const DstState& state = _dst_state[d];
        cout << "mosaic_wave dst " << d
             << " wave " << w
             << " t_us " << (double)EventList::now() / 1e6
             << " emitted " << wave.emitted
             << fixed << setprecision(3);
        const char* rail_names[MAX_CORRIDORS] = {"direct", "tx", "rx",
                                                 "sp3", "sp3b", "sp3c"};
        const char* pod_names[MAX_CORRIDORS] = {"direct", "l1", "l2",
                                                "l3", "l4", "l5"};
        const char* r2p_names[MAX_CORRIDORS] = {"dirA", "dirB", "tx",
                                                "rx", "sp3", "sp3b"};
        const char** names = _wiring == WIRING_POD ? pod_names
            : (_wiring == WIRING_RAIL2P ? r2p_names : rail_names);
        for (uint32_t c = 0; c < CORRIDORS; c++) {
            if (!corridor_enabled(c)) {
                continue;
            }
            double us = wave.corridor_last[c] > wave.start
                ? (double)(wave.corridor_last[c] - wave.start) / 1e6
                : 0.0;
            cout << " " << names[c]
                 << " " << wave.corridor_bytes[c]
                 << " " << us
                 << " " << state.lambda[c];
        }
        cout << defaultfloat << setprecision(6) << endl;
    }

    SmartGraph& _graph;
    uint32_t _node_count;
    uint32_t _gpus;
    uint32_t _servers;
    vector<vector<mem_b> > _demand;
    vector<vector<mem_b> > _remaining;
    uint32_t _waves;
    double _gamma;
    double _floor;
    double _threshold;
    bool _adapt;
    double _nv_gbps;
    double _nic_gbps;
    uint32_t _corridor_count;
    uint32_t _wiring;
    uint32_t _spine_count;
    double _planeb_factor;
    vector<ChainState*> _chains;
    vector<DstState> _dst_state;
    vector<SrcState> _src_state;           // src-driven dispatch (task 2)
    bool _src_driven = false;
    vector<uint64_t> _dst_expected;
    vector<uint64_t> _dst_delivered;
    vector<bool> _dst_done;
    uint32_t _done_count = 0;
    bool _trace = false;
    bool _donate_window = false;
    bool _donate_rewaterfill = false;
    bool _hop_progress_fix = false;
    uint32_t _cold_pace = 0;               // F6 admission pacing (0 = off)
    uint32_t _pace_hot = 0;                // F6 predicted straggler
    vector<char> _pace_is_cold;            // F6 classifier (gated)
    bool _releasing = false;               // release_paced re-entry guard
    bool _host_lambda = false;             // F7 per-source-host lambda
    uint32_t _skew_epoch = 0;
    uint32_t _pair_color = 0;   // O1: epoch sub-slot count, 0 = off
    // x41 ledger state (all unused unless -mosaic_ledger)
    bool _ledger = false;
    bool _lg_correct = true;
    mem_b _lg_chain_cap = 1048576;   // store-forward pipeline granularity
    uint32_t _lg_leaves = 0, _lg_spines = 1, _lg_fab0 = 0, _lg_nres = 0;
    vector<double> _lg_chg, _lg_dis;     // cumulative charge/discharge
    vector<double> _lg_dis_snap;         // telemetry view (strong/snap)
    vector<double> _lg_rate_snap;        // posted rates (strong/snap)
    vector<vector<double> > _lg_vchg, _lg_vdis;   // NONE: per-origin
    uint32_t _lg_sync = 0;               // LG_ORACLE default
    simtime_picosec _lg_sync_ps = 0;
    struct LgEmit {
        simtime_picosec due;
        uint32_t d;
        vector<double> dis;
        vector<double> rate;
    };
    vector<LgEmit> _lg_emitq;            // strong: delayed emissions
    vector<double> _lg_rate;             // effective gbps (corrected)
    vector<double> _lg_nom;              // nominal gbps
    vector<mem_b> _lg_arr;               // queued-at-hop bytes
    vector<simtime_picosec> _lg_prev;    // last delivery on resource
    vector<char> _lg_busyafter;          // work remained after it
    vector<double> _lg_bacc, _lg_tacc;   // busy-period byte/time accum
    vector<uint32_t> _lg_streak;         // consecutive deficit folds
    uint32_t _lg_locked = 0;             // NICs currently rate-locked
    // x44 protocol state (all unused unless set_ledger_proto)
    bool _lg_bcast = false;              // channel B: policy broadcasts
    bool _lg_spineagg = false;           // two-level: spine by feedback
    bool _lg_fabmask = false;            // fabric entries off the ledger
    bool _lg_forkprobe = false;          // replay-fork sensitivity probe
    mutable bool _lg_probe = false;      // probe pass flag (prev snap)
    simtime_picosec _lg_blag_ps = 0;     // broadcast latency (absolute)
    simtime_picosec _lg_stagger_ps = 0;  // decision slot spacing
    simtime_picosec _lg_next_snap = 0;   // next periodic snapshot tick
    double _lg_blagw = 0.0;              // extra lag, publish intervals
    vector<vector<double> > _lg_chg_by;  // per-publisher-dst charges
    vector<vector<double> > _lg_pub;     // last VISIBLE publication
    vector<double> _lg_pubsum;           // sum of visible publications
    vector<simtime_picosec> _lg_prevpub;
    struct LgPub {
        simtime_picosec due;
        uint32_t d;
        vector<double> chg;
    };
    vector<LgPub> _lg_pubq;              // in-flight publications
    uint64_t _lg_pub_cnt = 0;
    uint64_t _lg_pub_entries = 0;        // nonzero (src,corridor) shares
    vector<double> _lg_dis_snap_prev;    // one snap period older view
    double _lg_fork_bytes = 0.0, _lg_fork_total = 0.0;
    vector<double> _lg_phi;              // per-dst fabric share (-1 unset)

    double _guard_eps = 0.0;    // O3: min predicted gain to move lambda
    vector<uint32_t> _epoch_offset;
    uint64_t _wave_bytes = 0;              // F1: 0 = legacy fixed count
    vector<uint32_t> _dst_waves;           // F1: per-dst wave count
    vector<uint32_t> _align_offset;        // F2: W_max - W_d epoch shift
    uint32_t _ramp_start = 0;              // F3: all-dst corridor ramp
    uint32_t _ramp_cold = 0;               // F3b: slack-dst-only ramp
    bool _ramp_event = false;              // growth: waves vs completions
    vector<char> _ramp_is_cold;            // F3b classifier (gated)
    bool _phase_plan = false;
    vector<vector<double> > _plan_nv_share;  // [phase][dst], -1 = keep
    vector<uint32_t> _plan_order;
    // CUBE wiring ground truth (set_cube_profile).
    vector<vector<double> > _cube_local;
    vector<vector<vector<uint32_t> > > _cube_paths;
    vector<double> _cube_bin;
    vector<vector<uint32_t> > _cube_nbrs;
    bool _cube_nic_cap = false;
    bool _cube_lane_rot = false;
    vector<vector<vector<LpPath> > > _lp_intra;  // [j][d] path splits
    bool _lp_local = false;
    // LP-v2: [r][rd] -> (weight, transit rail j), sorted desc.
    vector<vector<vector<pair<double, uint32_t> > > > _lp_jdist;
    const vector<vector<mem_b> >* _prior_demand = NULL;
    uint64_t _expected_app_bytes = 0;
    uint64_t _delivered_app_bytes = 0;
    uint64_t _hops_emitted = 0;
    uint64_t _hop_bytes_emitted = 0;
};

#endif
#endif
#endif
