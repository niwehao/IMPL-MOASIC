// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef FAST_SCHEDULE_H
#define FAST_SCHEDULE_H

// FAST (arXiv 2505.09764, MangoBoost/CMU/UW/UPenn) integrated as an htsim
// scheduler. Ported from FAST/simulation (academic/non-commercial license):
// the analytical cost model there is replaced by real packet-level execution;
// the scheduling algorithm itself (intra-server balance, Birkhoff/Hungarian
// decomposition of the server-level matrix into permutation stages, receiver
// dispatch, stage pipelining) is kept faithful, at byte granularity, and it
// natively consumes NON-UNIFORM alltoallv demand matrices.
//
// The plan is emitted as path-level steps (same structs the TE-CCL runtime
// executes): chunk chains  origin -> [balance] -> inter (same local index)
// -> [dispatch]  with epochs  balance=0, intrinsic intra A2A=1,
// inter of stage p = 1+p, dispatch of stage p = 2+p. Chunk-availability is
// the hard dependency; min-epoch send priority preserves stage order while
// per-port pacing lets NVLink (balance/dispatch) overlap the NIC stages --
// exactly FAST's pipeline. gpus_per_server=1 degenerates to pure Birkhoff
// permutation scheduling of the node-level matrix (no scale-up layer).

#include "config.h"
#include "common.h"
#include "teccl_schedule.h"

#include <algorithm>
#include <map>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

class SmartFastScheduleScheduler {
public:
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<SmartFastScheduleScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk
            <SmartFastScheduleScheduler>(node, chunk);
    }

    static void send_chunk(SmartNode* node, const SmartNode::PendingChunk& chunk,
                           Route* route, int packet_size) {
        SchedulerCommon::send_chunk(node, chunk, route, packet_size);
    }

    static bool send_packet_from_chunk(
        SmartNode* node, SmartNode::ChunkIter chunk,
        Route* route, int packet_size) {
        return SchedulerCommon::send_packet_from_chunk(node, chunk, route,
                                                       packet_size);
    }

    static void do_next_event(SmartNode* node) {
        SchedulerCommon::do_next_event<SmartFastScheduleScheduler>(node);
    }

    // Mapping-semantics sensitivity knob (fairness check): pool-order
    // round-robin instead of min-epoch, mirroring the SyCCL semantics
    // experiment. Off by default (min-epoch preserves stage order).
    static bool& round_robin_ref() {
        static bool round_robin = false;
        return round_robin;
    }

    // Min-epoch priority: FAST stages are strictly ordered permutations, so
    // earlier stages must win the port; equal-epoch chunks round-robin via
    // send_packet_from_chunk's requeue-to-back.
    static SmartNode::ChunkIter
    select_chunk_to_send(SmartNode* node, Route*& selected_route) {
        if (round_robin_ref()) {
            for (SmartNode::ChunkIter it =
                     node->_chunk_pool.begin();
                 it != node->_chunk_pool.end(); ++it) {
                Route* route = sendable_route_for_chunk(node, *it);
                if (route) {
                    selected_route = route;
                    return it;
                }
            }
            selected_route = NULL;
            return node->_chunk_pool.end();
        }

        // Min-epoch selection via the per-epoch FIFO bucket index:
        // ascending epoch, FIFO within an epoch, first sendable chunk
        // wins. Identical to the historical linear scan (min epoch over
        // pool order with requeue-to-absolute-back: across epochs only
        // the minimum matters, within an epoch relative order = ticket
        // order), but O(epochs + unsendable prefix) instead of O(pool)
        // -- the pool holds ~20k live chunks per node at 1024 GPUs and
        // the old scan-plus-vector-memmove cost O(pool) per PACKET.
        for (map<uint32_t, map<uint64_t, SmartNode::ChunkIter> >::iterator
                 b = node->_epoch_buckets.begin();
             b != node->_epoch_buckets.end(); ++b) {
            for (map<uint64_t, SmartNode::ChunkIter>::iterator e =
                     b->second.begin();
                 e != b->second.end(); ++e) {
                SmartNode::ChunkIter it = e->second;
                Route* route = sendable_route_for_chunk(node, *it);
                if (route) {
                    selected_route = route;
                    return it;
                }
            }
        }
        selected_route = NULL;
        return node->_chunk_pool.end();
    }

    static Route* primary_route_for_chunk(SmartNode* node,
                                          const SmartNode::PendingChunk& chunk) {
        uint32_t next = select_next_hop_for_chunk(node, chunk);
        return route_for_next_hop(node, next);
    }

    static Route* route_for_split_chunk(SmartNode* node,
                                        SmartNode::PendingChunk& chunk) {
        uint32_t next = select_next_hop_for_chunk(node, chunk);
        if (next == UINT32_MAX) {
            return NULL;
        }
        chunk.selected_split_class = 0;
        chunk.selected_next_hop = next;
        return route_for_next_hop(node, next);
    }

    static Route* route_for_next_hop(SmartNode* node, uint32_t next) {
        return SchedulerCommon::route_for_next_hop(node, next);
    }

    static BaseQueue* output_queue(Route* route) {
        return SchedulerCommon::output_queue(route);
    }

    static uint32_t select_next_hop_for_chunk(
        SmartNode* node, const SmartNode::PendingChunk& chunk) {
        const vector<uint32_t>& path = chunk.meta->path;
        if (path.size() < 2) {
            return UINT32_MAX;
        }
        for (uint32_t i = 0; i + 1 < path.size(); i++) {
            if (path[i] == node->_id) {
                return path[i + 1];
            }
        }
        return UINT32_MAX;
    }

    static uint32_t choose_next_hop_to(SmartNode*, uint32_t) {
        return UINT32_MAX;
    }

    static uint32_t split_class_for_bucket(uint32_t) {
        return SchedulerCommon::split_class_for_bucket(0);
    }

    static uint32_t split_weight_total() {
        return SchedulerCommon::split_weight_total();
    }

    static void record_split_packet(uint32_t) {
        SchedulerCommon::record_split_packet(0);
    }

    static simtime_picosec pacing_interval(SmartNode* node, int packet_bytes,
                                           uint32_t next_hop) {
        return SchedulerCommon::pacing_interval(node, packet_bytes, next_hop);
    }

    static void schedule_send_event(SmartNode* node, simtime_picosec delay) {
        SchedulerCommon::schedule_send_event(node, delay);
    }

    static linkspeed_bps rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        return SchedulerCommon::rate_to_neighbor(node, neighbor);
    }

    static void reduce_rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        SchedulerCommon::reduce_rate_to_neighbor(node, neighbor);
    }
};

#endif


#ifdef SMART_GRAPH_DEFINED
#ifndef FAST_SCHEDULE_PLAN_H
#define FAST_SCHEDULE_PLAN_H

// Builds the FAST plan for an arbitrary (possibly non-uniform) alltoallv
// demand matrix on the already-constructed htsim graph. The result is a
// SmartTecclSchedule in path mode, executed by SmartTecclScheduleRuntime.
class FastPlanBuilder {
public:
    FastPlanBuilder(const SmartGraph& graph, uint32_t node_count,
                    uint32_t gpus_per_server,
                    const vector<vector<mem_b> >& demand)
        : _graph(graph), _node_count(node_count), _gpus(gpus_per_server),
          _demand(demand) {
        if (_gpus == 0 || _node_count % _gpus != 0) {
            throw runtime_error("fast: gpus_per_server must divide node count");
        }
        _servers = _node_count / _gpus;
        _next_chunk_id.resize(_node_count, 0);
    }

    // FAST's alpha-beta guard (LocalScheduler::check_balance_dispatch,
    // FAST/simulation/scheduler/local.cpp:187-216): when the intra-link
    // beta cost of a server pair's balance+dispatch is not clearly dominant
    // over its alpha cost (ratio < 2), skip the sender-side balance for
    // that pair and size the server-matrix entry by the max row instead of
    // the ceil-average. Off by default to keep prior results reproducible.
    void enable_alpha_beta_guard(double intra_alpha_us, double intra_gbps) {
        _guard_enabled = true;
        _intra_alpha_us = intra_alpha_us;
        _intra_us_per_byte = 8.0 / (intra_gbps * 1000.0);
    }

    uint32_t guard_skipped_pairs() const {
        return _guard_skipped;
    }

    // FAST's scaling path (FastAll2All::to_scaled_matrix) at byte
    // granularity: quantize server-matrix entries up to multiples of
    // `unit` before the Birkhoff decomposition. Fewer distinct levels ->
    // fewer permutation stages; the ceil slack behaves like SDSM padding
    // (pulls stop at real bytes, so it is never transmitted).
    void set_scaling_unit(mem_b unit) {
        _scaling_unit = unit > 0 ? unit : 1;
    }

    // false = raw Birkhoff extraction order (this port's historical
    // behavior); true (default) = production FAST's largest-freq-first.
    void set_stage_order_desc(bool desc) {
        _stage_order_desc = desc;
    }

    SmartTecclSchedule build() {
        SmartTecclSchedule schedule;
        schedule.node_count = _node_count;
        schedule.chunk_size_bytes = max_demand_bytes();
        schedule.epoch_duration = 1.0;
        schedule.hop_level = false;

        // Per-server working state: cross-server tile cells with
        // per-contribution provenance (origin gpu, bytes). cell(m, i, n, j)
        // holds bytes currently parked on (m,i) whose final dst is (n,j).
        _cells.assign(_servers,
                      vector<vector<vector<vector<Contribution> > > >(
                          _gpus,
                          vector<vector<vector<Contribution> > >(
                              _servers, vector<vector<Contribution> >(_gpus))));
        for (uint32_t m = 0; m < _servers; m++) {
            for (uint32_t i = 0; i < _gpus; i++) {
                for (uint32_t n = 0; n < _servers; n++) {
                    if (n == m) {
                        continue;
                    }
                    for (uint32_t j = 0; j < _gpus; j++) {
                        mem_b bytes = _demand[m * _gpus + i][n * _gpus + j];
                        if (bytes > 0) {
                            _cells[m][i][n][j].push_back(
                                Contribution(i, bytes));
                        }
                    }
                }
            }
        }

        // Intrinsic intra-server A2A: overlapped with the first inter stage.
        for (uint32_t m = 0; m < _servers; m++) {
            for (uint32_t i = 0; i < _gpus; i++) {
                for (uint32_t j = 0; j < _gpus; j++) {
                    if (i == j) {
                        continue;
                    }
                    mem_b bytes = _demand[m * _gpus + i][m * _gpus + j];
                    if (bytes > 0) {
                        emit_step(schedule, 1, m * _gpus + i, m * _gpus + i,
                                  m * _gpus + j, bytes);
                    }
                }
            }
        }

        // Sender-side balance (FAST server2server_balance): equalize each
        // server's per-row outgoing bytes toward every remote server so all
        // NICs of the bottleneck server run at full rate. Balance hops are
        // emitted per pulled contribution below; here we only move cell
        // provenance and compute the server-level matrix entry (the
        // post-balance per-row bytes, FAST's server2server_data).
        vector<vector<mem_b> > server_matrix(
            _servers, vector<mem_b>(_servers, 0));
        for (uint32_t m = 0; m < _servers; m++) {
            for (uint32_t n = 0; n < _servers; n++) {
                if (n == m) {
                    continue;
                }
                server_matrix[m][n] = balance_server_pair(m, n);
                if (_scaling_unit > 1) {
                    server_matrix[m][n] =
                        ((server_matrix[m][n] + _scaling_unit - 1)
                         / _scaling_unit) * _scaling_unit;
                }
            }
        }

        // Birkhoff decomposition of the (padded) server-level matrix into
        // permutation stages.
        vector<Permutation> stages = decompose(server_matrix);

        // Production FAST executes the stages largest-freq first ("put the
        // largest cross-node transfer at first to pipeline with intrinsic
        // alltoall", FAST/nvidia/src/fast_alltoall/
        // alltoall_global_scheduler.cc:287-292). Default follows that;
        // -fast_stage_order plan keeps the raw Birkhoff extraction order
        // (this port's historical behavior, for archived-number replay).
        if (_stage_order_desc) {
            stable_sort(stages.begin(), stages.end(),
                        [](const Permutation& a, const Permutation& b) {
                            return a.freq > b.freq;
                        });
        }

        // Per stage: every row (NIC) of server m sends up to freq bytes to
        // the same local index on the matched server; receiver-side dispatch
        // moves it on to the final gpu. Padding never materializes: pulls
        // stop when the real cells are empty.
        for (uint32_t p = 0; p < stages.size(); p++) {
            const Permutation& stage = stages[p];
            for (uint32_t m = 0; m < _servers; m++) {
                uint32_t n = stage.match[m];
                if (n == m) {
                    continue;  // synthetic padding self-match
                }
                for (uint32_t i = 0; i < _gpus; i++) {
                    pull_row(schedule, p, m, i, n, stage.freq);
                }
            }
        }

        for (uint32_t m = 0; m < _servers; m++) {
            for (uint32_t i = 0; i < _gpus; i++) {
                for (uint32_t n = 0; n < _servers; n++) {
                    for (uint32_t j = 0; j < _gpus; j++) {
                        if (!_cells[m][i][n][j].empty()) {
                            throw runtime_error(
                                "fast: undelivered demand left after all stages");
                        }
                    }
                }
            }
        }

        if (schedule.steps.empty()) {
            throw runtime_error("fast: demand matrix produced no transfers");
        }
        _stage_count = (uint32_t)stages.size();
        return schedule;
    }

    uint32_t stage_count() const {
        return _stage_count;
    }

    // Per-step role tags (parallel to schedule.steps) for the runtime's
    // optional stage synchronization (enable_fast_sync).
    const vector<FastStepTag>& step_tags() const {
        return _step_tags;
    }

private:
    struct Contribution {
        Contribution(uint32_t origin_gpu, mem_b byte_count)
            : origin(origin_gpu), bytes(byte_count) {
        }

        uint32_t origin;  // local gpu index the data started on
        mem_b bytes;
    };

    struct Permutation {
        vector<uint32_t> match;
        mem_b freq;
    };

    mem_b cell_bytes(uint32_t m, uint32_t i, uint32_t n, uint32_t j) const {
        mem_b total = 0;
        for (const Contribution& c : _cells[m][i][n][j]) {
            total += c.bytes;
        }
        return total;
    }

    mem_b row_bytes(uint32_t m, uint32_t i, uint32_t n) const {
        mem_b total = 0;
        for (uint32_t j = 0; j < _gpus; j++) {
            total += cell_bytes(m, i, n, j);
        }
        return total;
    }

    // FAST LocalScheduler::server2server_balance at byte granularity: move
    // provenance-tagged bytes from rows above the ceil-average to rows below
    // it. Returns the post-balance per-row target (server matrix entry).
    mem_b balance_server_pair(uint32_t m, uint32_t n) {
        vector<mem_b> sums(_gpus, 0);
        mem_b total = 0;
        for (uint32_t i = 0; i < _gpus; i++) {
            sums[i] = row_bytes(m, i, n);
            total += sums[i];
        }
        if (total == 0) {
            return 0;
        }
        if (_guard_enabled && guard_skip_balance(m, n, sums)) {
            _guard_skipped++;
            mem_b max_row = 0;
            for (uint32_t i = 0; i < _gpus; i++) {
                max_row = max(max_row, sums[i]);
            }
            return max_row;  // server2server_approx: no balance moves
        }
        mem_b target = (total + _gpus - 1) / _gpus;

        for (uint32_t big = 0; big < _gpus; big++) {
            if (sums[big] <= target) {
                continue;
            }
            for (uint32_t small = 0; small < _gpus && sums[big] > target;
                 small++) {
                if (sums[small] >= target) {
                    continue;
                }
                for (uint32_t j = 0; j < _gpus && sums[big] > target
                         && sums[small] < target; j++) {
                    vector<Contribution>& from = _cells[m][big][n][j];
                    vector<Contribution>& to = _cells[m][small][n][j];
                    while (!from.empty() && sums[big] > target
                           && sums[small] < target) {
                        Contribution& c = from.back();
                        mem_b move = min(c.bytes,
                                         min(sums[big] - target,
                                             target - sums[small]));
                        to.push_back(Contribution(c.origin, move));
                        c.bytes -= move;
                        if (c.bytes == 0) {
                            from.pop_back();
                        }
                        sums[big] -= move;
                        sums[small] += move;
                    }
                }
            }
        }
        return target;
    }

    // Faithful port of check_balance_dispatch: alpha cost = one balance
    // step plus one step per non-empty dispatch diagonal; beta cost = the
    // max balanced row plus the largest element of each diagonal.
    bool guard_skip_balance(uint32_t m, uint32_t n,
                            const vector<mem_b>& sums) const {
        mem_b max_row = 0;
        for (uint32_t i = 0; i < _gpus; i++) {
            max_row = max(max_row, sums[i]);
        }
        double alpha_cost = _intra_alpha_us;
        double beta_cost = (double)max_row * _intra_us_per_byte;
        for (uint32_t offset = 0; offset < _gpus; offset++) {
            mem_b largest = 0;
            for (uint32_t j = 0; j < _gpus; j++) {
                largest = max(largest,
                              cell_bytes(m, j, n, (j + offset) % _gpus));
            }
            if (largest != 0) {
                alpha_cost += _intra_alpha_us;
                beta_cost += (double)largest * _intra_us_per_byte;
            }
        }
        return beta_cost / alpha_cost < 2.0;
    }

    // FAST server2server_dispatch: row i of server m sends up to freq bytes
    // (greedy over final dst gpus) to the same local index on server n; each
    // pulled contribution becomes one chunk chain.
    void pull_row(SmartTecclSchedule& schedule, uint32_t stage, uint32_t m,
                  uint32_t i, uint32_t n, mem_b freq) {
        mem_b remaining = freq;
        for (uint32_t j = 0; j < _gpus && remaining > 0; j++) {
            vector<Contribution>& cell = _cells[m][i][n][j];
            while (!cell.empty() && remaining > 0) {
                Contribution& c = cell.back();
                mem_b take = min(c.bytes, remaining);
                emit_chain(schedule, stage, m, c.origin, i, n, j, take);
                c.bytes -= take;
                remaining -= take;
                if (c.bytes == 0) {
                    cell.pop_back();
                }
            }
        }
    }

    void emit_chain(SmartTecclSchedule& schedule, uint32_t stage, uint32_t m,
                    uint32_t origin, uint32_t relay, uint32_t n, uint32_t j,
                    mem_b bytes) {
        uint32_t src_gpu = m * _gpus + origin;
        uint32_t relay_gpu = m * _gpus + relay;
        uint32_t landing_gpu = n * _gpus + relay;
        uint32_t dst_gpu = n * _gpus + j;
        uint32_t chunk = _next_chunk_id[src_gpu]++;

        if (origin != relay) {
            emit_step_chunk(schedule, 0, src_gpu, chunk, src_gpu, relay_gpu,
                            bytes, FastStepTag(FastStepTag::BALANCE, 0));
        }
        emit_step_chunk(schedule, 1 + stage, src_gpu, chunk, relay_gpu,
                        landing_gpu, bytes,
                        FastStepTag(FastStepTag::INTER, stage));
        if (landing_gpu != dst_gpu) {
            emit_step_chunk(schedule, 2 + stage, src_gpu, chunk, landing_gpu,
                            dst_gpu, bytes,
                            FastStepTag(FastStepTag::DISPATCH, stage));
        }
    }

    void emit_step(SmartTecclSchedule& schedule, uint32_t epoch, uint32_t src,
                   uint32_t chunk_source, uint32_t dst, mem_b bytes) {
        emit_step_chunk(schedule, epoch, chunk_source,
                        _next_chunk_id[chunk_source]++, src, dst, bytes,
                        FastStepTag(FastStepTag::INTRA, 0));
    }

    void emit_step_chunk(SmartTecclSchedule& schedule, uint32_t epoch,
                         uint32_t chunk_source, uint32_t chunk, uint32_t src,
                         uint32_t dst, mem_b bytes, const FastStepTag& tag) {
        SmartTecclScheduleStep step;
        step.epoch = epoch;
        step.chunk_source = chunk_source;
        step.src = src;
        step.dst = dst;
        step.chunk = chunk;
        step.bytes = bytes;
        step.path = shortest_path(src, dst);
        schedule.steps.push_back(step);
        _step_tags.push_back(tag);
    }

    // Birkhoff-von-Neumann: pad to a scaled doubly stochastic matrix, then
    // repeatedly extract a perfect matching (Hungarian augmenting paths) and
    // subtract its minimum entry. Ported from FAST FastAll2All::decompose.
    vector<Permutation> decompose(const vector<vector<mem_b> >& matrix) const {
        uint32_t dim = _servers;
        vector<vector<mem_b> > padded = matrix;

        vector<mem_b> row_sum(dim, 0), col_sum(dim, 0);
        mem_b max_sum = 0;
        for (uint32_t i = 0; i < dim; i++) {
            for (uint32_t j = 0; j < dim; j++) {
                row_sum[i] += padded[i][j];
                col_sum[j] += padded[i][j];
            }
        }
        for (uint32_t i = 0; i < dim; i++) {
            max_sum = max(max_sum, max(row_sum[i], col_sum[i]));
        }
        for (uint32_t i = 0; i < dim; i++) {
            for (uint32_t j = 0; j < dim && row_sum[i] < max_sum; j++) {
                if (col_sum[j] >= max_sum) {
                    continue;
                }
                mem_b diff = min(max_sum - row_sum[i], max_sum - col_sum[j]);
                padded[i][j] += diff;
                row_sum[i] += diff;
                col_sum[j] += diff;
            }
        }

        vector<Permutation> stages;
        mem_b freq_done = 0;
        while (freq_done < max_sum) {
            vector<int> match_col(dim, -1);  // column -> row
            for (uint32_t row = 0; row < dim; row++) {
                vector<bool> visited(dim, false);
                if (!augment(padded, row, visited, match_col)) {
                    throw runtime_error(
                        "fast: no perfect matching in padded matrix");
                }
            }

            Permutation stage;
            stage.match.assign(dim, 0);
            stage.freq = 0;
            for (uint32_t col = 0; col < dim; col++) {
                uint32_t row = (uint32_t)match_col[col];
                stage.match[row] = col;
                mem_b value = padded[row][col];
                if (stage.freq == 0 || value < stage.freq) {
                    stage.freq = value;
                }
            }
            for (uint32_t col = 0; col < dim; col++) {
                padded[match_col[col]][col] -= stage.freq;
            }
            freq_done += stage.freq;
            stages.push_back(stage);
        }
        return stages;
    }

    static bool augment(const vector<vector<mem_b> >& padded, uint32_t row,
                        vector<bool>& visited, vector<int>& match_col) {
        uint32_t dim = (uint32_t)padded.size();
        for (uint32_t col = 0; col < dim; col++) {
            if (padded[row][col] <= 0 || visited[col]) {
                continue;
            }
            visited[col] = true;
            if (match_col[col] < 0
                || augment(padded, (uint32_t)match_col[col], visited,
                           match_col)) {
                match_col[col] = (int)row;
                return true;
            }
        }
        return false;
    }

    vector<uint32_t> shortest_path(uint32_t src, uint32_t dst) const {
        map<pair<uint32_t, uint32_t>, vector<uint32_t> >::const_iterator
            cached = _path_cache.find(make_pair(src, dst));
        if (cached != _path_cache.end()) {
            return cached->second;
        }

        uint32_t node_count = (uint32_t)_graph.nodes.size();
        vector<int> previous(node_count, -1);
        queue<uint32_t> pending;
        previous[src] = (int)src;
        pending.push(src);
        while (!pending.empty() && previous[dst] < 0) {
            uint32_t node = pending.front();
            pending.pop();
            for (uint32_t next : _graph.adjacency[node]) {
                if (previous[next] >= 0) {
                    continue;
                }
                previous[next] = (int)node;
                pending.push(next);
            }
        }
        if (previous[dst] < 0) {
            throw runtime_error("fast: graph path is disconnected");
        }

        vector<uint32_t> path;
        for (uint32_t node = dst; node != src;
             node = (uint32_t)previous[node]) {
            path.push_back(node);
        }
        path.push_back(src);
        reverse(path.begin(), path.end());
        _path_cache[make_pair(src, dst)] = path;
        return path;
    }

    mem_b max_demand_bytes() const {
        mem_b max_bytes = 0;
        for (uint32_t src = 0; src < _node_count; src++) {
            for (uint32_t dst = 0; dst < _node_count; dst++) {
                if (src != dst) {
                    max_bytes = max(max_bytes, _demand[src][dst]);
                }
            }
        }
        if (max_bytes <= 0) {
            throw runtime_error("fast: demand matrix is empty");
        }
        return max_bytes;
    }

    const SmartGraph& _graph;
    uint32_t _node_count;
    uint32_t _gpus;
    uint32_t _servers;
    const vector<vector<mem_b> >& _demand;
    vector<vector<vector<vector<vector<Contribution> > > > > _cells;
    vector<uint32_t> _next_chunk_id;
    vector<FastStepTag> _step_tags;
    mutable map<pair<uint32_t, uint32_t>, vector<uint32_t> > _path_cache;
    uint32_t _stage_count = 0;
    bool _guard_enabled = false;
    double _intra_alpha_us = 9.0;
    double _intra_us_per_byte = 8.0 / (2400.0 * 1000.0);
    uint32_t _guard_skipped = 0;
    mem_b _scaling_unit = 1;
    bool _stage_order_desc = true;
};

#endif
#endif
