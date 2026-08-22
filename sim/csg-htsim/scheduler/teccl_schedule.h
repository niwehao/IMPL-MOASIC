// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef TECCL_SCHEDULE_H
#define TECCL_SCHEDULE_H

#include "config.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SmartTecclScheduleStep {
    uint32_t epoch;
    uint32_t chunk_source;
    uint32_t src;
    uint32_t dst;
    uint32_t chunk;
    mem_b bytes;
    std::vector<uint32_t> path;
};

struct SmartTecclHopStep {
    uint32_t epoch;
    uint32_t demand_id;
    uint32_t fragment_id;
    uint32_t hop_index;
    uint32_t chunk_source;
    uint32_t chunk;
    uint32_t final_dst;
    uint32_t src;
    uint32_t dst;
    mem_b bytes;
    mem_b fragment_bytes;
    mem_b demand_bytes;
};

struct SmartTecclSchedule {
    SmartTecclSchedule() : node_count(0), chunk_size_bytes(0),
                           epoch_duration(1.0), hop_level(false) {
    }

    uint32_t node_count;
    mem_b chunk_size_bytes;
    double epoch_duration;
    bool hop_level;
    std::vector<SmartTecclScheduleStep> steps;
    std::vector<SmartTecclHopStep> hops;
};

// Per-step role tag recorded by FastPlanBuilder (fast_schedule.h), parallel
// to SmartTecclSchedule::steps, so the runtime can reproduce the official
// FAST implementation's execution-time stage synchronization (see
// SmartTecclScheduleRuntime::enable_fast_sync).
struct FastStepTag {
    enum Kind { BALANCE = 0, INTRA = 1, INTER = 2, DISPATCH = 3 };

    FastStepTag(uint8_t step_kind = BALANCE, uint32_t step_stage = 0)
        : kind(step_kind), stage(step_stage) {
    }

    uint8_t kind;
    uint32_t stage;  // Birkhoff permutation stage p (0 for BALANCE/INTRA)
};

static SmartTecclSchedule load_teccl_schedule(const std::string& filename) {
    std::ifstream input(filename.c_str());
    if (!input) {
        throw std::runtime_error("cannot open schedule file: " + filename);
    }

    SmartTecclSchedule schedule;
    std::string line;
    uint32_t line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::string kind;
        stream >> kind;
        if (kind == "node_count") {
            stream >> schedule.node_count;
        } else if (kind == "chunk_size_bytes") {
            stream >> schedule.chunk_size_bytes;
        } else if (kind == "epoch_duration") {
            stream >> schedule.epoch_duration;
        } else if (kind == "mode") {
            std::string mode;
            stream >> mode;
            if (mode == "hop_level") {
                schedule.hop_level = true;
            } else {
                throw std::runtime_error("unknown TE-CCL schedule mode '"
                                         + mode + "' at line "
                                         + ntoa(line_number));
            }
        } else if (kind == "step") {
            SmartTecclScheduleStep step;
            std::string path_keyword;
            stream >> step.epoch >> step.chunk_source >> step.src >> step.dst
                   >> step.chunk >> step.bytes >> path_keyword;
            if (!stream || path_keyword != "path") {
                throw std::runtime_error("bad schedule step at line "
                                         + ntoa(line_number));
            }
            uint32_t node = 0;
            while (stream >> node) {
                step.path.push_back(node);
            }
            if (step.path.size() < 2) {
                throw std::runtime_error("schedule step path must have at least two nodes at line "
                                         + ntoa(line_number));
            }
            if (step.path.front() != step.src || step.path.back() != step.dst) {
                throw std::runtime_error("schedule step path endpoints do not match src/dst at line "
                                         + ntoa(line_number));
            }
            std::set<uint32_t> seen_path_nodes;
            for (uint32_t path_node : step.path) {
                if (seen_path_nodes.find(path_node) != seen_path_nodes.end()) {
                    throw std::runtime_error("schedule step path repeats node "
                                             + ntoa(path_node) + " at line "
                                             + ntoa(line_number));
                }
                seen_path_nodes.insert(path_node);
            }
            schedule.steps.push_back(step);
        } else if (kind == "hop") {
            SmartTecclHopStep hop;
            stream >> hop.epoch >> hop.demand_id >> hop.fragment_id
                   >> hop.hop_index >> hop.chunk_source >> hop.chunk
                   >> hop.final_dst >> hop.src >> hop.dst >> hop.bytes
                   >> hop.fragment_bytes >> hop.demand_bytes;
            if (!stream) {
                throw std::runtime_error("bad schedule hop at line "
                                         + ntoa(line_number));
            }
            schedule.hop_level = true;
            schedule.hops.push_back(hop);
        } else {
            throw std::runtime_error("unknown schedule line kind '" + kind
                                     + "' at line " + ntoa(line_number));
        }
    }

    if (schedule.node_count == 0) {
        throw std::runtime_error("schedule missing node_count");
    }
    if (schedule.chunk_size_bytes <= 0) {
        throw std::runtime_error("schedule missing chunk_size_bytes");
    }
    if (!schedule.hop_level && schedule.steps.empty()) {
        throw std::runtime_error("schedule has no steps");
    }
    if (schedule.hop_level && schedule.hops.empty()) {
        throw std::runtime_error("hop-level schedule has no hops");
    }
    return schedule;
}

class SmartTecclScheduleScheduler {
public:
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<SmartTecclScheduleScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk
            <SmartTecclScheduleScheduler>(node, chunk);
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
        SchedulerCommon::do_next_event<SmartTecclScheduleScheduler>(node);
    }

    // Min-epoch priority: TE-CCL epochs quantize solver wall-clock ordering. Within an epoch,
    // send_packet_from_chunk's requeue-to-back still round-robins equal-epoch
    // chunks. NOTE (2026-07-08 experiment): replacing this with pure pool-order
    // round-robin fixed the clos_32 bottleneck-starvation anomaly for TE-CCL
    // (9464->8074us) but regressed relay pipelines badly elsewhere
    // (TACCL dgx_rail +59%), so per-solver semantics were kept; SyCCL uses
    // round-robin. See Note/2026-0710-experiment-summary-v4.md 2.4.
    static SmartNode::ChunkIter
    select_chunk_to_send(SmartNode* node, Route*& selected_route) {
        SmartNode::ChunkIter selected =
            node->_chunk_pool.end();
        uint32_t selected_epoch = UINT32_MAX;

        for (SmartNode::ChunkIter it =
                 node->_chunk_pool.begin();
             it != node->_chunk_pool.end(); ++it) {
            Route* route = sendable_route_for_chunk(node, *it);
            if (!route) {
                continue;
            }
            if (it->meta->round < selected_epoch) {
                selected = it;
                selected_route = route;
                selected_epoch = it->meta->round;
            }
        }
        if (selected == node->_chunk_pool.end()) {
            selected_route = NULL;
        }
        return selected;
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
#ifndef TECCL_SCHEDULE_RUNTIME_H
#define TECCL_SCHEDULE_RUNTIME_H

struct SmartTecclFlowState {
    SmartTecclFlowState(uint32_t flow_id, const SmartTecclScheduleStep& flow_step)
        : id(flow_id), step(flow_step), next_packet_id(1), packet_flow(NULL),
          injected(false), delivered(false) {
    }

    uint32_t id;
    SmartTecclScheduleStep step;
    packetid_t next_packet_id;
    PacketFlow packet_flow;
    bool injected;
    bool delivered;
};

struct SmartTecclHopFlowState {
    SmartTecclHopFlowState(uint32_t flow_id, const SmartTecclHopStep& hop_step)
        : id(flow_id), step(hop_step), next_packet_id(1), packet_flow(NULL),
          injected(false), delivered(false), injected_bytes(0),
          delivered_bytes(0) {
    }

    uint32_t id;
    SmartTecclHopStep step;
    packetid_t next_packet_id;
    PacketFlow packet_flow;
    bool injected;
    bool delivered;
    mem_b injected_bytes;
    mem_b delivered_bytes;
};

class SmartTecclScheduleRuntime : public SmartDeliveryObserver {
public:
    SmartTecclScheduleRuntime(SmartGraph& graph,
                              const SmartTecclSchedule& schedule)
        : _graph(graph), _schedule(schedule), _expected_bytes(0),
          _delivered_steps(0), _delivered_demands(0), _in_flight_steps(0) {
        if (_schedule.node_count > graph.nodes.size()) {
            throw runtime_error("schedule node_count exceeds graph nodes");
        }

        _node_states.resize(_schedule.node_count);
        if (_schedule.hop_level) {
            initialize_hop_level_schedule();
            return;
        }

        initialize_path_schedule();
    }

    void initialize_path_schedule() {
        for (uint32_t i = 0; i < _schedule.steps.size(); i++) {
            validate_step(_schedule.steps[i]);
            _flows.push_back(new SmartTecclFlowState(i, _schedule.steps[i]));
            _node_states[_schedule.steps[i].src]
                .steps_by_epoch[_schedule.steps[i].epoch].push_back(i);
            _expected_bytes += (uint64_t)_schedule.steps[i].bytes;
            _available_chunks.insert(chunk_location_key(
                _schedule.steps[i].chunk_source,
                _schedule.steps[i].chunk,
                _schedule.steps[i].chunk_source));
        }
        for (uint32_t node = 0; node < _node_states.size(); node++) {
            NodeState& state = _node_states[node];
            for (map<uint32_t, vector<uint32_t> >::const_iterator it =
                     state.steps_by_epoch.begin();
                 it != state.steps_by_epoch.end(); ++it) {
                state.epochs.push_back(it->first);
            }
        }
    }

    const vector<SmartTecclFlowState*>& flows() const {
        return _flows;
    }

    uint64_t expected_chunks() const {
        if (_schedule.hop_level) {
            return _demands.size();
        }
        return _flows.size();
    }

    uint64_t expected_bytes() const {
        return _expected_bytes;
    }

    uint64_t flow_count() const {
        if (_schedule.hop_level) {
            return _demands.size();
        }
        return _flows.size();
    }

    void start() {
        if (_schedule.hop_level) {
            try_inject_ready_steps();
            return;
        }

        for (uint32_t node = 0; node < _node_states.size(); node++) {
            start_next_node_epoch(node);
        }
        try_inject_ready_steps();
    }

    // FAST execution-time stage synchronization (arXiv 2505.09764). Call
    // before start(). Modes:
    //  - FAST_SYNC_CREDIT: faithful port of the official NVSHMEM kernel's
    //    runtime gates (FAST/nvidia/src/fast_alltoall/
    //    flash_alltoall_nvshmem.cu):
    //      (a) a rank's inter sends wait for its own balance receives
    //          (wait_balance_complete + stream2->stream1 event, :2544-2557);
    //      (b) inter sends are stage-serial per rank, and stage p starts
    //          only after stage p-1's put completed remotely (stream1
    //          ordering + nvshmem_quiet, :1717-1730);
    //      (c) receiver-driven credit over the 2-deep ping-pong buffers:
    //          stage p's send is granted once the receiver got ALL of its
    //          stage p-1 inter data and (p>=2) finished redistributing
    //          stage p-2 (grant_credit_to_peer, :1662-1689; initial credit
    //          for stage 0, :2515);
    //      (d) receiver side: dispatch (redistribute) of stage p starts
    //          only after ALL stage-p inter data arrived (wait_stage_ready,
    //          :2616) and behind the intrinsic A2A and the previous
    //          redistribute on the same stream (stream2 ordering).
    //  - FAST_SYNC_BARRIER: the official analytical model's global stage
    //    barrier (FAST/simulation/scheduler/global.cpp:150-212): segments
    //    [balance] -> [inter 0 || intrinsic] -> [inter p || dispatch p-1]
    //    -> [dispatch P-1], each segment starting only after the previous
    //    one completed globally.
    //  - FAST_SYNC_NONE (default): legacy behavior, chunk availability is
    //    the only inter-step constraint.
    enum FastSyncMode {
        FAST_SYNC_NONE = 0,
        FAST_SYNC_CREDIT = 1,
        FAST_SYNC_BARRIER = 2
    };

    void enable_fast_sync(FastSyncMode mode, const vector<FastStepTag>& tags,
                          uint32_t stage_count) {
        if (mode == FAST_SYNC_NONE) {
            return;
        }
        if (_schedule.hop_level) {
            throw runtime_error("fast_sync requires a path-mode schedule");
        }
        if (tags.size() != _schedule.steps.size()) {
            throw runtime_error("fast_sync tag/step count mismatch");
        }
        _fast_sync_mode = mode;
        _fast_tags = tags;
        uint32_t nodes = _schedule.node_count;
        uint32_t stages = max(stage_count, 1u);
        _fs_exp_bal_in.assign(nodes, 0);
        _fs_del_bal_in.assign(nodes, 0);
        _fs_exp_intra_out.assign(nodes, 0);
        _fs_del_intra_out.assign(nodes, 0);
        _fs_exp_inter_in.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_del_inter_in.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_exp_inter_out.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_del_inter_out.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_exp_disp_out.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_del_disp_out.assign(nodes, vector<uint32_t>(stages, 0));
        _fs_exp_stage_inter.assign(stages, 0);
        _fs_del_stage_inter.assign(stages, 0);
        _fs_exp_stage_disp.assign(stages, 0);
        _fs_del_stage_disp.assign(stages, 0);
        _fs_exp_bal_total = _fs_del_bal_total = 0;
        _fs_exp_intra_total = _fs_del_intra_total = 0;

        for (uint32_t i = 0; i < _schedule.steps.size(); i++) {
            const SmartTecclScheduleStep& step = _schedule.steps[i];
            const FastStepTag& tag = tags[i];
            if (tag.stage >= stages) {
                throw runtime_error("fast_sync tag stage out of range");
            }
            switch (tag.kind) {
            case FastStepTag::BALANCE:
                _fs_exp_bal_in[step.dst]++;
                _fs_exp_bal_total++;
                break;
            case FastStepTag::INTRA:
                _fs_exp_intra_out[step.src]++;
                _fs_exp_intra_total++;
                break;
            case FastStepTag::INTER:
                _fs_exp_inter_in[step.dst][tag.stage]++;
                _fs_exp_inter_out[step.src][tag.stage]++;
                _fs_exp_stage_inter[tag.stage]++;
                break;
            case FastStepTag::DISPATCH:
                _fs_exp_disp_out[step.src][tag.stage]++;
                _fs_exp_stage_disp[tag.stage]++;
                break;
            default:
                throw runtime_error("fast_sync unknown step kind");
            }
        }
    }

    void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                         uint32_t dst_node, mem_b bytes) override {
        if (_schedule.hop_level) {
            return;
        }

        (void)chunk_id;
        (void)bytes;
        if (app_flow_id >= _flows.size()) {
            throw runtime_error("delivered unknown TE-CCL flow id");
        }

        SmartTecclFlowState* flow = _flows[app_flow_id];
        const SmartTecclScheduleStep& step = flow->step;
        if (dst_node != step.dst) {
            throw runtime_error("TE-CCL step delivered to wrong node");
        }
        if (flow->delivered) {
            throw runtime_error("TE-CCL step delivered more than once");
        }

        flow->delivered = true;
        _available_chunks.insert(chunk_location_key(step.chunk_source,
                                                    step.chunk,
                                                    step.dst));
        if (_fast_sync_mode != FAST_SYNC_NONE) {
            const FastStepTag& tag = _fast_tags[app_flow_id];
            switch (tag.kind) {
            case FastStepTag::BALANCE:
                _fs_del_bal_in[step.dst]++;
                _fs_del_bal_total++;
                break;
            case FastStepTag::INTRA:
                _fs_del_intra_out[step.src]++;
                _fs_del_intra_total++;
                break;
            case FastStepTag::INTER:
                _fs_del_inter_in[step.dst][tag.stage]++;
                _fs_del_inter_out[step.src][tag.stage]++;
                _fs_del_stage_inter[tag.stage]++;
                break;
            case FastStepTag::DISPATCH:
                _fs_del_disp_out[step.src][tag.stage]++;
                _fs_del_stage_disp[tag.stage]++;
                break;
            }
        }
        _delivered_steps++;
        if (_in_flight_steps == 0) {
            throw runtime_error("TE-CCL delivered step with no in-flight step");
        }
        _in_flight_steps--;

        NodeState& state = _node_states[step.src];
        if (state.in_flight == 0) {
            throw runtime_error("TE-CCL delivered step with no local in-flight step");
        }
        state.in_flight--;
        state.delivered++;

        try_inject_ready_steps();
    }

    void packet_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                          uint32_t dst_node, mem_b bytes) override {
        if (_schedule.hop_level) {
            hop_packet_delivered(app_flow_id, chunk_id, dst_node, bytes);
        }
    }

    void print_schedule_report() const {
        cout << "teccl_schedule_report"
             << " mode " << (_schedule.hop_level ? "hop_level" : "path")
             << " forwarding " << (SmartNode::store_forward_enabled()
                                    ? "store_forward" : "cut_through")
             << " local_epoch_slots " << local_epoch_slot_count()
             << " steps " << (_schedule.hop_level ? _hop_flows.size() : _flows.size())
             << " delivered_steps " << _delivered_steps
             << " delivered_demands " << _delivered_demands
             << " active_nodes " << active_node_count()
             << " pending_steps " << pending_step_count()
             << " in_flight_steps " << _in_flight_steps
             << endl;
    }

private:
    struct NodeState {
        NodeState()
            : epoch_index(0), current_epoch(UINT32_MAX), expected(0),
              delivered(0), in_flight(0), complete(false) {
        }

        map<uint32_t, vector<uint32_t> > steps_by_epoch;
        vector<uint32_t> epochs;
        vector<uint32_t> pending;
        uint32_t epoch_index;
        uint32_t current_epoch;
        uint32_t expected;
        uint32_t delivered;
        uint32_t in_flight;
        bool complete;
    };

    struct DemandState {
        DemandState()
            : id(UINT32_MAX), chunk_source(UINT32_MAX), chunk(UINT32_MAX),
              final_dst(UINT32_MAX), expected_bytes(0), delivered_bytes(0),
              complete(false) {
        }

        uint32_t id;
        uint32_t chunk_source;
        uint32_t chunk;
        uint32_t final_dst;
        mem_b expected_bytes;
        mem_b delivered_bytes;
        bool complete;
    };

    struct FragmentState {
        FragmentState()
            : id(UINT32_MAX), demand_id(UINT32_MAX), bytes(0),
              current_node(UINT32_MAX), next_hop_index(0), total_hops(0),
              delivered(false) {
        }

        uint32_t id;
        uint32_t demand_id;
        mem_b bytes;
        uint32_t current_node;
        uint32_t next_hop_index;
        uint32_t total_hops;
        bool delivered;
    };

    // TE-CCL epochs quantize solver wall-clock; the hard inter-step
    // constraint is chunk availability (step_is_ready). A node advances to
    // its next epoch once the current epoch's sends are all issued, not
    // end-to-end delivered. Counters are cumulative across epochs.
    void start_next_node_epoch(uint32_t node) {
        NodeState& state = _node_states[node];
        if (state.epoch_index >= state.epochs.size()) {
            state.complete = true;
            state.current_epoch = UINT32_MAX;
            return;
        }

        state.complete = false;
        state.current_epoch = state.epochs[state.epoch_index++];
        const vector<uint32_t>& steps =
            state.steps_by_epoch[state.current_epoch];
        state.pending.insert(state.pending.end(), steps.begin(), steps.end());
        state.expected += steps.size();
    }

    bool step_is_ready(const SmartTecclFlowState& flow) const {
        const SmartTecclScheduleStep& step = flow.step;
        uint64_t key = chunk_location_key(step.chunk_source, step.chunk, step.src);
        return _available_chunks.find(key) != _available_chunks.end();
    }

    // FAST stage-sync gates (see enable_fast_sync for the source mapping).
    // All gates are functions of delivery counters only, so re-evaluating
    // on every chunk_delivered() is sufficient for progress.
    bool fast_gate_open(const SmartTecclFlowState& flow) const {
        if (_fast_sync_mode == FAST_SYNC_NONE) {
            return true;
        }
        const FastStepTag& tag = _fast_tags[flow.id];
        uint32_t src = flow.step.src;
        uint32_t p = tag.stage;

        if (_fast_sync_mode == FAST_SYNC_BARRIER) {
            // Global segments: [bal] -> [inter 0 || intra] ->
            // [inter p || dispatch p-1] -> [dispatch P-1].
            switch (tag.kind) {
            case FastStepTag::BALANCE:
                return true;
            case FastStepTag::INTRA:
                return _fs_del_bal_total == _fs_exp_bal_total;
            case FastStepTag::INTER:
                if (_fs_del_bal_total != _fs_exp_bal_total) {
                    return false;
                }
                if (p == 0) {
                    return true;
                }
                if (_fs_del_stage_inter[p - 1] != _fs_exp_stage_inter[p - 1]) {
                    return false;
                }
                if (p == 1) {
                    return _fs_del_intra_total == _fs_exp_intra_total;
                }
                return _fs_del_stage_disp[p - 2] == _fs_exp_stage_disp[p - 2];
            case FastStepTag::DISPATCH:
                if (_fs_del_stage_inter[p] != _fs_exp_stage_inter[p]) {
                    return false;
                }
                if (p == 0) {
                    return _fs_del_intra_total == _fs_exp_intra_total;
                }
                return _fs_del_stage_disp[p - 1] == _fs_exp_stage_disp[p - 1];
            }
            return true;
        }

        // FAST_SYNC_CREDIT: pairwise gates of the official kernel.
        switch (tag.kind) {
        case FastStepTag::BALANCE:
            return true;
        case FastStepTag::INTRA:
            // stream2 ordering: intrinsic A2A launches after this rank's
            // balance receives completed.
            return _fs_del_bal_in[src] == _fs_exp_bal_in[src];
        case FastStepTag::INTER: {
            uint32_t receiver = flow.step.dst;
            // (a) own balance receives done (event0 dependency);
            if (_fs_del_bal_in[src] != _fs_exp_bal_in[src]) {
                return false;
            }
            // (b) stage-serial sends: previous stage's put quiet;
            if (p > 0 && _fs_del_inter_out[src][p - 1]
                             != _fs_exp_inter_out[src][p - 1]) {
                return false;
            }
            // (c) credit: receiver got all of stage p-1 ...
            if (p > 0 && _fs_del_inter_in[receiver][p - 1]
                             != _fs_exp_inter_in[receiver][p - 1]) {
                return false;
            }
            // ... and freed the ping-pong buffer of stage p-2.
            if (p > 1 && _fs_del_disp_out[receiver][p - 2]
                             != _fs_exp_disp_out[receiver][p - 2]) {
                return false;
            }
            return true;
        }
        case FastStepTag::DISPATCH:
            // (d) stream2 ordering: behind intrinsic A2A and dispatch p-1,
            // and only after ALL of stage p's inter data arrived here.
            if (_fs_del_intra_out[src] != _fs_exp_intra_out[src]) {
                return false;
            }
            if (_fs_del_inter_in[src][p] != _fs_exp_inter_in[src][p]) {
                return false;
            }
            if (p > 0 && _fs_del_disp_out[src][p - 1]
                             != _fs_exp_disp_out[src][p - 1]) {
                return false;
            }
            return true;
        }
        return true;
    }

    bool hop_is_ready(const SmartTecclHopFlowState& flow) const {
        const SmartTecclHopStep& hop = flow.step;
        if (hop.fragment_id >= _fragments.size()) {
            return false;
        }
        const FragmentState& fragment = _fragments[hop.fragment_id];
        return !fragment.delivered
            && fragment.current_node == hop.src
            && fragment.next_hop_index == hop.hop_index
            && (SmartNode::store_forward_enabled() || hop.hop_index == 0);
    }

    void inject_step(SmartTecclFlowState& flow) {
        if (flow.injected) {
            throw runtime_error("TE-CCL step injected more than once");
        }
        if (!step_is_ready(flow)) {
            throw runtime_error("TE-CCL tried to inject a step before dependency was ready");
        }

        const SmartTecclScheduleStep& step = flow.step;
        flow.injected = true;
        _in_flight_steps++;
        _node_states[step.src].in_flight++;
        _graph.nodes[step.src]->add_scheduled_chunk(
            flow.packet_flow, step.chunk_source, step.dst, step.epoch, flow.id,
            step.chunk, step.bytes, (uint32_t)(step.path.size() + 4),
            flow.next_packet_id, step.path);
    }

    void inject_hop(SmartTecclHopFlowState& flow) {
        if (flow.injected) {
            throw runtime_error("TE-CCL hop injected more than once");
        }
        if (!hop_is_ready(flow)) {
            throw runtime_error("TE-CCL tried to inject a hop before fragment was ready");
        }

        const SmartTecclHopStep& hop = flow.step;
        vector<uint32_t> path = physical_path(hop.src, hop.dst);
        flow.injected = true;
        flow.injected_bytes = hop.bytes;
        _in_flight_steps++;
        _node_states[hop.src].in_flight++;
        _graph.nodes[hop.src]->add_scheduled_chunk(
            flow.packet_flow, hop.chunk_source, hop.dst, hop.epoch, flow.id,
            hop.fragment_id, hop.bytes, (uint32_t)(path.size() + 4),
            flow.next_packet_id, path, false);
    }

    void inject_hop_packet(SmartTecclHopFlowState& flow, mem_b bytes) {
        if (flow.delivered) {
            throw runtime_error("TE-CCL tried to inject packet into delivered hop");
        }
        if (flow.injected_bytes + bytes > flow.step.bytes) {
            throw runtime_error("TE-CCL hop packet injection exceeds hop bytes");
        }

        const SmartTecclHopStep& hop = flow.step;
        vector<uint32_t> path = physical_path(hop.src, hop.dst);
        if (!flow.injected) {
            flow.injected = true;
            _in_flight_steps++;
            _node_states[hop.src].in_flight++;
            remove_pending_hop(flow.id);
        }
        flow.injected_bytes += bytes;
        _graph.nodes[hop.src]->add_scheduled_chunk(
            flow.packet_flow, hop.chunk_source, hop.dst, hop.epoch, flow.id,
            hop.fragment_id, bytes, (uint32_t)(path.size() + 4),
            flow.next_packet_id, path, false);
    }

    void remove_pending_hop(uint32_t flow_id) {
        vector<uint32_t> still_pending;
        for (uint32_t pending : _pending_hops) {
            if (pending != flow_id) {
                still_pending.push_back(pending);
            }
        }
        _pending_hops.swap(still_pending);
    }

    void try_inject_ready_steps() {
        if (_schedule.hop_level) {
            try_inject_ready_hops_dag();
            return;
        }

        for (uint32_t node = 0; node < _node_states.size(); node++) {
            NodeState& state = _node_states[node];
            if (state.complete) {
                continue;
            }

            bool progressed = true;
            while (progressed) {
                progressed = false;
                vector<uint32_t> still_pending;
                for (uint32_t flow_id : state.pending) {
                    SmartTecclFlowState* flow = _flows[flow_id];
                    if (flow->delivered || flow->injected) {
                        continue;
                    }
                    if (step_is_ready(*flow) && fast_gate_open(*flow)) {
                        inject_step(*flow);
                    } else {
                        still_pending.push_back(flow_id);
                    }
                }
                state.pending = still_pending;
                if (!state.complete && state.pending.empty()) {
                    start_next_node_epoch(node);
                    progressed = !state.complete;
                }
            }
        }
    }

    void try_inject_ready_hops_dag() {
        vector<uint32_t> still_pending;
        for (uint32_t flow_id : _pending_hops) {
            SmartTecclHopFlowState* flow = _hop_flows[flow_id];
            if (flow->delivered || flow->injected) {
                continue;
            }
            if (hop_is_ready(*flow)) {
                inject_hop(*flow);
            } else {
                still_pending.push_back(flow_id);
            }
        }
        _pending_hops.swap(still_pending);
    }

    uint32_t pending_step_count() const {
        uint32_t count = 0;
        if (_schedule.hop_level) {
            return (uint32_t)_pending_hops.size();
        }
        for (const NodeState& state : _node_states) {
            count += state.pending.size();
        }
        return count;
    }

    uint32_t active_node_count() const {
        uint32_t count = 0;
        if (_schedule.hop_level) {
            return _in_flight_steps > 0 || !_pending_hops.empty() ? 1 : 0;
        }
        for (const NodeState& state : _node_states) {
            if (!state.complete) {
                count++;
            }
        }
        return count;
    }

    uint32_t local_epoch_slot_count() const {
        uint32_t count = 0;
        if (_schedule.hop_level) {
            set<pair<uint32_t, uint32_t> > slots;
            for (const SmartTecclHopFlowState* flow : _hop_flows) {
                slots.insert(make_pair(flow->step.src, flow->step.epoch));
            }
            return (uint32_t)slots.size();
        }
        for (const NodeState& state : _node_states) {
            count += state.epochs.size();
        }
        return count;
    }

    void initialize_hop_level_schedule() {
        uint32_t max_demand = 0;
        uint32_t max_fragment = 0;
        for (uint32_t i = 0; i < _schedule.hops.size(); i++) {
            const SmartTecclHopStep& hop = _schedule.hops[i];
            validate_hop(hop);
            if (hop.demand_id > max_demand) {
                max_demand = hop.demand_id;
            }
            if (hop.fragment_id > max_fragment) {
                max_fragment = hop.fragment_id;
            }
        }

        _demands.resize(max_demand + 1);
        _fragments.resize(max_fragment + 1);
        vector<set<uint32_t> > fragment_hops(max_fragment + 1);

        for (uint32_t i = 0; i < _schedule.hops.size(); i++) {
            const SmartTecclHopStep& hop = _schedule.hops[i];

            DemandState& demand = _demands[hop.demand_id];
            if (demand.id == UINT32_MAX) {
                demand.id = hop.demand_id;
                demand.chunk_source = hop.chunk_source;
                demand.chunk = hop.chunk;
                demand.final_dst = hop.final_dst;
                demand.expected_bytes = hop.demand_bytes;
                _expected_bytes += hop.demand_bytes;
            } else if (demand.chunk_source != hop.chunk_source
                       || demand.chunk != hop.chunk
                       || demand.final_dst != hop.final_dst
                       || demand.expected_bytes != hop.demand_bytes) {
                throw runtime_error("inconsistent TE-CCL hop demand metadata");
            }

            FragmentState& fragment = _fragments[hop.fragment_id];
            if (fragment.id == UINT32_MAX) {
                fragment.id = hop.fragment_id;
                fragment.demand_id = hop.demand_id;
                fragment.bytes = hop.fragment_bytes;
                fragment.current_node = hop.chunk_source;
            } else if (fragment.demand_id != hop.demand_id
                       || fragment.bytes != hop.fragment_bytes) {
                throw runtime_error("inconsistent TE-CCL hop fragment metadata");
            }
            fragment_hops[hop.fragment_id].insert(hop.hop_index);

            uint32_t flow_id = (uint32_t)_hop_flows.size();
            _hop_flows.push_back(new SmartTecclHopFlowState(flow_id, hop));
            _pending_hops.push_back(flow_id);
        }

        sort(_pending_hops.begin(), _pending_hops.end(),
             [this](uint32_t left, uint32_t right) {
                 const SmartTecclHopStep& left_hop = _hop_flows[left]->step;
                 const SmartTecclHopStep& right_hop = _hop_flows[right]->step;
                 if (left_hop.epoch != right_hop.epoch) {
                     return left_hop.epoch < right_hop.epoch;
                 }
                 return left < right;
             });

        for (uint32_t fragment_id = 0; fragment_id < _fragments.size();
             fragment_id++) {
            FragmentState& fragment = _fragments[fragment_id];
            if (fragment.id == UINT32_MAX) {
                throw runtime_error("TE-CCL hop schedule has missing fragment id");
            }
            const set<uint32_t>& hops = fragment_hops[fragment_id];
            for (uint32_t index = 0; index < hops.size(); index++) {
                if (hops.find(index) == hops.end()) {
                    throw runtime_error("TE-CCL hop schedule has non-contiguous hop_index");
                }
            }
            fragment.total_hops = (uint32_t)hops.size();
        }

        _hop_flow_by_fragment_hop.resize(_fragments.size());
        for (uint32_t i = 0; i < _hop_flows.size(); i++) {
            const SmartTecclHopStep& hop = _hop_flows[i]->step;
            _hop_flow_by_fragment_hop[hop.fragment_id].resize(
                _fragments[hop.fragment_id].total_hops, UINT32_MAX);
            _hop_flow_by_fragment_hop[hop.fragment_id][hop.hop_index] = i;
        }
        for (uint32_t fragment_id = 0; fragment_id < _hop_flow_by_fragment_hop.size();
             fragment_id++) {
            for (uint32_t hop_index = 0;
                 hop_index < _hop_flow_by_fragment_hop[fragment_id].size();
                 hop_index++) {
                if (_hop_flow_by_fragment_hop[fragment_id][hop_index] == UINT32_MAX) {
                    throw runtime_error("TE-CCL hop schedule missing fragment hop flow");
                }
            }
        }

        for (uint32_t demand_id = 0; demand_id < _demands.size();
             demand_id++) {
            if (_demands[demand_id].id == UINT32_MAX) {
                throw runtime_error("TE-CCL hop schedule has missing demand id");
            }
        }
    }

    void hop_packet_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                              uint32_t dst_node, mem_b bytes) {
        (void)chunk_id;
        if (app_flow_id >= _hop_flows.size()) {
            throw runtime_error("delivered unknown TE-CCL hop id");
        }

        SmartTecclHopFlowState* flow = _hop_flows[app_flow_id];
        const SmartTecclHopStep& hop = flow->step;
        if (dst_node != hop.dst) {
            throw runtime_error("TE-CCL hop delivered to wrong node");
        }
        if (flow->delivered) {
            throw runtime_error("TE-CCL hop delivered packet after completion");
        }
        if (flow->delivered_bytes + bytes > hop.bytes) {
            throw runtime_error("TE-CCL hop delivered too many bytes");
        }

        FragmentState& fragment = _fragments[hop.fragment_id];
        if (fragment.delivered) {
            throw runtime_error("TE-CCL fragment delivered packet after completion");
        }
        flow->delivered_bytes += bytes;

        if (hop.hop_index + 1 < fragment.total_hops) {
            uint32_t next_flow_id =
                _hop_flow_by_fragment_hop[hop.fragment_id][hop.hop_index + 1];
            SmartTecclHopFlowState* next_flow = _hop_flows[next_flow_id];
            if (next_flow->step.src != hop.dst) {
                throw runtime_error("TE-CCL next hop does not start at current dst");
            }
            if (!SmartNode::store_forward_enabled()) {
                inject_hop_packet(*next_flow, bytes);
            }
        } else {
            DemandState& demand = _demands[fragment.demand_id];
            if (hop.dst != demand.final_dst) {
                throw runtime_error("TE-CCL final hop did not reach demand dst");
            }
            demand.delivered_bytes += bytes;
            if (demand.delivered_bytes > demand.expected_bytes) {
                throw runtime_error("TE-CCL demand delivered too many bytes");
            }
            if (!demand.complete
                && demand.delivered_bytes == demand.expected_bytes) {
                demand.complete = true;
                _delivered_demands++;
                SmartNode::record_application_delivery(demand.expected_bytes);
            }
        }

        if (flow->delivered_bytes == hop.bytes) {
            flow->delivered = true;
            _delivered_steps++;
            if (_in_flight_steps == 0) {
                throw runtime_error("TE-CCL delivered hop with no in-flight step");
            }
            _in_flight_steps--;

            NodeState& state = _node_states[hop.src];
            if (state.in_flight > 0) {
                state.in_flight--;
            }
            state.delivered++;

            advance_completed_fragment_hops(hop.fragment_id);
        }

        try_inject_ready_steps();
    }

    void advance_completed_fragment_hops(uint32_t fragment_id) {
        FragmentState& fragment = _fragments[fragment_id];
        while (fragment.next_hop_index < fragment.total_hops) {
            uint32_t flow_id =
                _hop_flow_by_fragment_hop[fragment_id][fragment.next_hop_index];
            SmartTecclHopFlowState* flow = _hop_flows[flow_id];
            if (!flow->delivered) {
                break;
            }
            fragment.current_node = flow->step.dst;
            fragment.next_hop_index++;
        }
        if (fragment.next_hop_index >= fragment.total_hops) {
            fragment.delivered = true;
        }
    }

    void validate_hop(const SmartTecclHopStep& hop) const {
        uint32_t node_count = (uint32_t)_graph.nodes.size();
        if (hop.chunk_source >= node_count || hop.final_dst >= node_count
            || hop.src >= node_count || hop.dst >= node_count) {
            throw runtime_error("TE-CCL hop node outside graph");
        }
        if (hop.bytes <= 0 || hop.fragment_bytes <= 0
            || hop.demand_bytes <= 0) {
            throw runtime_error("TE-CCL hop bytes must be positive");
        }
        if (hop.bytes != hop.fragment_bytes) {
            throw runtime_error("TE-CCL hop bytes must equal fragment bytes");
        }
        physical_path(hop.src, hop.dst);
    }

    void validate_step(const SmartTecclScheduleStep& step) const {
        uint32_t node_count = (uint32_t)_graph.nodes.size();
        if (step.chunk_source >= node_count || step.src >= node_count
            || step.dst >= node_count) {
            throw runtime_error("TE-CCL schedule step node outside graph");
        }
        if (step.path.size() < 2 || step.path.front() != step.src
            || step.path.back() != step.dst) {
            throw runtime_error("TE-CCL schedule step has invalid path endpoints");
        }
        for (uint32_t node : step.path) {
            if (node >= node_count) {
                throw runtime_error("TE-CCL schedule path node outside graph");
            }
        }
        set<uint32_t> seen_path_nodes;
        for (uint32_t node : step.path) {
            if (seen_path_nodes.find(node) != seen_path_nodes.end()) {
                throw runtime_error("TE-CCL schedule path repeats node "
                                    + ntoa(node)
                                    + "; explicit paths must be simple");
            }
            seen_path_nodes.insert(node);
        }
        for (uint32_t i = 0; i + 1 < step.path.size(); i++) {
            const vector<uint32_t>& neighbors = _graph.adjacency[step.path[i]];
            if (find(neighbors.begin(), neighbors.end(), step.path[i + 1])
                == neighbors.end()) {
                throw runtime_error("TE-CCL schedule path uses non-edge "
                                    + ntoa(step.path[i]) + "->"
                                    + ntoa(step.path[i + 1]));
            }
        }
    }

    uint64_t chunk_location_key(uint32_t source, uint32_t chunk,
                                uint32_t node) const {
        uint64_t key = source;
        key = key * _schedule.node_count + node;
        key = (key << 32) | chunk;
        return key;
    }

    vector<uint32_t> physical_path(uint32_t src, uint32_t dst) const {
        uint32_t node_count = (uint32_t)_graph.nodes.size();
        if (src >= node_count || dst >= node_count) {
            throw runtime_error("TE-CCL physical path endpoint outside graph");
        }
        if (src == dst) {
            vector<uint32_t> path;
            path.push_back(src);
            return path;
        }

        vector<int> previous(node_count, -1);
        queue<uint32_t> pending;
        previous[src] = (int)src;
        pending.push(src);
        while (!pending.empty() && previous[dst] < 0) {
            uint32_t node = pending.front();
            pending.pop();
            const vector<uint32_t>& neighbors = _graph.adjacency[node];
            for (uint32_t i = 0; i < neighbors.size(); i++) {
                uint32_t next = neighbors[i];
                if (next >= node_count || previous[next] >= 0) {
                    continue;
                }
                previous[next] = (int)node;
                pending.push(next);
                if (next == dst) {
                    break;
                }
            }
        }
        if (previous[dst] < 0) {
            throw runtime_error("TE-CCL physical path is disconnected");
        }

        vector<uint32_t> reversed;
        for (uint32_t node = dst; node != src;
             node = (uint32_t)previous[node]) {
            reversed.push_back(node);
        }
        reversed.push_back(src);

        vector<uint32_t> path;
        for (vector<uint32_t>::reverse_iterator it = reversed.rbegin();
             it != reversed.rend(); ++it) {
            path.push_back(*it);
        }
        return path;
    }

    SmartGraph& _graph;
    SmartTecclSchedule _schedule;
    vector<SmartTecclFlowState*> _flows;
    vector<SmartTecclHopFlowState*> _hop_flows;
    vector<NodeState> _node_states;
    vector<DemandState> _demands;
    vector<FragmentState> _fragments;
    vector<vector<uint32_t> > _hop_flow_by_fragment_hop;
    vector<uint32_t> _pending_hops;
    set<uint64_t> _available_chunks;
    uint64_t _expected_bytes;
    uint64_t _delivered_steps;
    uint64_t _delivered_demands;
    uint32_t _in_flight_steps;

    // FAST stage-sync state (enable_fast_sync); inert when mode is NONE.
    FastSyncMode _fast_sync_mode = FAST_SYNC_NONE;
    vector<FastStepTag> _fast_tags;
    vector<uint32_t> _fs_exp_bal_in, _fs_del_bal_in;
    vector<uint32_t> _fs_exp_intra_out, _fs_del_intra_out;
    vector<vector<uint32_t> > _fs_exp_inter_in, _fs_del_inter_in;
    vector<vector<uint32_t> > _fs_exp_inter_out, _fs_del_inter_out;
    vector<vector<uint32_t> > _fs_exp_disp_out, _fs_del_disp_out;
    vector<uint32_t> _fs_exp_stage_inter, _fs_del_stage_inter;
    vector<uint32_t> _fs_exp_stage_disp, _fs_del_stage_disp;
    uint32_t _fs_exp_bal_total = 0, _fs_del_bal_total = 0;
    uint32_t _fs_exp_intra_total = 0, _fs_del_intra_total = 0;
};


#endif
#endif
