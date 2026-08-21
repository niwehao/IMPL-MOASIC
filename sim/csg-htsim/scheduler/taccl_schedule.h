// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef TACCL_SCHEDULE_H
#define TACCL_SCHEDULE_H

#include "config.h"
#include "common.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SmartTacclChunk {
    uint32_t addr;
    uint32_t source;
    uint32_t final_dst;
    mem_b bytes;
};

struct SmartTacclDemand {
    uint32_t addr;
    uint32_t source;
    uint32_t final_dst;
    mem_b bytes;
};

struct SmartTacclScheduleStep {
    uint32_t step_id;
    uint32_t epoch;
    uint32_t addr;
    uint32_t src;
    uint32_t dst;
    mem_b bytes;
    vector<uint32_t> path;
};

struct SmartTacclSchedule {
    SmartTacclSchedule()
        : node_count(0), chunk_size_bytes(0), epoch_duration(1.0),
          collective("alltoall") {
    }

    uint32_t node_count;
    mem_b chunk_size_bytes;
    double epoch_duration;
    string collective;
    vector<SmartTacclChunk> chunks;
    vector<SmartTacclDemand> demands;
    vector<SmartTacclScheduleStep> steps;
};

static SmartTacclSchedule load_taccl_schedule(const string& filename) {
    ifstream input(filename.c_str());
    if (!input) {
        throw runtime_error("cannot open TACCL schedule file: " + filename);
    }

    SmartTacclSchedule schedule;
    string line;
    uint32_t line_number = 0;
    while (getline(input, line)) {
        line_number++;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        istringstream stream(line);
        string kind;
        stream >> kind;
        if (kind == "node_count") {
            stream >> schedule.node_count;
        } else if (kind == "chunk_size_bytes") {
            stream >> schedule.chunk_size_bytes;
        } else if (kind == "epoch_duration") {
            stream >> schedule.epoch_duration;
        } else if (kind == "collective") {
            stream >> schedule.collective;
        } else if (kind == "mode") {
            string mode;
            stream >> mode;
            if (mode != "taccl_step") {
                throw runtime_error("unknown TACCL schedule mode '" + mode
                                    + "' at line " + ntoa(line_number));
            }
        } else if (kind == "chunk") {
            SmartTacclChunk chunk;
            stream >> chunk.addr >> chunk.source >> chunk.final_dst
                   >> chunk.bytes;
            if (!stream) {
                throw runtime_error("bad TACCL chunk at line "
                                    + ntoa(line_number));
            }
            schedule.chunks.push_back(chunk);
        } else if (kind == "demand") {
            SmartTacclDemand demand;
            stream >> demand.addr >> demand.source >> demand.final_dst
                   >> demand.bytes;
            if (!stream) {
                throw runtime_error("bad TACCL demand at line "
                                    + ntoa(line_number));
            }
            schedule.demands.push_back(demand);
        } else if (kind == "step") {
            SmartTacclScheduleStep step;
            string path_keyword;
            stream >> step.step_id >> step.epoch >> step.addr >> step.src
                   >> step.dst >> step.bytes >> path_keyword;
            if (!stream || path_keyword != "path") {
                throw runtime_error("bad TACCL step at line "
                                    + ntoa(line_number));
            }

            uint32_t node = 0;
            while (stream >> node) {
                step.path.push_back(node);
            }
            if (step.path.size() < 2) {
                throw runtime_error("TACCL step path must have at least two nodes at line "
                                    + ntoa(line_number));
            }
            if (step.path.front() != step.src || step.path.back() != step.dst) {
                throw runtime_error("TACCL step path endpoints do not match src/dst at line "
                                    + ntoa(line_number));
            }
            set<uint32_t> seen;
            for (uint32_t path_node : step.path) {
                if (seen.find(path_node) != seen.end()) {
                    throw runtime_error("TACCL step path repeats node "
                                        + ntoa(path_node) + " at line "
                                        + ntoa(line_number));
                }
                seen.insert(path_node);
            }
            schedule.steps.push_back(step);
        } else {
            throw runtime_error("unknown TACCL schedule line kind '" + kind
                                + "' at line " + ntoa(line_number));
        }
    }

    if (schedule.node_count == 0) {
        throw runtime_error("TACCL schedule missing node_count");
    }
    if (schedule.chunk_size_bytes <= 0) {
        throw runtime_error("TACCL schedule missing chunk_size_bytes");
    }
    if (schedule.collective != "alltoall") {
        throw runtime_error("TACCL schedule currently supports collective alltoall only");
    }
    if (schedule.chunks.empty()) {
        throw runtime_error("TACCL schedule has no chunk metadata");
    }
    if (schedule.demands.empty()) {
        throw runtime_error("TACCL schedule has no app-level demands");
    }
    if (schedule.steps.empty()) {
        throw runtime_error("TACCL schedule has no transport steps");
    }
    return schedule;
}

class SmartTacclScheduleScheduler {
public:
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<SmartTacclScheduleScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk
            <SmartTacclScheduleScheduler>(node, chunk);
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
        SchedulerCommon::do_next_event<SmartTacclScheduleScheduler>(node);
    }

    // Min-epoch priority: TACCL step/t order encodes relay pipeline criticality. Within an epoch,
    // send_packet_from_chunk's requeue-to-back still round-robins equal-epoch
    // chunks. NOTE (2026-07-08 experiment): replacing this with pure pool-order
    // round-robin fixed the clos_32 bottleneck-starvation anomaly for TE-CCL
    // (9464->8074us) but regressed relay pipelines badly elsewhere
    // (TACCL dgx_rail +59%), so per-solver semantics were kept; SyCCL uses
    // round-robin. See Note/experiment_summary.md 2.4.
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
#ifndef TACCL_SCHEDULE_RUNTIME_H
#define TACCL_SCHEDULE_RUNTIME_H

struct SmartTacclTransportFlow {
    SmartTacclTransportFlow(uint32_t flow_id,
                            const SmartTacclScheduleStep& flow_step)
        : id(flow_id), step(flow_step), next_packet_id(1), packet_flow(NULL),
          injected(false), delivered(false) {
    }

    uint32_t id;
    SmartTacclScheduleStep step;
    packetid_t next_packet_id;
    PacketFlow packet_flow;
    bool injected;
    bool delivered;
};

class SmartTacclScheduleRuntime : public SmartDeliveryObserver {
public:
    SmartTacclScheduleRuntime(SmartGraph& graph,
                              const SmartTacclSchedule& schedule)
        : _graph(graph), _schedule(schedule), _expected_bytes(0),
          _delivered_steps(0), _delivered_demands(0), _in_flight_steps(0) {
        if (_schedule.node_count > graph.nodes.size()) {
            throw runtime_error("TACCL schedule node_count exceeds graph nodes");
        }

        _node_states.resize(_schedule.node_count);
        initialize_demands();
        initialize_steps();
    }

    const vector<SmartTacclTransportFlow*>& flows() const {
        return _flows;
    }

    uint64_t expected_chunks() const {
        return _demands.size();
    }

    uint64_t expected_bytes() const {
        return _expected_bytes;
    }

    uint64_t delivered_steps() const {
        return _delivered_steps;
    }

    uint64_t delivered_demands() const {
        return _delivered_demands;
    }

    uint32_t in_flight_steps() const {
        return _in_flight_steps;
    }

    void start() {
        for (uint32_t node = 0; node < _node_states.size(); node++) {
            start_next_node_epoch(node);
        }
        try_inject_ready_steps();
    }

    void packet_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                          uint32_t dst_node, mem_b bytes) override {
        (void)chunk_id;
        if (app_flow_id >= _flows.size()) {
            throw runtime_error("delivered unknown TACCL transport id");
        }
        SmartTacclTransportFlow* flow = _flows[app_flow_id];
        const SmartTacclScheduleStep& step = flow->step;
        if (dst_node != step.dst) {
            throw runtime_error("TACCL step delivered packet to wrong node");
        }

        map<uint32_t, DemandState>::iterator demand_it =
            _demands_by_addr.find(step.addr);
        if (demand_it == _demands_by_addr.end()) {
            return;
        }
        DemandState& demand = demand_it->second;
        if (step.dst != demand.final_dst) {
            return;
        }
        if (demand.delivered) {
            throw runtime_error("TACCL demand received packet after completion");
        }

        demand.delivered_bytes += bytes;
        if (demand.delivered_bytes > demand.expected_bytes) {
            throw runtime_error("TACCL demand delivered too many bytes");
        }
        if (demand.delivered_bytes == demand.expected_bytes) {
            demand.delivered = true;
            _delivered_demands++;
            SmartNode::record_application_delivery(demand.expected_bytes);
        }
    }

    void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                         uint32_t dst_node, mem_b bytes) override {
        (void)chunk_id;
        (void)bytes;
        if (app_flow_id >= _flows.size()) {
            throw runtime_error("delivered unknown TACCL transport id");
        }

        SmartTacclTransportFlow* flow = _flows[app_flow_id];
        const SmartTacclScheduleStep& step = flow->step;
        if (dst_node != step.dst) {
            throw runtime_error("TACCL step delivered to wrong node");
        }
        if (flow->delivered) {
            throw runtime_error("TACCL step delivered more than once");
        }

        flow->delivered = true;
        _available_chunks.insert(chunk_location_key(step.addr, step.dst));
        _delivered_steps++;
        if (_in_flight_steps == 0) {
            throw runtime_error("TACCL delivered step with no in-flight step");
        }
        _in_flight_steps--;
        _in_flight_flow_ids.erase(app_flow_id);

        NodeState& state = _node_states[step.src];
        if (state.in_flight == 0) {
            throw runtime_error("TACCL delivered step with no local in-flight step");
        }
        state.in_flight--;
        state.delivered++;

        try_inject_ready_steps();
    }

    void print_schedule_report() const {
        cout << "taccl_schedule_report"
             << " app_demands " << _demands.size()
             << " delivered_demands " << _delivered_demands
             << " transport_steps " << _flows.size()
             << " delivered_steps " << _delivered_steps
             << " local_epoch_slots " << local_epoch_slot_count()
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
            : addr(UINT32_MAX), source(UINT32_MAX), final_dst(UINT32_MAX),
              expected_bytes(0), delivered_bytes(0), delivered(false) {
        }

        uint32_t addr;
        uint32_t source;
        uint32_t final_dst;
        mem_b expected_bytes;
        mem_b delivered_bytes;
        bool delivered;
    };

    struct ChunkState {
        ChunkState()
            : addr(UINT32_MAX), source(UINT32_MAX), final_dst(UINT32_MAX),
              bytes(0) {
        }

        uint32_t addr;
        uint32_t source;
        uint32_t final_dst;
        mem_b bytes;
    };

    void initialize_demands() {
        for (const SmartTacclChunk& chunk : _schedule.chunks) {
            if (chunk.source >= _schedule.node_count
                || chunk.final_dst >= _schedule.node_count) {
                throw runtime_error("TACCL chunk endpoint outside schedule node_count");
            }
            if (_chunks_by_addr.find(chunk.addr) != _chunks_by_addr.end()) {
                throw runtime_error("TACCL schedule has duplicate chunk addr "
                                    + ntoa(chunk.addr));
            }
            ChunkState state;
            state.addr = chunk.addr;
            state.source = chunk.source;
            state.final_dst = chunk.final_dst;
            state.bytes = chunk.bytes;
            _chunks_by_addr[chunk.addr] = state;
            _available_chunks.insert(chunk_location_key(chunk.addr,
                                                        chunk.source));
        }

        for (const SmartTacclDemand& demand : _schedule.demands) {
            if (demand.source >= _schedule.node_count
                || demand.final_dst >= _schedule.node_count) {
                throw runtime_error("TACCL demand endpoint outside schedule node_count");
            }
            if (demand.source == demand.final_dst) {
                throw runtime_error("TACCL app-level demand should exclude self-copy chunks");
            }
            if (_demands_by_addr.find(demand.addr) != _demands_by_addr.end()) {
                throw runtime_error("TACCL schedule has duplicate demand addr "
                                    + ntoa(demand.addr));
            }
            map<uint32_t, ChunkState>::const_iterator chunk_it =
                _chunks_by_addr.find(demand.addr);
            if (chunk_it == _chunks_by_addr.end()) {
                throw runtime_error("TACCL demand references unknown chunk addr "
                                    + ntoa(demand.addr));
            }
            if (chunk_it->second.source != demand.source
                || chunk_it->second.final_dst != demand.final_dst) {
                throw runtime_error("TACCL demand does not match chunk metadata for addr "
                                    + ntoa(demand.addr));
            }

            DemandState state;
            state.addr = demand.addr;
            state.source = demand.source;
            state.final_dst = demand.final_dst;
            state.expected_bytes = demand.bytes;
            _demands_by_addr[demand.addr] = state;
            _demands.push_back(demand.addr);
            _expected_bytes += (uint64_t)demand.bytes;
        }
    }

    void initialize_steps() {
        set<uint32_t> seen_step_ids;
        for (uint32_t i = 0; i < _schedule.steps.size(); i++) {
            const SmartTacclScheduleStep& step = _schedule.steps[i];
            if (seen_step_ids.find(step.step_id) != seen_step_ids.end()) {
                throw runtime_error("TACCL schedule has duplicate step_id "
                                    + ntoa(step.step_id));
            }
            seen_step_ids.insert(step.step_id);
            validate_step(step);
            _flows.push_back(new SmartTacclTransportFlow(i, step));
            _node_states[step.src].steps_by_epoch[step.epoch].push_back(i);
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

    // TACCL steps are a discretization of continuous solver send times; the
    // only hard inter-step constraint is data availability, which
    // step_is_ready() enforces. A node therefore advances to its next epoch
    // as soon as the current epoch's sends are all *issued* (injected), not
    // end-to-end delivered — the solver never asked for a delivery barrier.
    // Counters are cumulative across epochs.
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

    bool step_is_ready(const SmartTacclTransportFlow& flow) const {
        return _available_chunks.find(chunk_location_key(
            flow.step.addr, flow.step.src)) != _available_chunks.end();
    }

    void inject_step(SmartTacclTransportFlow& flow) {
        if (flow.injected) {
            throw runtime_error("TACCL step injected more than once");
        }
        if (!step_is_ready(flow)) {
            throw runtime_error("TACCL tried to inject a step before dependency was ready");
        }

        const SmartTacclScheduleStep& step = flow.step;
        flow.injected = true;
        _in_flight_steps++;
        _in_flight_flow_ids.insert(flow.id);
        _node_states[step.src].in_flight++;
        _graph.nodes[step.src]->add_scheduled_chunk(
            flow.packet_flow, step.src, step.dst, step.epoch, flow.id,
            step.addr, step.bytes, (uint32_t)(step.path.size() + 4),
            flow.next_packet_id, step.path, false);
    }

    void try_inject_ready_steps() {
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
                    SmartTacclTransportFlow* flow = _flows[flow_id];
                    if (flow->delivered || flow->injected) {
                        continue;
                    }
                    if (step_is_ready(*flow)) {
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

        if (pending_step_count() > 0 && !pending_can_eventually_be_satisfied()) {
            SmartTacclTransportFlow* blocked = first_pending_flow();
            const SmartTacclScheduleStep& step = blocked->step;
            throw runtime_error("TACCL schedule deadlock at node "
                                + ntoa(step.src) + " is waiting for addr "
                                + ntoa(step.addr)
                                + " in local epoch " + ntoa(step.epoch));
        }
    }

    uint32_t pending_step_count() const {
        uint32_t count = 0;
        for (const NodeState& state : _node_states) {
            count += state.pending.size();
        }
        return count;
    }

    uint32_t active_node_count() const {
        uint32_t count = 0;
        for (const NodeState& state : _node_states) {
            if (!state.complete) {
                count++;
            }
        }
        return count;
    }

    uint32_t local_epoch_slot_count() const {
        uint32_t count = 0;
        for (const NodeState& state : _node_states) {
            count += state.epochs.size();
        }
        return count;
    }

    SmartTacclTransportFlow* first_pending_flow() const {
        for (const NodeState& state : _node_states) {
            if (!state.pending.empty()) {
                return _flows[state.pending.front()];
            }
        }
        throw runtime_error("internal error: no pending TACCL flow");
    }

    bool pending_can_eventually_be_satisfied() const {
        if (_in_flight_flow_ids.empty()) {
            return false;
        }

        set<uint64_t> future_available = _available_chunks;
        for (uint32_t flow_id : _in_flight_flow_ids) {
            const SmartTacclScheduleStep& step = _flows[flow_id]->step;
            future_available.insert(chunk_location_key(step.addr, step.dst));
        }

        vector<uint32_t> simulated_epoch_index(_node_states.size(), 0);
        for (uint32_t node = 0; node < _node_states.size(); node++) {
            const NodeState& state = _node_states[node];
            if (state.complete) {
                simulated_epoch_index[node] = state.epochs.size();
            } else if (state.epoch_index == 0) {
                simulated_epoch_index[node] = 0;
            } else {
                simulated_epoch_index[node] = state.epoch_index - 1;
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t node = 0; node < _node_states.size(); node++) {
                const NodeState& state = _node_states[node];
                uint32_t index = simulated_epoch_index[node];
                if (index >= state.epochs.size()) {
                    continue;
                }

                uint32_t epoch = state.epochs[index];
                map<uint32_t, vector<uint32_t> >::const_iterator it =
                    state.steps_by_epoch.find(epoch);
                if (it == state.steps_by_epoch.end()) {
                    simulated_epoch_index[node]++;
                    changed = true;
                    continue;
                }

                bool epoch_can_complete = true;
                for (uint32_t flow_id : it->second) {
                    SmartTacclTransportFlow* flow = _flows[flow_id];
                    if (flow->delivered || flow->injected) {
                        continue;
                    }
                    uint64_t needed =
                        chunk_location_key(flow->step.addr, flow->step.src);
                    if (future_available.find(needed) == future_available.end()) {
                        epoch_can_complete = false;
                        break;
                    }
                }
                if (!epoch_can_complete) {
                    continue;
                }

                for (uint32_t flow_id : it->second) {
                    SmartTacclTransportFlow* flow = _flows[flow_id];
                    if (flow->delivered) {
                        continue;
                    }
                    future_available.insert(chunk_location_key(
                        flow->step.addr, flow->step.dst));
                }
                simulated_epoch_index[node]++;
                changed = true;
            }
        }

        for (const NodeState& state : _node_states) {
            for (uint32_t flow_id : state.pending) {
                uint64_t needed =
                    chunk_location_key(_flows[flow_id]->step.addr,
                                       _flows[flow_id]->step.src);
                if (future_available.find(needed) != future_available.end()) {
                    return true;
                }
            }
        }
        return false;
    }

    void validate_step(const SmartTacclScheduleStep& step) const {
        if (step.src >= _schedule.node_count || step.dst >= _schedule.node_count) {
            throw runtime_error("TACCL step endpoint outside schedule node_count");
        }
        if (_chunks_by_addr.find(step.addr) == _chunks_by_addr.end()) {
            throw runtime_error("TACCL step references unknown chunk addr "
                                + ntoa(step.addr));
        }
        if (step.path.size() < 2 || step.path.front() != step.src
            || step.path.back() != step.dst) {
            throw runtime_error("TACCL step has invalid path endpoints");
        }
        for (uint32_t node : step.path) {
            if (node >= _graph.nodes.size()) {
                throw runtime_error("TACCL step path node outside graph");
            }
        }
        for (uint32_t i = 0; i + 1 < step.path.size(); i++) {
            const vector<uint32_t>& neighbors = _graph.adjacency[step.path[i]];
            if (find(neighbors.begin(), neighbors.end(), step.path[i + 1])
                == neighbors.end()) {
                throw runtime_error("TACCL step path uses non-edge "
                                    + ntoa(step.path[i]) + "->"
                                    + ntoa(step.path[i + 1]));
            }
        }
    }

    static uint64_t chunk_location_key(uint32_t addr, uint32_t node) {
        return (((uint64_t)addr) << 32) | node;
    }

    SmartGraph& _graph;
    SmartTacclSchedule _schedule;
    vector<NodeState> _node_states;
    vector<SmartTacclTransportFlow*> _flows;
    vector<uint32_t> _demands;
    map<uint32_t, ChunkState> _chunks_by_addr;
    map<uint32_t, DemandState> _demands_by_addr;
    set<uint64_t> _available_chunks;
    set<uint32_t> _in_flight_flow_ids;
    uint64_t _expected_bytes;
    uint64_t _delivered_steps;
    uint64_t _delivered_demands;
    uint32_t _in_flight_steps;
};

#endif
#endif
