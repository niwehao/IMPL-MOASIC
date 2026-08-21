// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef SYCCL_SCHEDULE_H
#define SYCCL_SCHEDULE_H

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

struct SmartSycclScheduleStep {
    uint32_t epoch;
    uint32_t chunk_source;
    uint32_t src;
    uint32_t dst;
    uint32_t chunk;
    mem_b bytes;
};

struct SmartSycclScheduleChunk {
    uint32_t chunk_source;
    uint32_t chunk;
    mem_b bytes;
    vector<uint32_t> init_nodes;
};

struct SmartSycclScheduleDemand {
    uint32_t src;
    uint32_t dst;
    uint32_t chunk;
    mem_b bytes;
};

struct SmartSycclScheduleEvent {
    uint32_t id;
    uint32_t epoch;
    uint32_t chunk_source;
    uint32_t chunk;
    uint32_t src;
    uint32_t dst;
    mem_b bytes;
    double amount;
    bool copy;
    bool reduce;
    int layer_used;
    vector<uint32_t> path;
};

struct SmartSycclSchedule {
    SmartSycclSchedule() : node_count(0), chunk_size_bytes(0),
                           epoch_duration(1.0), event_mode(false),
                           dag_mode(false), solver_predicted_us(-1.0) {
    }

    uint32_t node_count;
    mem_b chunk_size_bytes;
    double epoch_duration;
    bool event_mode;
    bool dag_mode;
    double solver_predicted_us;
    vector<SmartSycclScheduleChunk> chunks;
    vector<SmartSycclScheduleDemand> demands;
    vector<SmartSycclScheduleEvent> events;
    vector<SmartSycclScheduleStep> steps;
};

static SmartSycclSchedule load_syccl_schedule(const string& filename) {
    ifstream input(filename.c_str());
    if (!input) {
        throw runtime_error("cannot open SyCCL schedule file: " + filename);
    }

    SmartSycclSchedule schedule;
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
        } else if (kind == "mode") {
            string mode;
            stream >> mode;
            if (mode == "syccl_event") {
                schedule.event_mode = true;
            } else if (mode == "syccl_event_dag") {
                schedule.event_mode = true;
                schedule.dag_mode = true;
            } else {
                throw runtime_error("unknown SyCCL schedule mode '"
                                    + mode + "' at line "
                                    + ntoa(line_number));
            }
        } else if (kind == "solver_predicted_us") {
            stream >> schedule.solver_predicted_us;
        } else if (kind == "chunk") {
            SmartSycclScheduleChunk chunk;
            stream >> chunk.chunk_source >> chunk.chunk >> chunk.bytes;
            if (!stream) {
                throw runtime_error("bad SyCCL chunk line at line "
                                    + ntoa(line_number));
            }
            string init_keyword;
            if (stream >> init_keyword) {
                if (init_keyword != "init") {
                    throw runtime_error("bad SyCCL chunk init keyword at line "
                                        + ntoa(line_number));
                }
                uint32_t init_node = 0;
                while (stream >> init_node) {
                    chunk.init_nodes.push_back(init_node);
                }
            }
            if (chunk.init_nodes.empty()) {
                chunk.init_nodes.push_back(chunk.chunk_source);
            }
            schedule.event_mode = true;
            schedule.chunks.push_back(chunk);
        } else if (kind == "demand") {
            SmartSycclScheduleDemand demand;
            stream >> demand.src >> demand.dst >> demand.chunk >> demand.bytes;
            if (!stream) {
                throw runtime_error("bad SyCCL demand line at line "
                                    + ntoa(line_number));
            }
            schedule.event_mode = true;
            schedule.demands.push_back(demand);
        } else if (kind == "event") {
            SmartSycclScheduleEvent event;
            uint32_t copy = 0;
            uint32_t reduce = 0;
            stream >> event.id >> event.epoch >> event.chunk_source
                   >> event.chunk >> event.src >> event.dst >> event.bytes
                   >> event.amount >> copy >> reduce >> event.layer_used;
            if (!stream) {
                throw runtime_error("bad SyCCL event line at line "
                                    + ntoa(line_number));
            }
            event.copy = copy != 0;
            event.reduce = reduce != 0;
            string path_keyword;
            if (stream >> path_keyword) {
                if (path_keyword != "path") {
                    throw runtime_error("bad SyCCL event path keyword at line "
                                        + ntoa(line_number));
                }
                uint32_t path_node = 0;
                while (stream >> path_node) {
                    event.path.push_back(path_node);
                }
                if (event.path.size() < 2) {
                    throw runtime_error("SyCCL event path must have at least two nodes at line "
                                        + ntoa(line_number));
                }
                if (event.path.front() != event.src
                    || event.path.back() != event.dst) {
                    throw runtime_error("SyCCL event path endpoints do not match src/dst at line "
                                        + ntoa(line_number));
                }
            }
            schedule.event_mode = true;
            schedule.events.push_back(event);
        } else if (kind == "step") {
            SmartSycclScheduleStep step;
            stream >> step.epoch >> step.chunk_source >> step.src >> step.dst
                   >> step.chunk >> step.bytes;
            if (!stream) {
                throw runtime_error("bad SyCCL schedule step at line "
                                    + ntoa(line_number));
            }
            schedule.steps.push_back(step);
        } else {
            throw runtime_error("unknown SyCCL schedule line kind '" + kind
                                + "' at line " + ntoa(line_number));
        }
    }

    if (schedule.node_count == 0) {
        throw runtime_error("SyCCL schedule missing node_count");
    }
    if (schedule.chunk_size_bytes <= 0) {
        throw runtime_error("SyCCL schedule missing chunk_size_bytes");
    }
    if (schedule.event_mode) {
        if (schedule.events.empty()) {
            throw runtime_error("SyCCL event schedule has no events");
        }
        if (schedule.demands.empty()) {
            throw runtime_error("SyCCL event schedule has no demands");
        }
    } else if (schedule.steps.empty()) {
        throw runtime_error("SyCCL schedule has no steps");
    }
    return schedule;
}

class SmartSycclScheduleScheduler {
public:
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<SmartSycclScheduleScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk
            <SmartSycclScheduleScheduler>(node, chunk);
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
        SchedulerCommon::do_next_event<SmartSycclScheduleScheduler>(node);
    }

    // Pick the first sendable chunk in pool order (packet-level round-robin
    // via send_packet_from_chunk's requeue-to-back); injection order remains
    // epoch-sorted. See teccl_schedule.h select_chunk_to_send for rationale.
    static SmartNode::ChunkIter
    select_chunk_to_send(SmartNode* node, Route*& selected_route) {
        for (SmartNode::ChunkIter it =
                 node->_chunk_pool.begin();
             it != node->_chunk_pool.end(); ++it) {
            Route* route = sendable_route_for_chunk(node, *it);
            if (!route) {
                continue;
            }
            selected_route = route;
            return it;
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
        if (!chunk.meta->path.empty()) {
            uint32_t next = select_next_hop_for_chunk(node, chunk);
            chunk.selected_next_hop = next;
            chunk.selected_split_class = 0;
            return route_for_next_hop(node, next);
        }

        const SmartNode::ForwardingChoices* choices =
            node->forwarding_choices_to(chunk.meta->dst);
        if (!choices) {
            uint32_t next = select_next_hop_for_chunk(node, chunk);
            chunk.selected_next_hop = next;
            return route_for_next_hop(node, next);
        }

        uint32_t total_weight = split_weight_total();
        if (total_weight == 0) {
            uint32_t next = select_next_hop_for_chunk(node, chunk);
            chunk.selected_next_hop = next;
            return route_for_next_hop(node, next);
        }

        return SchedulerCommon::route_for_choices_on_link(node, chunk,
                                                          choices);
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
        if (!path.empty()) {
            for (uint32_t i = 0; i + 1 < path.size(); i++) {
                if (path[i] == node->_id) {
                    return path[i + 1];
                }
            }
            return UINT32_MAX;
        }
        return choose_next_hop_to(node, chunk.meta->dst);
    }

    static uint32_t choose_next_hop_to(SmartNode* node, uint32_t dst) {
        map<uint32_t, SmartNode::ForwardingChoices>::const_iterator it =
            node->_forwarding_table.find(dst);
        if (it != node->_forwarding_table.end()) {
            const SmartNode::ForwardingChoices& choices = it->second;
            if (!choices.minus_one_hops.empty()) {
                return choices.minus_one_hops.front();
            }
            if (!choices.zero_hops.empty()) {
                return choices.zero_hops.front();
            }
            if (!choices.plus_one_hops.empty()) {
                return choices.plus_one_hops.front();
            }
        }
        return UINT32_MAX;
    }

    static uint32_t split_class_for_bucket(uint32_t bucket) {
        return SchedulerCommon::split_class_for_bucket(bucket);
    }

    static uint32_t split_weight_total() {
        return SchedulerCommon::split_weight_total();
    }

    static void record_split_packet(uint32_t split_class) {
        SchedulerCommon::record_split_packet(split_class);
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
#ifndef SYCCL_SCHEDULE_RUNTIME_H
#define SYCCL_SCHEDULE_RUNTIME_H

struct SmartSycclFlowState {
    SmartSycclFlowState(uint32_t flow_id,
                        const SmartSycclScheduleEvent& flow_event)
        : id(flow_id), event(flow_event), next_packet_id(1), packet_flow(NULL),
          injected(false), delivered(false), deps_remaining(0) {
    }

    uint32_t id;
    SmartSycclScheduleEvent event;
    packetid_t next_packet_id;
    PacketFlow packet_flow;
    bool injected;
    bool delivered;
    uint32_t deps_remaining;
    vector<uint32_t> dependents;
};

class SmartSycclScheduleRuntime : public SmartDeliveryObserver {
public:
    SmartSycclScheduleRuntime(SmartGraph& graph,
                              const SmartSycclSchedule& schedule)
        : _graph(graph), _schedule(schedule), _expected_bytes(0),
          _delivered_steps(0), _delivered_demands(0),
          _in_flight_steps(0) {
        if (_schedule.node_count > graph.nodes.size()) {
            throw runtime_error("SyCCL schedule node_count exceeds graph nodes");
        }

        _node_states.resize(_schedule.node_count);
        if (_schedule.event_mode) {
            initialize_event_schedule();
        } else {
            initialize_legacy_step_schedule();
        }
        if (_schedule.dag_mode) {
            build_event_dag_dependencies();
            for (uint32_t flow_id = 0; flow_id < _flows.size(); flow_id++) {
                _dag_pending_flow_ids.insert(flow_id);
            }
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

    const vector<SmartSycclFlowState*>& flows() const {
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

    uint32_t in_flight_steps() const {
        return _in_flight_steps;
    }

    void start() {
        if (!_schedule.dag_mode) {
            for (uint32_t node = 0; node < _node_states.size(); node++) {
                start_next_node_epoch(node);
            }
        }
        try_inject_ready_steps();
    }

    void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                         uint32_t dst_node, mem_b bytes) override {
        (void)chunk_id;
        (void)bytes;
        if (app_flow_id >= _flows.size()) {
            throw runtime_error("delivered unknown SyCCL flow id");
        }

        SmartSycclFlowState* flow = _flows[app_flow_id];
        const SmartSycclScheduleEvent& event = flow->event;
        if (dst_node != event.dst) {
            throw runtime_error("SyCCL event delivered to wrong node");
        }
        if (flow->delivered) {
            throw runtime_error("SyCCL event delivered more than once");
        }

        flow->delivered = true;
        _available_chunks.insert(chunk_location_key(event.chunk_source,
                                                    event.chunk,
                                                    event.dst));
        record_demand_delivery(event, dst_node, bytes);
        _delivered_steps++;
        if (_in_flight_steps == 0) {
            throw runtime_error("SyCCL delivered event with no in-flight step");
        }
        _in_flight_steps--;
        _in_flight_flow_ids.erase(app_flow_id);

        if (_schedule.dag_mode) {
            for (uint32_t dependent_id : flow->dependents) {
                SmartSycclFlowState* dependent = _flows[dependent_id];
                if (dependent->deps_remaining == 0) {
                    throw runtime_error("SyCCL DAG dependency underflow");
                }
                dependent->deps_remaining--;
            }
        } else {
            NodeState& state = _node_states[event.src];
            if (state.in_flight == 0) {
                throw runtime_error("SyCCL delivered event with no local in-flight step");
            }
            state.in_flight--;
            state.delivered++;

            if (state.delivered >= state.expected) {
                start_next_node_epoch(event.src);
            }
        }

        try_inject_ready_steps();
    }

    void print_schedule_report() const {
        cout << "syccl_schedule_report"
             << " mode " << (_schedule.dag_mode ? "syccl_event_dag"
                                                : (_schedule.event_mode
                                                   ? "syccl_event"
                                                   : "legacy_step"))
             << " app_demands " << _demands.size()
             << " delivered_demands " << _delivered_demands
             << " transport_steps " << _flows.size()
             << " delivered_steps " << _delivered_steps
             << " active_nodes " << active_node_count()
             << " pending_steps " << pending_step_count()
             << " in_flight_steps " << _in_flight_steps
             << " local_epoch_slots " << local_epoch_slot_count()
             << " solver_predicted_us " << _schedule.solver_predicted_us
             << endl;
    }

protected:
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
        DemandState(uint32_t demand_id,
                    const SmartSycclScheduleDemand& schedule_demand)
            : id(demand_id), demand(schedule_demand), delivered_bytes(0),
              complete(false) {
        }

        uint32_t id;
        SmartSycclScheduleDemand demand;
        mem_b delivered_bytes;
        bool complete;
    };

    void initialize_legacy_step_schedule() {
        for (uint32_t i = 0; i < _schedule.steps.size(); i++) {
            validate_step(_schedule.steps[i]);
            const SmartSycclScheduleStep& step = _schedule.steps[i];

            SmartSycclScheduleEvent event;
            event.id = i;
            event.epoch = step.epoch;
            event.chunk_source = step.chunk_source;
            event.chunk = step.chunk;
            event.src = step.src;
            event.dst = step.dst;
            event.bytes = step.bytes;
            event.amount = 1.0;
            event.copy = true;
            event.reduce = false;
            event.layer_used = -1;
            add_event(event);

            SmartSycclScheduleDemand demand;
            demand.src = step.chunk_source;
            demand.dst = step.dst;
            demand.chunk = step.chunk;
            demand.bytes = step.bytes;
            add_demand(demand);

            _available_chunks.insert(chunk_location_key(step.chunk_source,
                                                        step.chunk,
                                                        step.chunk_source));
        }
    }

    void initialize_event_schedule() {
        for (const SmartSycclScheduleChunk& chunk : _schedule.chunks) {
            validate_chunk(chunk);
            _chunk_sizes[make_pair(chunk.chunk_source, chunk.chunk)] =
                chunk.bytes;
            for (uint32_t init_node : chunk.init_nodes) {
                _available_chunks.insert(chunk_location_key(
                    chunk.chunk_source, chunk.chunk, init_node));
            }
        }

        for (const SmartSycclScheduleDemand& demand : _schedule.demands) {
            validate_demand(demand);
            add_demand(demand);
        }

        for (const SmartSycclScheduleEvent& event : _schedule.events) {
            validate_event(event);
            add_event(event);
            pair<uint32_t, uint32_t> chunk_key =
                make_pair(event.chunk_source, event.chunk);
            if (_chunk_sizes.find(chunk_key) == _chunk_sizes.end()) {
                _chunk_sizes[chunk_key] = event.bytes;
                _available_chunks.insert(chunk_location_key(
                    event.chunk_source, event.chunk, event.chunk_source));
            }
        }
    }

    void add_event(const SmartSycclScheduleEvent& event) {
        uint32_t flow_id = (uint32_t)_flows.size();
        _flows.push_back(new SmartSycclFlowState(flow_id, event));
        _node_states[event.src].steps_by_epoch[event.epoch].push_back(flow_id);
    }

    void add_demand(const SmartSycclScheduleDemand& demand) {
        uint32_t demand_id = (uint32_t)_demands.size();
        _demands.push_back(DemandState(demand_id, demand));
        _demand_by_delivery[demand_key(demand.src, demand.chunk,
                                       demand.dst)].push_back(demand_id);
        _expected_bytes += (uint64_t)demand.bytes;
    }

    void start_next_node_epoch(uint32_t node) {
        NodeState& state = _node_states[node];
        if (state.epoch_index >= state.epochs.size()) {
            state.complete = true;
            state.current_epoch = UINT32_MAX;
            state.pending.clear();
            state.expected = 0;
            state.delivered = 0;
            state.in_flight = 0;
            return;
        }

        state.complete = false;
        state.current_epoch = state.epochs[state.epoch_index++];
        state.pending = state.steps_by_epoch[state.current_epoch];
        state.expected = state.pending.size();
        state.delivered = 0;
        state.in_flight = 0;
    }

    bool step_is_ready(const SmartSycclFlowState& flow) const {
        const SmartSycclScheduleEvent& event = flow.event;
        uint64_t key = chunk_location_key(event.chunk_source, event.chunk,
                                          event.src);
        return _available_chunks.find(key) != _available_chunks.end();
    }

    void inject_step(SmartSycclFlowState& flow) {
        if (flow.injected) {
            throw runtime_error("SyCCL step injected more than once");
        }
        if (!step_is_ready(flow)) {
            throw runtime_error("SyCCL tried to inject a step before dependency was ready");
        }

        const SmartSycclScheduleEvent& event = flow.event;
        flow.injected = true;
        _in_flight_steps++;
        _in_flight_flow_ids.insert(flow.id);
        if (!_schedule.dag_mode) {
            _node_states[event.src].in_flight++;
        }
        _graph.nodes[event.src]->add_scheduled_chunk(
            flow.packet_flow, event.chunk_source, event.dst, event.epoch,
            flow.id, event.chunk, event.bytes, 32, flow.next_packet_id,
            event.path, false);
    }

    void try_inject_ready_steps() {
        if (_schedule.dag_mode) {
            vector<uint32_t> still_pending;
            for (uint32_t flow_id : _dag_pending_flow_ids) {
                SmartSycclFlowState* flow = _flows[flow_id];
                if (flow->delivered || flow->injected) {
                    continue;
                }
                if (flow->deps_remaining == 0 && step_is_ready(*flow)) {
                    inject_step(*flow);
                } else {
                    still_pending.push_back(flow_id);
                }
            }

            _dag_pending_flow_ids.clear();
            for (uint32_t flow_id : still_pending) {
                _dag_pending_flow_ids.insert(flow_id);
            }

            if (!_dag_pending_flow_ids.empty()
                && _in_flight_flow_ids.empty()) {
                SmartSycclFlowState* blocked =
                    _flows[*_dag_pending_flow_ids.begin()];
                const SmartSycclScheduleEvent& event = blocked->event;
                throw runtime_error("SyCCL DAG schedule is blocked at event "
                                    + ntoa(event.id)
                                    + " waiting for chunk "
                                    + ntoa(event.chunk)
                                    + " from source "
                                    + ntoa(event.chunk_source));
            }
            return;
        }

        for (uint32_t node = 0; node < _node_states.size(); node++) {
            NodeState& state = _node_states[node];
            if (state.complete) {
                continue;
            }

            vector<uint32_t> still_pending;
            for (uint32_t flow_id : state.pending) {
                SmartSycclFlowState* flow = _flows[flow_id];
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
        }

        if (pending_step_count() > 0 && !pending_can_eventually_be_satisfied()) {
            SmartSycclFlowState* blocked = first_pending_flow();
            const SmartSycclScheduleEvent& event = blocked->event;
            throw runtime_error("SyCCL schedule deadlock at node "
                                + ntoa(event.src) + " is waiting for chunk "
                                + ntoa(event.chunk) + " from source "
                                + ntoa(event.chunk_source)
                                + " in local epoch " + ntoa(event.epoch)
                                + " and no in-flight/local-epoch chain can satisfy any pending dependency");
        }
    }

    uint32_t pending_step_count() const {
        if (_schedule.dag_mode) {
            return (uint32_t)_dag_pending_flow_ids.size();
        }
        uint32_t count = 0;
        for (const NodeState& state : _node_states) {
            count += state.pending.size();
        }
        return count;
    }

    uint32_t active_node_count() const {
        if (_schedule.dag_mode) {
            set<uint32_t> active_nodes;
            for (uint32_t flow_id : _dag_pending_flow_ids) {
                active_nodes.insert(_flows[flow_id]->event.src);
            }
            for (uint32_t flow_id : _in_flight_flow_ids) {
                active_nodes.insert(_flows[flow_id]->event.src);
            }
            return (uint32_t)active_nodes.size();
        }
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

    SmartSycclFlowState* first_pending_flow() const {
        if (_schedule.dag_mode) {
            if (!_dag_pending_flow_ids.empty()) {
                return _flows[*_dag_pending_flow_ids.begin()];
            }
            throw runtime_error("internal error: no pending SyCCL DAG flow");
        }
        for (const NodeState& state : _node_states) {
            if (!state.pending.empty()) {
                return _flows[state.pending.front()];
            }
        }
        throw runtime_error("internal error: no pending SyCCL flow");
    }

    bool pending_can_eventually_be_satisfied() const {
        if (_in_flight_flow_ids.empty()) {
            return false;
        }

        set<uint64_t> future_available = _available_chunks;
        for (uint32_t flow_id : _in_flight_flow_ids) {
            const SmartSycclScheduleEvent& event = _flows[flow_id]->event;
            future_available.insert(chunk_location_key(event.chunk_source,
                                                       event.chunk,
                                                       event.dst));
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
                    SmartSycclFlowState* flow = _flows[flow_id];
                    if (flow->delivered || flow->injected) {
                        continue;
                    }
                    const SmartSycclScheduleEvent& event = flow->event;
                    uint64_t source_key = chunk_location_key(event.chunk_source,
                                                             event.chunk,
                                                             event.src);
                    if (future_available.find(source_key)
                        == future_available.end()) {
                        epoch_can_complete = false;
                        break;
                    }
                }
                if (!epoch_can_complete) {
                    continue;
                }

                for (uint32_t flow_id : it->second) {
                    SmartSycclFlowState* flow = _flows[flow_id];
                    if (flow->delivered) {
                        continue;
                    }
                    const SmartSycclScheduleEvent& event = flow->event;
                    future_available.insert(chunk_location_key(event.chunk_source,
                                                               event.chunk,
                                                               event.dst));
                }
                simulated_epoch_index[node]++;
                changed = true;
            }
        }

        for (const NodeState& state : _node_states) {
            for (uint32_t flow_id : state.pending) {
                const SmartSycclScheduleEvent& event = _flows[flow_id]->event;
                uint64_t needed = chunk_location_key(event.chunk_source,
                                                     event.chunk,
                                                     event.src);
                if (future_available.find(needed) != future_available.end()) {
                    return true;
                }
            }
        }
        return false;
    }

    void record_demand_delivery(const SmartSycclScheduleEvent& event,
                                uint32_t dst_node, mem_b bytes) {
        uint64_t key = demand_key(event.chunk_source, event.chunk, dst_node);
        map<uint64_t, vector<uint32_t> >::const_iterator it =
            _demand_by_delivery.find(key);
        if (it == _demand_by_delivery.end()) {
            return;
        }

        for (uint32_t demand_id : it->second) {
            DemandState& demand = _demands[demand_id];
            if (demand.complete) {
                throw runtime_error("SyCCL demand received bytes after completion");
            }
            demand.delivered_bytes += bytes;
            if (demand.delivered_bytes > demand.demand.bytes) {
                throw runtime_error("SyCCL demand delivered too many bytes");
            }
            if (demand.delivered_bytes == demand.demand.bytes) {
                demand.complete = true;
                _delivered_demands++;
                SmartNode::record_application_delivery(demand.demand.bytes);
            }
        }
    }

    void validate_chunk(const SmartSycclScheduleChunk& chunk) const {
        if (chunk.chunk_source >= _schedule.node_count) {
            throw runtime_error("SyCCL chunk source outside node_count");
        }
        if (chunk.bytes <= 0) {
            throw runtime_error("SyCCL chunk bytes must be positive");
        }
        for (uint32_t init_node : chunk.init_nodes) {
            if (init_node >= _schedule.node_count) {
                throw runtime_error("SyCCL chunk init node outside node_count");
            }
        }
    }

    void validate_demand(const SmartSycclScheduleDemand& demand) const {
        if (demand.src >= _schedule.node_count
            || demand.dst >= _schedule.node_count) {
            throw runtime_error("SyCCL demand node outside node_count");
        }
        if (demand.bytes <= 0) {
            throw runtime_error("SyCCL demand bytes must be positive");
        }
    }

    void validate_event(const SmartSycclScheduleEvent& event) const {
        if (event.chunk_source >= _schedule.node_count
            || event.src >= _schedule.node_count
            || event.dst >= _schedule.node_count) {
            throw runtime_error("SyCCL event node outside node_count");
        }
        if (event.bytes <= 0) {
            throw runtime_error("SyCCL event bytes must be positive");
        }
        if (event.src >= _graph.nodes.size() || event.dst >= _graph.nodes.size()) {
            throw runtime_error("SyCCL event node outside graph");
        }
        if (!event.path.empty()) {
            if (event.path.front() != event.src || event.path.back() != event.dst) {
                throw runtime_error("SyCCL event path endpoints do not match event src/dst");
            }
            set<uint32_t> seen;
            for (uint32_t node : event.path) {
                if (node >= _graph.nodes.size()) {
                    throw runtime_error("SyCCL event path node outside graph");
                }
                if (seen.find(node) != seen.end()) {
                    throw runtime_error("SyCCL event path repeats node "
                                        + ntoa(node));
                }
                seen.insert(node);
            }
            for (uint32_t i = 0; i + 1 < event.path.size(); i++) {
                const vector<uint32_t>& neighbors =
                    _graph.adjacency[event.path[i]];
                if (find(neighbors.begin(), neighbors.end(), event.path[i + 1])
                    == neighbors.end()) {
                    throw runtime_error("SyCCL event path uses non-edge "
                                        + ntoa(event.path[i]) + "->"
                                        + ntoa(event.path[i + 1]));
                }
            }
        }
    }

    void validate_step(const SmartSycclScheduleStep& step) const {
        if (step.chunk_source >= _schedule.node_count
            || step.src >= _schedule.node_count
            || step.dst >= _schedule.node_count) {
            throw runtime_error("SyCCL schedule step host outside schedule node_count");
        }
        if (step.bytes <= 0) {
            throw runtime_error("SyCCL schedule step bytes must be positive");
        }
        if (step.src >= _graph.nodes.size() || step.dst >= _graph.nodes.size()) {
            throw runtime_error("SyCCL schedule step node outside graph");
        }
    }

    uint64_t chunk_location_key(uint32_t source, uint32_t chunk,
                                uint32_t node) const {
        uint64_t key = source;
        key = key * _schedule.node_count + node;
        key = (key << 32) | chunk;
        return key;
    }

    uint64_t demand_key(uint32_t source, uint32_t chunk,
                        uint32_t dst) const {
        return chunk_location_key(source, chunk, dst);
    }

    void build_event_dag_dependencies() {
        map<uint64_t, uint32_t> producer_by_location;
        vector<uint32_t> order;
        for (uint32_t flow_id = 0; flow_id < _flows.size(); flow_id++) {
            order.push_back(flow_id);
        }
        sort(order.begin(), order.end(),
             [this](uint32_t left, uint32_t right) {
                 const SmartSycclScheduleEvent& a = _flows[left]->event;
                 const SmartSycclScheduleEvent& b = _flows[right]->event;
                 if (a.epoch != b.epoch) {
                     return a.epoch < b.epoch;
                 }
                 return a.id < b.id;
             });

        for (uint32_t flow_id : order) {
            SmartSycclFlowState* flow = _flows[flow_id];
            const SmartSycclScheduleEvent& event = flow->event;
            uint64_t source_key =
                chunk_location_key(event.chunk_source, event.chunk,
                                   event.src);
            if (_available_chunks.find(source_key) == _available_chunks.end()) {
                map<uint64_t, uint32_t>::const_iterator producer =
                    producer_by_location.find(source_key);
                if (producer != producer_by_location.end()) {
                    uint32_t producer_id = producer->second;
                    flow->deps_remaining++;
                    _flows[producer_id]->dependents.push_back(flow_id);
                }
            }

            uint64_t dst_key =
                chunk_location_key(event.chunk_source, event.chunk,
                                   event.dst);
            producer_by_location[dst_key] = flow_id;
        }
    }

    SmartGraph& _graph;
    SmartSycclSchedule _schedule;
    vector<SmartSycclFlowState*> _flows;
    vector<NodeState> _node_states;
    vector<DemandState> _demands;
    map<pair<uint32_t, uint32_t>, mem_b> _chunk_sizes;
    map<uint64_t, vector<uint32_t> > _demand_by_delivery;
    set<uint64_t> _available_chunks;
    set<uint32_t> _dag_pending_flow_ids;
    uint64_t _expected_bytes;
    uint64_t _delivered_steps;
    uint64_t _delivered_demands;
    uint32_t _in_flight_steps;
    set<uint32_t> _in_flight_flow_ids;
};

#endif
#endif
