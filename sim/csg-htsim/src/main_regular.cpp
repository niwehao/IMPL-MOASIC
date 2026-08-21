// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "network.h"
#include "../topo/clos_graph.h"
#include "../topo/dgx_rail_graph.h"
#include "../topo/flex_pod_graph.h"
#include "../topo/flex_cube_graph.h"
#include "../topo/ndv2_graph.h"
#include "../topo/regular_graph.h"
#include "node.h"
#include "packet.h"
#include "../scheduler/fast_schedule.h"
#include "../scheduler/mosaic_schedule.h"
#include "../scheduler/syccl_schedule.h"
#include "../scheduler/taccl_schedule.h"
#include "../scheduler/teccl_schedule.h"
#include "../scheduler/static_ecmp.h"
#include "wait_queue.h"
#include "workflow.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <string.h>
#include <vector>

struct SmartFlowState {
    SmartFlowState(uint32_t flow_id, uint32_t workflow_round,
                   uint32_t src_node, uint32_t dst_node,
                   mem_b message_bytes, mem_b chunk_bytes)
        : id(flow_id), round(workflow_round), src(src_node), dst(dst_node),
          message_size(message_bytes), chunk_size(chunk_bytes),
          chunk_count((uint32_t)((message_bytes + chunk_bytes - 1) / chunk_bytes)),
          next_chunk_id(0), next_packet_id(1), packet_flow(NULL) {
    }

    bool done() const {
        return next_chunk_id >= chunk_count;
    }

    mem_b current_chunk_size() const {
        mem_b remaining = message_size - next_chunk_id * chunk_size;
        return remaining < chunk_size ? remaining : chunk_size;
    }

    uint32_t id;
    uint32_t round;
    uint32_t src;
    uint32_t dst;
    mem_b message_size;
    mem_b chunk_size;
    uint32_t chunk_count;
    uint32_t next_chunk_id;
    packetid_t next_packet_id;
    PacketFlow packet_flow;
};

class SmartWorkflowRuntime : public SmartDeliveryObserver {
public:
    SmartWorkflowRuntime(SmartGraph& graph, const SmartWorkflow& workflow,
                         mem_b chunk_size)
        : _graph(graph), _workflow(workflow), _chunk_size(chunk_size),
          _expected_chunks(0), _expected_bytes(0) {
        uint32_t rounds = workflow.round_count();
        uint32_t nodes = workflow.node_count;

        _outgoing_flow_ids.resize(rounds);
        _expected_in_chunks.resize(rounds, vector<uint64_t>(nodes, 0));
        _delivered_in_chunks.resize(rounds, vector<uint64_t>(nodes, 0));
        _round_started.resize(rounds, vector<bool>(nodes, false));
        _round_start_count.resize(rounds, 0);
        _round_first_start_time.resize(rounds, UINT64_MAX);
        _round_last_start_time.resize(rounds, 0);
        _round_expected_chunks.resize(rounds, 0);
        _round_delivered_chunks.resize(rounds, 0);
        _round_complete_time.resize(rounds, UINT64_MAX);

        for (uint32_t round = 0; round < rounds; round++) {
            _outgoing_flow_ids[round].resize(nodes);
            vector<SmartWorkflowTransfer> transfers =
                workflow.transfers_for_round(round);
            for (const SmartWorkflowTransfer& transfer : transfers) {
                uint32_t flow_id = (uint32_t)_flows.size();
                SmartFlowState* flow = new SmartFlowState(
                    flow_id, transfer.round, transfer.src, transfer.dst,
                    transfer.bytes, chunk_size);
                _flows.push_back(flow);
                _outgoing_flow_ids[round][transfer.src].push_back(flow_id);
                _expected_in_chunks[round][transfer.dst] += flow->chunk_count;
                _round_expected_chunks[round] += flow->chunk_count;
                _expected_chunks += flow->chunk_count;
                _expected_bytes += (uint64_t)flow->message_size;
            }
        }
    }

    const vector<SmartFlowState*>& flows() const {
        return _flows;
    }

    uint64_t expected_chunks() const {
        return _expected_chunks;
    }

    uint64_t expected_bytes() const {
        return _expected_bytes;
    }

    void start_ready_rounds() {
        for (uint32_t node = 0; node < _workflow.node_count; node++) {
            try_start_round(node, 0);
        }
    }

    void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                         uint32_t dst_node, mem_b bytes) override {
        (void)chunk_id;
        (void)bytes;

        if (app_flow_id >= _flows.size()) {
            return;
        }

        SmartFlowState* flow = _flows[app_flow_id];
        if (dst_node != flow->dst) {
            abort();
        }

        _delivered_in_chunks[flow->round][dst_node]++;
        _round_delivered_chunks[flow->round]++;
        if (_round_delivered_chunks[flow->round] >= _round_expected_chunks[flow->round]
            && _round_complete_time[flow->round] == UINT64_MAX) {
            _round_complete_time[flow->round] = EventList::now();
        }
        try_start_round(dst_node, flow->round + 1);
    }

    void print_round_report() const {
        cout << "workflow_round_report";
        for (uint32_t round = 0; round < _workflow.round_count(); round++) {
            cout << " round " << round
                 << " started_nodes " << _round_start_count[round]
                 << " first_start_us " << time_or_na_us(_round_first_start_time[round])
                 << " last_start_us " << time_or_na_us(_round_last_start_time[round])
                 << " complete_us " << time_or_na_us(_round_complete_time[round]);

            if (round > 0 && _round_first_start_time[round] != UINT64_MAX
                && _round_complete_time[round - 1] != UINT64_MAX
                && _round_first_start_time[round] < _round_complete_time[round - 1]) {
                cout << " overlaps_prev yes";
            } else if (round > 0) {
                cout << " overlaps_prev no";
            }
        }
        cout << endl;
    }

private:
    void try_start_round(uint32_t node, uint32_t round) {
        while (round < _workflow.round_count()
               && !_round_started[round][node]
               && previous_round_complete(node, round)) {
            _round_started[round][node] = true;
            record_round_start(round);
            inject_node_round(node, round);

            if (_expected_in_chunks[round][node] != 0) {
                break;
            }
            round++;
        }
    }

    bool previous_round_complete(uint32_t node, uint32_t round) const {
        if (round == 0) {
            return true;
        }

        uint32_t previous = round - 1;
        return _delivered_in_chunks[previous][node]
            >= _expected_in_chunks[previous][node];
    }

    void inject_node_round(uint32_t node, uint32_t round) {
        for (uint32_t flow_id : _outgoing_flow_ids[round][node]) {
            SmartFlowState* flow = _flows[flow_id];
            while (!flow->done()) {
                _graph.nodes[flow->src]->add_local_chunk(
                    flow->packet_flow, flow->dst, flow->round, flow->id,
                    flow->next_chunk_id, flow->current_chunk_size(),
                    32, flow->next_packet_id);
                flow->next_chunk_id++;
            }
        }
    }

    void record_round_start(uint32_t round) {
        simtime_picosec now = EventList::now();
        _round_start_count[round]++;
        if (now < _round_first_start_time[round]) {
            _round_first_start_time[round] = now;
        }
        if (now > _round_last_start_time[round]) {
            _round_last_start_time[round] = now;
        }
    }

    static double time_or_na_us(simtime_picosec timestamp) {
        if (timestamp == UINT64_MAX) {
            return -1.0;
        }
        return timeAsUs(timestamp);
    }

    SmartGraph& _graph;
    const SmartWorkflow& _workflow;
    mem_b _chunk_size;
    vector<SmartFlowState*> _flows;
    vector<vector<vector<uint32_t> > > _outgoing_flow_ids;
    vector<vector<uint64_t> > _expected_in_chunks;
    vector<vector<uint64_t> > _delivered_in_chunks;
    vector<vector<bool> > _round_started;
    vector<uint32_t> _round_start_count;
    vector<simtime_picosec> _round_first_start_time;
    vector<simtime_picosec> _round_last_start_time;
    vector<uint64_t> _round_expected_chunks;
    vector<uint64_t> _round_delivered_chunks;
    vector<simtime_picosec> _round_complete_time;
    uint64_t _expected_chunks;
    uint64_t _expected_bytes;
};

static void usage(char* program) {
    cout << "Usage: " << program
         << " [-nodes N] [-degree D] [-src node] [-dst node]"
         << " [-all_to_all] [-packet_size bytes] [-message_size bytes]"
         << " [-chunk_size bytes] [-output_buffer_size bytes] [-rounds count] [-link_rate_gbps Gbps]"
         << " [-ecn_threshold_bytes bytes] [-pause_threshold_bytes bytes]"
         << " [-split_minus W] [-split_zero W] [-split_plus W]"
         << " [-topology regular|clos|ndv2|ndv2_32|dgx_rail]"
         << " [-clos_leafs L] [-clos_spines S]"
         << " [-regular_hetero_ring] [-regular_hetero_multiplier value]"
         << " [-scheduler name] [-workload name] [-schedule_file file]"
         << " [-demand_file file] [-fast_gpus_per_server G]"
         << " [-fast_sync none|credit|barrier]"
         << " [-fast_stage_order desc|plan]"
         << " [-store_forward] [-event_batch ignored] [-quiet]" << endl;
}

static void print_completion_cdf() {
    vector<simtime_picosec> times = SmartNode::app_completion_times();
    if (times.empty()) {
        cout << "completion_cdf empty" << endl;
        return;
    }

    sort(times.begin(), times.end());
    uint32_t percentiles[] = {10, 25, 50, 75, 90, 95, 99, 100};
    cout << "completion_cdf samples " << times.size();
    for (uint32_t p : percentiles) {
        uint64_t rank = ((uint64_t)p * times.size() + 99) / 100;
        if (rank == 0) {
            rank = 1;
        }
        if (rank > times.size()) {
            rank = times.size();
        }
        double time_us = (double)times[rank - 1] / 1000000.0;
        cout << " p" << p << "_us " << time_us;
    }
    cout << endl;
}

int main(int argc, char** argv) {
    uint32_t node_count = 8;
    uint32_t degree = 2;
    uint32_t src = 0;
    uint32_t dst = 4;
    bool all_to_all = false;
    bool quiet = false;
    int packet_size = 1024;
    mem_b message_size = 2 * 1024 * 1024;
    mem_b chunk_size = 2 * 1024 * 1024;
    mem_b output_buffer_size = 0;
    bool chunk_size_set = false;
    uint32_t workflow_rounds = 1;
    double link_rate_gbps = 100.0;
    mem_b ecn_threshold_bytes = 0;
    mem_b pause_threshold_bytes = 0;
    uint32_t split_minus = 1;
    uint32_t split_zero = 0;
    uint32_t split_plus = 0;
    string topology_name = "regular";
    uint32_t clos_leafs = 0;
    uint32_t clos_spines = 0;
    bool regular_hetero_ring = false;
    double regular_hetero_multiplier = 2.0;
    string scheduler_algorithm = "basic";
    string workload_name = "A2A";
    string schedule_file = "";
    string demand_file = "";
    uint32_t fast_gpus_per_server = 0;  // 0 = auto by topology
    bool fast_alpha_beta_guard = false;
    uint32_t mosaic_waves = 6;
    double mosaic_gamma = 0.7;
    double mosaic_floor = 0.02;
    double mosaic_threshold = 0.75;
    bool mosaic_adapt = true;
    bool mosaic_trace = false;
    bool mosaic_donate_window = false;
    bool mosaic_donate_rewaterfill = false;
    uint32_t mosaic_skew_epoch = 0;
    uint32_t mosaic_pair_color = 0;
    double mosaic_guard = 0.0;
    bool mosaic_phase_plan = false;
    bool mosaic_hop_progress = false;
    uint32_t mosaic_cold_pace = 0;
    string mosaic_decision_trace;  // 决策轨迹 dump 路径(真机对拍用,默认关)
    bool mosaic_host_lambda = false;
    uint64_t mosaic_wave_bytes = 0;
    uint32_t mosaic_wave_min = 1;
    bool mosaic_wave_align = false;
    uint32_t mosaic_ramp_start = 0;
    uint32_t mosaic_ramp_cold = 0;
    bool mosaic_ramp_event = false;
    bool mosaic_ramp_cold_force = false;
    bool mosaic_uniform_init = false;
    bool mosaic_cube_nic_cap = false;
    bool mosaic_cube_lane_rot = false;
    string mosaic_lp_intra_file = "";
    bool mosaic_lp_local = false;
    string mosaic_lp_spread_file = "";
    bool mosaic_rx_init = false;
    string mosaic_prior_demand_file = "";
    mem_b fast_scaling = 0;
    bool fast_rr_select = false;
    // Execution-time stage sync (official FAST runtime semantics):
    // none (default, legacy) | credit (NVSHMEM kernel gates) |
    // barrier (analytical-model global stage barrier).
    string fast_sync = "none";
    // desc (default) = production FAST's largest-freq-first stage order;
    // plan = raw Birkhoff extraction order (historical numbers).
    string fast_stage_order = "desc";
    bool store_forward = false;
    double dgx_taper_gbps = 180.0;
    double link_delay_us = 5.0;
    double clos_uplink_gbps = 0.0;  // 0 = auto (2:1 leaf oversubscription)
    uint32_t flex_hosts = 4;
    uint32_t flex_gpus_per_host = 8;
    double flex_nv_gbps = 2400.0;
    double flex_nic_gbps = 180.0;
    double flex_taper_gbps = 180.0;
    uint32_t flex_spines = 1;
    mem_b flex_fabric_buffer_mb = 1024;
    uint32_t flex_degrade_gpu = UINT32_MAX;
    double flex_degrade_factor = 1.0;
    uint32_t flex_degrade_dir = 0;    // 0 both, 1 egress only, 2 ingress only
    vector<char> flex_degrade_mask;   // -flex_degrade_gpus a,b,c
    double flex_planeb_factor = 1.0;
    uint32_t mosaic_corridors = 4;
    bool mosaic_ledger = false;       // x41 MOSAIC-L resource ledger
    bool mosaic_ledger_raw = false;   // ledger without rate correction
    string mosaic_ledger_sync = "oracle";  // x42: oracle|strong|snap|none
    double mosaic_sync_us = 100.0;         // x42: sync latency/period
    bool mosaic_lg_bcast = false;          // x44: policy broadcasts
    double mosaic_lg_blag_us = 0.0;        // x44: broadcast latency
    double mosaic_lg_blagw = 0.0;          // x44: lag in publish beats
    bool mosaic_lg_spineagg = false;       // x44: two-level fabric
    bool mosaic_lg_fabmask = false;        // x44: ablation floor
    bool mosaic_lg_forkprobe = false;      // x44: replay-fork probe
    double mosaic_lg_stagger_us = 0.0;     // x44: decision slot spacing
    bool mosaic_explicit_spine = false;
    double mosaic_known_planeb = 1.0;
    // flex_cube: intra-host cube-mesh speed layout.
    string flex_cube_speeds = "400x6,200x6,100x4";
    uint32_t flex_cube_mix_rot = 0;
    double flex_cube_full_gbps = 0.0;  // >0 = full crossbar degenerate mode
    double flex_cube_link_us = 0.7;

    int i = 1;
    while (i < argc) {
        if (!strcmp(argv[i], "-nodes") && i + 1 < argc) {
            node_count = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-degree") && i + 1 < argc) {
            degree = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-src") && i + 1 < argc) {
            src = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-dst") && i + 1 < argc) {
            dst = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-packet_size") && i + 1 < argc) {
            packet_size = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-message_size") && i + 1 < argc) {
            message_size = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-chunk_size") && i + 1 < argc) {
            chunk_size = atoll(argv[i + 1]);
            chunk_size_set = true;
            i++;
        } else if (!strcmp(argv[i], "-output_buffer_size") && i + 1 < argc) {
            output_buffer_size = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-rounds") && i + 1 < argc) {
            workflow_rounds = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-link_rate_gbps") && i + 1 < argc) {
            link_rate_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-ecn_threshold_bytes") && i + 1 < argc) {
            ecn_threshold_bytes = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-pause_threshold_bytes") && i + 1 < argc) {
            pause_threshold_bytes = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-split_minus") && i + 1 < argc) {
            split_minus = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-split_zero") && i + 1 < argc) {
            split_zero = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-split_plus") && i + 1 < argc) {
            split_plus = atoi(argv[i + 1]);
            i++;
        } else if ((!strcmp(argv[i], "-topology") || !strcmp(argv[i], "-topo"))
                   && i + 1 < argc) {
            topology_name = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-clos_leafs") && i + 1 < argc) {
            clos_leafs = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-clos_spines") && i + 1 < argc) {
            clos_spines = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-regular_hetero_ring")) {
            regular_hetero_ring = true;
        } else if (!strcmp(argv[i], "-regular_hetero_multiplier") && i + 1 < argc) {
            regular_hetero_multiplier = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-scheduler") && i + 1 < argc) {
            scheduler_algorithm = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-workload") && i + 1 < argc) {
            workload_name = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-schedule_file") && i + 1 < argc) {
            schedule_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-demand_file") && i + 1 < argc) {
            demand_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-fast_gpus_per_server") && i + 1 < argc) {
            fast_gpus_per_server = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-fast_alpha_beta_guard")) {
            fast_alpha_beta_guard = true;
        } else if (!strcmp(argv[i], "-mosaic_waves") && i + 1 < argc) {
            mosaic_waves = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_gamma") && i + 1 < argc) {
            mosaic_gamma = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_floor") && i + 1 < argc) {
            mosaic_floor = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_threshold") && i + 1 < argc) {
            mosaic_threshold = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_ledger")) {
            mosaic_ledger = true;
        } else if (!strcmp(argv[i], "-mosaic_ledger_raw")) {
            mosaic_ledger = true;
            mosaic_ledger_raw = true;
        } else if (!strcmp(argv[i], "-mosaic_ledger_sync") && i + 1 < argc) {
            mosaic_ledger = true;
            mosaic_ledger_sync = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-mosaic_sync_us") && i + 1 < argc) {
            mosaic_sync_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_lg_bcast")) {
            mosaic_lg_bcast = true;                // x44 channel B
        } else if (!strcmp(argv[i], "-mosaic_lg_blag_us") && i + 1 < argc) {
            mosaic_lg_blag_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_lg_blagw") && i + 1 < argc) {
            mosaic_lg_blagw = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_lg_spineagg")) {
            mosaic_lg_spineagg = true;             // x44 two-level
        } else if (!strcmp(argv[i], "-mosaic_lg_fabmask")) {
            mosaic_lg_fabmask = true;              // x44 ablation floor
        } else if (!strcmp(argv[i], "-mosaic_lg_stagger_us") && i + 1 < argc) {
            mosaic_lg_stagger_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_lg_forkprobe")) {
            mosaic_lg_forkprobe = true;            // x44 diagnostics
        } else if (!strcmp(argv[i], "-mosaic_static")) {
            mosaic_adapt = false;
        } else if (!strcmp(argv[i], "-mosaic_trace")) {
            mosaic_trace = true;
        } else if (!strcmp(argv[i], "-mosaic_donate_window")) {
            mosaic_donate_window = true;
        } else if (!strcmp(argv[i], "-mosaic_donate_rewaterfill")) {
            mosaic_donate_rewaterfill = true;
        } else if (!strcmp(argv[i], "-mosaic_pair_color") && i + 1 < argc) {
            mosaic_pair_color = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_pair_color_auto")) {
            mosaic_pair_color = UINT32_MAX;
        } else if (!strcmp(argv[i], "-mosaic_guard") && i + 1 < argc) {
            mosaic_guard = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_skew_epoch") && i + 1 < argc) {
            mosaic_skew_epoch = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_phase_plan")) {
            mosaic_phase_plan = true;
        } else if (!strcmp(argv[i], "-mosaic_hop_progress")) {
            mosaic_hop_progress = true;
        } else if (!strcmp(argv[i], "-mosaic_cold_pace") && i + 1 < argc) {
            mosaic_cold_pace = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_decision_trace")
                   && i + 1 < argc) {
            mosaic_decision_trace = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-mosaic_host_lambda")) {
            mosaic_host_lambda = true;
        } else if (!strcmp(argv[i], "-mosaic_wave_bytes") && i + 1 < argc) {
            mosaic_wave_bytes = strtoull(argv[i + 1], NULL, 10);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_wave_min") && i + 1 < argc) {
            mosaic_wave_min = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_wave_align")) {
            mosaic_wave_align = true;
        } else if (!strcmp(argv[i], "-mosaic_ramp_start") && i + 1 < argc) {
            mosaic_ramp_start = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_ramp_cold") && i + 1 < argc) {
            mosaic_ramp_cold = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_ramp_event")) {
            mosaic_ramp_event = true;
        } else if (!strcmp(argv[i], "-mosaic_ramp_cold_force")) {
            mosaic_ramp_cold_force = true;
        } else if (!strcmp(argv[i], "-mosaic_cube_nic_cap")) {
            mosaic_cube_nic_cap = true;
        } else if (!strcmp(argv[i], "-mosaic_cube_lane_rot")) {
            mosaic_cube_lane_rot = true;
        } else if (!strcmp(argv[i], "-mosaic_lp_intra") && i + 1 < argc) {
            mosaic_lp_intra_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-mosaic_lp_local")) {
            mosaic_lp_local = true;
        } else if (!strcmp(argv[i], "-mosaic_lp_spread") && i + 1 < argc) {
            mosaic_lp_spread_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-mosaic_uniform_init")) {
            mosaic_uniform_init = true;
        } else if (!strcmp(argv[i], "-mosaic_rx_init")) {
            mosaic_rx_init = true;
        } else if (!strcmp(argv[i], "-mosaic_prior_demand_file")
                   && i + 1 < argc) {
            mosaic_prior_demand_file = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-fast_scaling") && i + 1 < argc) {
            fast_scaling = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-fast_rr_select")) {
            fast_rr_select = true;
        } else if (!strcmp(argv[i], "-fast_sync") && i + 1 < argc) {
            fast_sync = argv[i + 1];
            i++;
            if (fast_sync != "none" && fast_sync != "credit"
                && fast_sync != "barrier") {
                cout << "-fast_sync must be none, credit or barrier" << endl;
                return 1;
            }
        } else if (!strcmp(argv[i], "-fast_stage_order") && i + 1 < argc) {
            fast_stage_order = argv[i + 1];
            i++;
            if (fast_stage_order != "desc" && fast_stage_order != "plan") {
                cout << "-fast_stage_order must be desc or plan" << endl;
                return 1;
            }
        } else if (!strcmp(argv[i], "-dgx_taper_gbps") && i + 1 < argc) {
            dgx_taper_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_hosts") && i + 1 < argc) {
            flex_hosts = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_gpus_per_host") && i + 1 < argc) {
            flex_gpus_per_host = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_nv_gbps") && i + 1 < argc) {
            flex_nv_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_nic_gbps") && i + 1 < argc) {
            flex_nic_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_taper_gbps") && i + 1 < argc) {
            flex_taper_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_cube_speeds") && i + 1 < argc) {
            flex_cube_speeds = argv[i + 1];
            i++;
        } else if (!strcmp(argv[i], "-flex_cube_mix_rot") && i + 1 < argc) {
            flex_cube_mix_rot = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_cube_full_gbps") && i + 1 < argc) {
            flex_cube_full_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_cube_link_us") && i + 1 < argc) {
            flex_cube_link_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_spines") && i + 1 < argc) {
            flex_spines = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_fabric_buffer_mb") && i + 1 < argc) {
            flex_fabric_buffer_mb = atoll(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_degrade_gpu") && i + 1 < argc) {
            flex_degrade_gpu = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_degrade_factor") && i + 1 < argc) {
            flex_degrade_factor = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_degrade_dir") && i + 1 < argc) {
            // 32 组 instrument: which half of the NIC is slowed
            // (0 = both, the 19 组 default; 1 = egress; 2 = ingress).
            flex_degrade_dir = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-flex_degrade_gpus") && i + 1 < argc) {
            // 32 组 instrument: comma-separated GPU set degraded by the
            // same factor/dir (oracle throttling experiments).
            const char* p = argv[i + 1];
            while (*p) {
                uint32_t g = (uint32_t)atoi(p);
                if (flex_degrade_mask.size() <= g) {
                    flex_degrade_mask.resize(g + 1, 0);
                }
                flex_degrade_mask[g] = 1;
                while (*p && *p != ',') {
                    p++;
                }
                if (*p == ',') {
                    p++;
                }
            }
            i++;
        } else if (!strcmp(argv[i], "-flex_planeb_factor") && i + 1 < argc) {
            flex_planeb_factor = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_corridors") && i + 1 < argc) {
            mosaic_corridors = atoi(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-mosaic_explicit_spine")) {
            mosaic_explicit_spine = true;
        } else if (!strcmp(argv[i], "-mosaic_known_planeb") && i + 1 < argc) {
            mosaic_known_planeb = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-link_delay_us") && i + 1 < argc) {
            link_delay_us = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-clos_uplink_gbps") && i + 1 < argc) {
            clos_uplink_gbps = atof(argv[i + 1]);
            i++;
        } else if (!strcmp(argv[i], "-store_forward")) {
            store_forward = true;
        } else if (!strcmp(argv[i], "-event_batch") && i + 1 < argc) {
            i++;
        } else if (!strcmp(argv[i], "-all_to_all")) {
            all_to_all = true;
        } else if (!strcmp(argv[i], "-quiet")) {
            quiet = true;
        } else {
            usage(argv[0]);
            return 1;
        }
        i++;
    }

    if (all_to_all) {
        quiet = true;
    }

    if (schedule_file.empty() && (src >= node_count || dst >= node_count)) {
        usage(argv[0]);
        return 1;
    }
    if (packet_size <= 0 || packet_size > 65535) {
        cout << "packet_size must be in range 1..65535" << endl;
        return 1;
    }
    if (message_size <= 0 || chunk_size <= 0) {
        cout << "message_size and chunk_size must be greater than zero" << endl;
        return 1;
    }
    if (output_buffer_size < 0) {
        cout << "output_buffer_size must be non-negative" << endl;
        return 1;
    }
    if (output_buffer_size > 0 && !all_to_all) {
        cout << "output_buffer_size requires -all_to_all" << endl;
        return 1;
    }
    if (workflow_rounds == 0) {
        cout << "rounds must be greater than zero" << endl;
        return 1;
    }
    if (link_rate_gbps <= 0) {
        cout << "link_rate_gbps must be greater than zero" << endl;
        return 1;
    }
    if (ecn_threshold_bytes < 0 || pause_threshold_bytes < 0) {
        cout << "ecn_threshold_bytes and pause_threshold_bytes must be non-negative" << endl;
        return 1;
    }
    if (split_minus == 0 && split_zero == 0 && split_plus == 0) {
        cout << "at least one split weight must be greater than zero" << endl;
        return 1;
    }
    if (topology_name != "regular" && topology_name != "clos"
        && topology_name != "ndv2"
        && topology_name != "dgx_rail"
        && topology_name != "ndv2_32"
        && topology_name != "flex_rail"
        && topology_name != "flex_pod"
        && topology_name != "flex_rail2p"
        && topology_name != "flex_cube") {
        cout << "topology must be regular, clos, ndv2, ndv2_32, dgx_rail, "
             << "flex_rail, flex_pod, flex_rail2p or flex_cube" << endl;
        return 1;
    }
    if ((topology_name == "flex_rail" || topology_name == "flex_pod"
         || topology_name == "flex_rail2p" || topology_name == "flex_cube")
        && node_count != flex_hosts * flex_gpus_per_host) {
        cout << "topology flex_rail requires nodes = flex_hosts * "
             << "flex_gpus_per_host" << endl;
        return 1;
    }
    if (topology_name == "dgx_rail" && node_count != DGX_RAIL_NODE_COUNT) {
        cout << "topology dgx_rail requires nodes=" << DGX_RAIL_NODE_COUNT
             << endl;
        return 1;
    }
    if (topology_name == "ndv2_32" && node_count != NDV2_32_NODE_COUNT) {
        cout << "topology ndv2_32 requires nodes=" << NDV2_32_NODE_COUNT
             << endl;
        return 1;
    }
    if (scheduler_algorithm != "basic"
        && scheduler_algorithm != "static_ecmp"
        && scheduler_algorithm != "teccl_schedule"
        && scheduler_algorithm != "syccl_schedule"
        && scheduler_algorithm != "taccl_schedule"
        && scheduler_algorithm != "fast_schedule"
        && scheduler_algorithm != "mosaic_schedule") {
        cout << "scheduler must be basic, static_ecmp, teccl_schedule, "
             << "syccl_schedule, taccl_schedule, fast_schedule, "
             << "or mosaic_schedule" << endl;
        return 1;
    }
    if (scheduler_algorithm == "fast_schedule" && !schedule_file.empty()) {
        cout << "fast_schedule computes its plan online; -schedule_file "
             << "is not supported" << endl;
        return 1;
    }
    if (topology_name == "ndv2" && node_count != NDV2_LIKE_NODE_COUNT) {
        cout << "topology ndv2 requires nodes=" << NDV2_LIKE_NODE_COUNT
             << endl;
        return 1;
    }
    if (regular_hetero_multiplier <= 0.0) {
        cout << "regular_hetero_multiplier must be greater than zero" << endl;
        return 1;
    }
    if (!schedule_file.empty()) {
        if (scheduler_algorithm.empty() || scheduler_algorithm == "basic") {
            scheduler_algorithm = "teccl_schedule";
        }
        if (scheduler_algorithm != "teccl_schedule"
            && scheduler_algorithm != "syccl_schedule"
            && scheduler_algorithm != "taccl_schedule") {
            cout << "schedule_file requires scheduler teccl_schedule, syccl_schedule, or taccl_schedule"
                 << endl;
            return 1;
        }
        if (scheduler_algorithm == "teccl_schedule"
            && topology_name != "regular" && topology_name != "clos"
            && topology_name != "ndv2"
            && topology_name != "dgx_rail"
            && topology_name != "flex_rail"
            && topology_name != "flex_cube"
            && topology_name != "ndv2_32") {
            cout << "TE-CCL schedule mode currently requires topology=regular "
                 << "or clos or ndv2 or ndv2_32 or dgx_rail or flex_rail "
                 << "or flex_cube"
                 << endl;
            return 1;
        }
    }
    EventList eventlist;
    eventlist.setEndtime(timeFromSec(10000));

    Packet::set_packet_size(packet_size);

    if (link_delay_us < 0) {
        cout << "link_delay_us must be non-negative" << endl;
        return 1;
    }

    linkspeed_bps speed = speedFromGbps(link_rate_gbps);
    mem_b buffer = chunk_size;
    // Uniform per-link pipe delay for regular/clos; the solver-side topo
    // description must use the same alpha (teccl regular default is 5e-6 s
    // to match). ndv2/ndv2_32/dgx_rail carry their own per-edge delays.
    simtime_picosec delay = timeFromUs(link_delay_us);

    Logfile logfile("smart_regular.dat", eventlist);
    logfile.setStartTime(0);

    SmartNode::set_verbose(!quiet);
    SmartNode::reset_stats();
    SmartNode::set_split_weights(split_minus, split_zero, split_plus);
    SmartNode::set_scheduler_algorithm(scheduler_algorithm);
    SmartNode::set_store_forward(store_forward);

    SmartGraph graph;
    if (topology_name == "regular") {
        graph = build_d_regular_graph(node_count, degree, eventlist,
                                      logfile, speed, buffer, delay,
                                      regular_hetero_ring,
                                      regular_hetero_multiplier);
    } else if (topology_name == "clos") {
        graph = build_two_layer_clos_graph(
            node_count, clos_leafs, clos_spines, eventlist, logfile, speed,
            speedFromGbps(clos_uplink_gbps), delay,
            (uint32_t)flex_fabric_buffer_mb);
    } else if (topology_name == "ndv2") {
        graph = build_ndv2_like_graph(eventlist, logfile, buffer, delay);
    } else if (topology_name == "dgx_rail") {
        graph = build_dgx_rail_graph(eventlist, logfile, dgx_taper_gbps);
    } else if (topology_name == "flex_rail") {
        graph = build_flex_rail_graph(eventlist, logfile, flex_hosts,
                                      flex_gpus_per_host, flex_nv_gbps,
                                      flex_nic_gbps, flex_taper_gbps,
                                      flex_spines, flex_fabric_buffer_mb,
                                      flex_degrade_gpu, flex_degrade_factor,
                                      flex_degrade_dir, flex_degrade_mask);
    } else if (topology_name == "flex_rail2p") {
        graph = build_flex_rail2p_graph(eventlist, logfile, flex_hosts,
                                        flex_gpus_per_host, flex_nv_gbps,
                                        flex_nic_gbps, flex_taper_gbps,
                                        flex_planeb_factor,
                                        flex_fabric_buffer_mb);
    } else if (topology_name == "flex_pod") {
        // flex_taper_gbps = TOTAL leaf uplink (split across flex_spines).
        graph = build_flex_pod_graph(eventlist, logfile, flex_hosts,
                                     flex_gpus_per_host, flex_nv_gbps,
                                     flex_nic_gbps, flex_taper_gbps,
                                     flex_spines, flex_fabric_buffer_mb);
    } else if (topology_name == "flex_cube") {
        vector<vector<double> > cube_local = flex_cube_local_matrix(
            flex_gpus_per_host, flex_cube_speeds, flex_cube_mix_rot,
            flex_cube_full_gbps);
        graph = build_flex_cube_graph(eventlist, logfile, flex_hosts,
                                      flex_gpus_per_host, cube_local,
                                      flex_nic_gbps, flex_taper_gbps,
                                      flex_spines, flex_cube_link_us,
                                      flex_fabric_buffer_mb);
    } else {
        graph = build_ndv2_32_graph(eventlist, logfile);
    }

    if (scheduler_algorithm == "static_ecmp") {
        StaticEcmpScheduler::install_paths(graph);
    }

    for (BaseQueue* queue : graph.queues) {
        SmartQueue* smart_queue = dynamic_cast<SmartQueue*>(queue);
        if (smart_queue) {
            smart_queue->set_congestion_thresholds(ecn_threshold_bytes,
                                                   pause_threshold_bytes);
        }
    }

    uint64_t expected_chunks = 0;
    uint64_t expected_bytes = 0;
    uint64_t flow_count = 0;
    string run_mode = "workflow";
    SmartWorkflowRuntime* workflow_runtime = NULL;
    SmartTecclScheduleRuntime* schedule_runtime = NULL;
    SmartSycclScheduleRuntime* syccl_schedule_runtime = NULL;
    SmartTacclScheduleRuntime* taccl_schedule_runtime = NULL;

    SmartMosaicRuntime* mosaic_runtime = NULL;

    vector<vector<mem_b> > demand;
    if (!demand_file.empty()) {
        demand = load_demand_matrix(demand_file, node_count);
    } else if (scheduler_algorithm == "fast_schedule"
               || scheduler_algorithm == "mosaic_schedule") {
        // Default uniform A2A demand from the workload contract.
        demand.assign(node_count, vector<mem_b>(node_count, 0));
        for (uint32_t s = 0; s < node_count; s++) {
            for (uint32_t d = 0; d < node_count; d++) {
                if (s != d) {
                    demand[s][d] = message_size;
                }
            }
        }
    }

    uint32_t plan_gpus = fast_gpus_per_server;
    if (plan_gpus == 0) {
        if (topology_name == "dgx_rail") {
            plan_gpus = DGX_RAIL_GPUS_PER_HOST;
        } else if (topology_name == "flex_rail"
                   || topology_name == "flex_pod"
                   || topology_name == "flex_rail2p"
                   || topology_name == "flex_cube") {
            plan_gpus = flex_gpus_per_host;
        } else if (topology_name == "ndv2_32") {
            plan_gpus = NDV2_32_GPUS_PER_CHASSIS;
        } else if (topology_name == "ndv2") {
            plan_gpus = 8;
        } else {
            plan_gpus = 1;  // clos/regular: no scale-up layer
        }
    }

    if (scheduler_algorithm == "mosaic_schedule") {
        double mosaic_nv = 0.0, mosaic_nic = 0.0;
        if (topology_name == "flex_rail" || topology_name == "flex_pod"
            || topology_name == "flex_rail2p") {
            mosaic_nv = flex_nv_gbps;
            mosaic_nic = flex_nic_gbps;
        } else if (topology_name == "dgx_rail") {
            mosaic_nv = DGX_RAIL_NV_GBPS;
            mosaic_nic = DGX_RAIL_NIC_GBPS;
        } else if (topology_name == "flex_cube") {
            // Scalar fallback = mean per-GPU B_in; the cube profile
            // installed below refines every per-dst computation.
            double sum = 0.0;
            for (uint32_t a = 0; a < flex_gpus_per_host; a++) {
                for (uint32_t b = 0; b < flex_gpus_per_host; b++) {
                    sum += FLEX_CUBE_LOCAL_GBPS[a][b];
                }
            }
            mosaic_nv = sum / flex_gpus_per_host;
            mosaic_nic = flex_nic_gbps;
        }
        if (mosaic_uniform_init) {
            mosaic_nv = 0.0;  // disables the waterfilling prior
        }
        auto mosaic_plan_start = std::chrono::steady_clock::now();
        uint32_t mosaic_wiring = SmartMosaicRuntime::WIRING_RAIL;
        if (topology_name == "flex_pod") {
            mosaic_wiring = SmartMosaicRuntime::WIRING_POD;
        } else if (topology_name == "flex_rail2p") {
            mosaic_wiring = SmartMosaicRuntime::WIRING_RAIL2P;
        } else if (topology_name == "flex_cube") {
            mosaic_wiring = SmartMosaicRuntime::WIRING_CUBE;
        }
        uint32_t mosaic_spines = mosaic_explicit_spine ? flex_spines : 0;
        // 32 组 guard: explicit pinning maps RX lane m to spine m % S, so
        // with only C-1 lanes a pair reaches at most C-1 of the S spines
        // and the effective leaf uplink collapses to (C-1)/S of the
        // installed one. Generation-time arithmetic => fall back to the
        // implicit landing-point binding instead of minting that cut.
        if (mosaic_spines > 0 && mosaic_corridors < mosaic_spines + 1) {
            cout << "mosaic_explicit_spine off guard reach corridors "
                 << mosaic_corridors << " lanes " << (mosaic_corridors - 1)
                 << " spines " << mosaic_spines << endl;
            mosaic_spines = 0;
        }
        mosaic_runtime = new SmartMosaicRuntime(
            graph, node_count, plan_gpus, demand, mosaic_waves,
            mosaic_gamma, mosaic_floor, mosaic_threshold, mosaic_adapt,
            mosaic_nv, mosaic_nic, mosaic_corridors, mosaic_wiring,
            mosaic_spines, mosaic_known_planeb);
        if (mosaic_ledger) {
            mosaic_runtime->set_ledger(flex_taper_gbps, flex_spines,
                                       !mosaic_ledger_raw);
            mosaic_runtime->set_ledger_sync(mosaic_ledger_sync,
                                            mosaic_sync_us);
            if (mosaic_lg_bcast || mosaic_lg_spineagg
                || mosaic_lg_fabmask || mosaic_lg_forkprobe) {
                mosaic_runtime->set_ledger_proto(
                    mosaic_lg_bcast, mosaic_lg_blag_us, mosaic_lg_blagw,
                    mosaic_lg_spineagg, mosaic_lg_fabmask,
                    mosaic_lg_forkprobe, mosaic_lg_stagger_us);
            }
        }
        if (topology_name == "flex_cube") {
            mosaic_runtime->set_cube_profile(FLEX_CUBE_LOCAL_GBPS,
                                             FLEX_CUBE_LOCAL_PATH);
            if (mosaic_cube_nic_cap) {
                mosaic_runtime->set_cube_nic_cap(true);
            }
            if (mosaic_cube_lane_rot) {
                mosaic_runtime->set_cube_lane_rot(true);
            }
            if (!mosaic_lp_intra_file.empty()) {
                // MOSAIC-LP table: "path <d> <j> <w> <v0> ... <vk>".
                vector<vector<vector<SmartMosaicRuntime::LpPath> > > table(
                    flex_gpus_per_host,
                    vector<vector<SmartMosaicRuntime::LpPath> >(
                        flex_gpus_per_host));
                std::ifstream lpf(mosaic_lp_intra_file.c_str());
                if (!lpf) {
                    cout << "cannot open mosaic_lp_intra file "
                         << mosaic_lp_intra_file << endl;
                    return 1;
                }
                string line;
                uint32_t lp_rows = 0;
                while (std::getline(lpf, line)) {
                    std::istringstream iss(line);
                    string kind;
                    if (!(iss >> kind) || kind != "path") {
                        continue;
                    }
                    uint32_t d, j;
                    SmartMosaicRuntime::LpPath p;
                    iss >> d >> j >> p.weight;
                    uint32_t v;
                    while (iss >> v) {
                        p.verts.push_back(v);
                    }
                    if (d >= flex_gpus_per_host || j >= flex_gpus_per_host
                        || p.verts.size() < 2 || p.verts.front() != j
                        || p.verts.back() != d) {
                        cout << "bad mosaic_lp_intra row: " << line << endl;
                        return 1;
                    }
                    table[j][d].push_back(p);
                    lp_rows++;
                }
                mosaic_runtime->set_lp_intra(table, mosaic_lp_local);
                cout << "mosaic_lp_intra " << mosaic_lp_intra_file
                     << " rows " << lp_rows
                     << " local " << (mosaic_lp_local ? "on" : "off")
                     << endl;
            }
        }
        static vector<vector<mem_b> > mosaic_prior;
        if (!mosaic_prior_demand_file.empty()) {
            mosaic_prior = load_demand_matrix(mosaic_prior_demand_file,
                                              node_count);
            mosaic_runtime->set_prior_demand(mosaic_prior);
        }
        if (mosaic_rx_init) {
            mosaic_runtime->set_rx_init();
        }
        if (mosaic_trace) {
            mosaic_runtime->set_trace(true);
        }
        if (mosaic_donate_window) {
            mosaic_runtime->set_donate_window(true);
        }
        if (mosaic_donate_rewaterfill) {
            mosaic_runtime->set_donate_rewaterfill(true);
        }
        if (mosaic_skew_epoch > 0) {
            mosaic_runtime->set_skew_epoch(mosaic_skew_epoch);
        }
        if (mosaic_pair_color == UINT32_MAX) {
            // 38 组统一方案自动门(auto 模式;显式 -mosaic_pair_color K
            // 不受限,诊断用):
            // 1) rail2p wiring:PC 结构性为负(34/35 组探针,双平面破坏
            //    Latin square 对齐)。
            // 2) lp_spread 表存在:jdist 落点摊布与 PC 重着色争 lane
            //    时序(38 组 cube mix 归因 +3.1%),LP 产物优先。
            if (mosaic_wiring == SmartMosaicRuntime::WIRING_RAIL2P) {
                cout << "mosaic_pair_color_auto off wiring rail2p" << endl;
            } else if (!mosaic_lp_spread_file.empty()) {
                cout << "mosaic_pair_color_auto off lp_spread" << endl;
            } else {
                mosaic_runtime->set_pair_color_auto();
            }
        } else if (mosaic_pair_color > 0) {
            mosaic_runtime->set_pair_color(mosaic_pair_color);
        }
        if (!mosaic_decision_trace.empty()) {
            mosaic_runtime->set_decision_trace(mosaic_decision_trace.c_str());
        }
        if (mosaic_guard > 0.0) {
            mosaic_runtime->set_lambda_guard(mosaic_guard);
        }
        if (mosaic_phase_plan) {
            mosaic_runtime->set_phase_plan(true);
        }
        if (mosaic_hop_progress) {
            mosaic_runtime->set_hop_progress_fix(true);
        }
        if (mosaic_wave_bytes > 0) {
            mosaic_runtime->set_wave_bytes(mosaic_wave_bytes,
                                           mosaic_wave_min);
        }
        if (mosaic_wave_align) {
            mosaic_runtime->set_wave_align(true);
        }
        if (mosaic_ramp_start > 0) {
            mosaic_runtime->set_ramp(mosaic_ramp_start, mosaic_ramp_event);
        }
        if (mosaic_ramp_cold > 0) {
            double ramp_taper = 0.0;  // 0 = unknown, gate 2 skipped
            if (topology_name == "flex_rail"
                || topology_name == "flex_cube") {
                ramp_taper = flex_taper_gbps;
            } else if (topology_name == "dgx_rail") {
                ramp_taper = dgx_taper_gbps;
            }
            mosaic_runtime->set_ramp_cold(mosaic_ramp_cold,
                                          mosaic_ramp_event, ramp_taper,
                                          mosaic_ramp_cold_force);
        }
        if (mosaic_cold_pace > 0) {
            double pace_taper = 0.0;  // 0 = unknown, gate 3 skipped
            if (topology_name == "flex_rail"
                || topology_name == "flex_cube") {
                pace_taper = flex_taper_gbps;
            } else if (topology_name == "dgx_rail") {
                pace_taper = dgx_taper_gbps;
            }
            mosaic_runtime->set_cold_pace(mosaic_cold_pace, pace_taper);
        }
        if (mosaic_host_lambda) {
            mosaic_runtime->set_host_lambda(true);
        }
        if (!mosaic_lp_spread_file.empty()) {
            if (topology_name != "flex_cube"
                && topology_name != "flex_rail") {
                cout << "-mosaic_lp_spread needs flex_cube or flex_rail"
                     << endl;
                return 1;
            }
            vector<vector<vector<pair<double, uint32_t> > > > jd(
                flex_gpus_per_host,
                vector<vector<pair<double, uint32_t> > >(
                    flex_gpus_per_host));
            std::ifstream jf(mosaic_lp_spread_file.c_str());
            if (!jf) {
                cout << "cannot open mosaic_lp_spread file "
                     << mosaic_lp_spread_file << endl;
                return 1;
            }
            string line;
            uint32_t jd_rows = 0;
            while (std::getline(jf, line)) {
                std::istringstream iss(line);
                string kind;
                if (!(iss >> kind) || kind != "jdist") {
                    continue;
                }
                uint32_t r, rd, j;
                double wgt;
                iss >> r >> rd >> j >> wgt;
                if (r >= flex_gpus_per_host || rd >= flex_gpus_per_host
                    || j >= flex_gpus_per_host || j == r || j == rd) {
                    cout << "bad mosaic_lp_spread row: " << line << endl;
                    return 1;
                }
                jd[r][rd].push_back(make_pair(wgt, j));
                jd_rows++;
            }
            mosaic_runtime->set_lp_spread(jd);
            cout << "mosaic_lp_spread " << mosaic_lp_spread_file
                 << " rows " << jd_rows << endl;
        }
        double mosaic_plan_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mosaic_plan_start).count();
        cout << "mosaic_plan gpus_per_server " << plan_gpus
             << " corridors " << mosaic_corridors
             << " plan_us " << mosaic_plan_us << endl;
        SmartNode::set_delivery_observer(mosaic_runtime);
        mosaic_runtime->start();
        run_mode = "mosaic_schedule";
    } else if (scheduler_algorithm == "fast_schedule") {
        uint32_t gpus = plan_gpus;

        auto plan_start = std::chrono::steady_clock::now();
        FastPlanBuilder builder(graph, node_count, gpus, demand);
        if (fast_alpha_beta_guard) {
            double intra_alpha_us = 9.0;
            double intra_gbps = DGX_RAIL_NV_GBPS;
            if (topology_name == "flex_rail"
                || topology_name == "flex_pod") {
                intra_gbps = flex_nv_gbps;
            } else if (topology_name == "flex_cube") {
                // FAST has no heterogeneous intra model; feed the guard
                // the mean cube link speed (documented approximation).
                double sum = 0.0;
                uint32_t cnt = 0;
                for (uint32_t a = 0; a < flex_gpus_per_host; a++) {
                    for (uint32_t b = a + 1; b < flex_gpus_per_host; b++) {
                        if (FLEX_CUBE_LOCAL_GBPS[a][b] > 0.0) {
                            sum += FLEX_CUBE_LOCAL_GBPS[a][b];
                            cnt++;
                        }
                    }
                }
                intra_alpha_us = flex_cube_link_us;
                intra_gbps = cnt ? sum / cnt : DGX_RAIL_NV_GBPS;
            } else if (topology_name == "ndv2_32") {
                intra_alpha_us = 0.7;
                intra_gbps = NDV2_32_NV_GBPS;
            }
            builder.enable_alpha_beta_guard(intra_alpha_us, intra_gbps);
        }
        if (fast_scaling > 0) {
            builder.set_scaling_unit(fast_scaling);
        }
        builder.set_stage_order_desc(fast_stage_order == "desc");
        SmartFastScheduleScheduler::round_robin_ref() = fast_rr_select;
        SmartTecclSchedule fast_plan = builder.build();
        double plan_us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - plan_start).count();
        cout << "fast_plan gpus_per_server " << gpus
             << " servers " << (node_count / gpus)
             << " stages " << builder.stage_count()
             << " steps " << fast_plan.steps.size()
             << " guard " << (fast_alpha_beta_guard ? "on" : "off")
             << " guard_skipped_pairs " << builder.guard_skipped_pairs()
             << " sync " << fast_sync
             << " stage_order " << fast_stage_order
             << " plan_us " << plan_us << endl;

        schedule_runtime = new SmartTecclScheduleRuntime(graph, fast_plan);
        if (fast_sync != "none") {
            schedule_runtime->enable_fast_sync(
                fast_sync == "credit"
                    ? SmartTecclScheduleRuntime::FAST_SYNC_CREDIT
                    : SmartTecclScheduleRuntime::FAST_SYNC_BARRIER,
                builder.step_tags(), builder.stage_count());
        }
        SmartNode::set_delivery_observer(schedule_runtime);
        schedule_runtime->start();
        expected_chunks = schedule_runtime->expected_chunks();
        expected_bytes = schedule_runtime->expected_bytes();
        flow_count = schedule_runtime->flow_count();
        run_mode = "fast_schedule";
    } else if (!schedule_file.empty() && scheduler_algorithm == "teccl_schedule") {
        SmartTecclSchedule schedule = load_teccl_schedule(schedule_file);
        schedule_runtime = new SmartTecclScheduleRuntime(graph, schedule);
        SmartNode::set_delivery_observer(schedule_runtime);
        schedule_runtime->start();
        expected_chunks = schedule_runtime->expected_chunks();
        expected_bytes = schedule_runtime->expected_bytes();
        flow_count = schedule_runtime->flow_count();
        run_mode = "teccl_schedule";
    } else if (!schedule_file.empty() && scheduler_algorithm == "syccl_schedule") {
        SmartSycclSchedule schedule = load_syccl_schedule(schedule_file);
        syccl_schedule_runtime = new SmartSycclScheduleRuntime(graph, schedule);
        SmartNode::set_delivery_observer(syccl_schedule_runtime);
        syccl_schedule_runtime->start();
        expected_chunks = syccl_schedule_runtime->expected_chunks();
        expected_bytes = syccl_schedule_runtime->expected_bytes();
        flow_count = syccl_schedule_runtime->flows().size();
        run_mode = "syccl_schedule";
    } else if (!schedule_file.empty() && scheduler_algorithm == "taccl_schedule") {
        SmartTacclSchedule schedule = load_taccl_schedule(schedule_file);
        taccl_schedule_runtime = new SmartTacclScheduleRuntime(graph, schedule);
        SmartNode::set_delivery_observer(taccl_schedule_runtime);
        taccl_schedule_runtime->start();
        expected_chunks = taccl_schedule_runtime->expected_chunks();
        expected_bytes = taccl_schedule_runtime->expected_bytes();
        flow_count = taccl_schedule_runtime->flows().size();
        run_mode = "taccl_schedule";
    } else {
        SmartWorkflow workflow;
        if (!demand.empty()) {
            workflow = build_workflow_from_demand(demand, node_count,
                                                  workflow_rounds);
        } else if (output_buffer_size > 0) {
            workflow = build_all_to_all_output_buffer_workflow(
                node_count, workflow_rounds, output_buffer_size);
            mem_b bytes_per_pair = (output_buffer_size + (node_count - 1) / 2)
                / (node_count - 1);
            if (!chunk_size_set) {
                chunk_size = bytes_per_pair;
            }
        } else {
            workflow = build_workflow(
                workload_name, node_count, workflow_rounds, all_to_all, src, dst,
                message_size);
        }
        workflow_runtime = new SmartWorkflowRuntime(graph, workflow, chunk_size);
        SmartNode::set_delivery_observer(workflow_runtime);
        workflow_runtime->start_ready_rounds();
        expected_chunks = workflow_runtime->expected_chunks();
        expected_bytes = workflow_runtime->expected_bytes();
        flow_count = workflow_runtime->flows().size();
    }

    if (mosaic_runtime) {
        // Waves are generated adaptively, so the expected totals are only
        // known once the runtime declares completion.
        while (!mosaic_runtime->complete()) {
            if (!eventlist.doNextEvent()) {
                cout << "stopped before mosaic completed" << endl;
                break;
            }
        }
        expected_chunks = mosaic_runtime->app_chains();
        expected_bytes = mosaic_runtime->expected_app_bytes();
        flow_count = mosaic_runtime->app_chains();
    }

    while (SmartNode::delivered_chunks() < expected_chunks) {
        if (!eventlist.doNextEvent()) {
            cout << "stopped before all chunks were delivered" << endl;
            break;
        }
    }

    while (SmartNode::ecn_feedback_arrived()
           < SmartNode::ecn_feedback_generated()) {
        if (!eventlist.doNextEvent()) {
            cout << "stopped before all ecn feedback returned" << endl;
            break;
        }
    }

    bool passed = SmartNode::delivered_chunks() == expected_chunks
        && SmartNode::delivered_bytes() == expected_bytes;
    double completion_time_us = (double)EventList::now() / 1000000.0;

    uint64_t queue_ecn_marks = 0;
    uint64_t queue_pause_events = 0;
    uint64_t queue_drops = 0;
    for (BaseQueue* queue : graph.queues) {
        SmartQueue* smart_queue = dynamic_cast<SmartQueue*>(queue);
        if (smart_queue) {
            queue_ecn_marks += smart_queue->ecn_marks();
            queue_pause_events += smart_queue->pause_events();
            queue_drops += smart_queue->num_drops();
        }
    }

    if (mosaic_runtime) {
        mosaic_runtime->print_report();
    } else if (schedule_runtime) {
        schedule_runtime->print_schedule_report();
    } else if (syccl_schedule_runtime) {
        syccl_schedule_runtime->print_schedule_report();
    } else if (taccl_schedule_runtime) {
        taccl_schedule_runtime->print_schedule_report();
    } else if (workflow_runtime) {
        workflow_runtime->print_round_report();
    }
    print_completion_cdf();

    cout << "validation "
         << (passed ? "PASSED" : "FAILED")
         << " rounds " << workflow_rounds
         << " completion_us " << completion_time_us
         << " flows " << flow_count
         << " expected_chunks " << expected_chunks
         << " delivered_chunks " << SmartNode::delivered_chunks()
         << " expected_bytes " << expected_bytes
         << " delivered_bytes " << SmartNode::delivered_bytes()
         << " sent_chunks " << SmartNode::sent_chunks()
         << " pooled_transit_chunks " << SmartNode::pooled_transit_chunks()
         << " queue_ecn_marks " << queue_ecn_marks
         << " queue_pause_events " << queue_pause_events
         << " queue_drops " << queue_drops
         << " split_minus " << split_minus
         << " split_zero " << split_zero
         << " split_plus " << split_plus
         << " split_minus_packets " << SmartNode::split_minus_packets()
         << " split_zero_packets " << SmartNode::split_zero_packets()
         << " split_plus_packets " << SmartNode::split_plus_packets()
         << " ecn_feedback_generated " << SmartNode::ecn_feedback_generated()
         << " ecn_feedback_forwarded " << SmartNode::ecn_feedback_forwarded()
         << " ecn_feedback_arrived " << SmartNode::ecn_feedback_arrived()
         << " ecn_rate_cuts " << SmartNode::ecn_rate_cuts()
         << " scheduler " << scheduler_algorithm
         << " forwarding " << (SmartNode::store_forward_enabled()
                                ? "store_forward" : "cut_through")
         << " workload " << workload_name
         << " topology " << topology_name
         << " mode " << run_mode
         << " schedule_file " << (schedule_file.empty() ? "none" : schedule_file)
         << endl;

    return passed ? 0 : 2;
}
