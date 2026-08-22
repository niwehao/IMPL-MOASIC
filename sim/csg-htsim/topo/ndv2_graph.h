// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef NDV2_GRAPH_H
#define NDV2_GRAPH_H

#include "regular_graph.h"
#include "dgx_rail_graph.h"

#include <map>
#include <stdexcept>

static const uint32_t NDV2_LIKE_NODE_COUNT = 16;

static double ndv2_like_code_to_gbps(int code) {
    if (code == 23) {
        return 400.0;  // 50 GB/s
    }
    if (code == 46) {
        return 200.0;  // 25 GB/s
    }
    if (code == 107) {
        return 100.0;  // 12.5 GB/s
    }
    return 0.0;
}

static simtime_picosec ndv2_like_code_to_delay(int code) {
    if (code == 107) {
        return timeFromUs(1.3);
    }
    if (code == 23 || code == 46) {
        return timeFromUs(0.7);
    }
    return 0;
}

static vector<vector<int> > ndv2_like_raw_capacity_codes() {
    int rows[16][16] = {
        {0, 23, 46, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {23, 0, 46, 23, 0, 46, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0},
        {46, 46, 0, 23, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {46, 23, 23, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0},
        {23, 0, 0, 0, 0, 23, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 46, 0, 0, 23, 0, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 23, 0, 46, 46, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 46, 46, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 46, 46, 23, 0, 0, 0},
        {107, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 23, 0, 46, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 23, 0, 0, 23, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 46, 23, 23, 0, 0, 0, 0, 46},
        {0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 23, 46, 46},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 23, 0, 46, 23},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 46, 0, 23},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 23, 23, 0}
    };

    vector<vector<int> > matrix(NDV2_LIKE_NODE_COUNT,
                                vector<int>(NDV2_LIKE_NODE_COUNT, 0));
    for (uint32_t i = 0; i < NDV2_LIKE_NODE_COUNT; i++) {
        for (uint32_t j = 0; j < NDV2_LIKE_NODE_COUNT; j++) {
            matrix[j][i] = rows[i][j];
        }
    }
    return matrix;
}

// ---------------------------------------------------------------------------
// ndv2_32: 32-GPU NDv2-like pod that ALL FOUR baselines can run on
// (static_ecmp / TE-CCL / TACCL / SyCCL).
//
// The 16-GPU ndv2 above is TE-CCL-exclusive: heterogeneous 400/200/100G
// links and sparse point-to-point cross-chassis wiring are outside TACCL's
// replicated-host model and SyCCL's layered config (see
// Note/2026-0708-solver-topology-capabilities.md). To land in the three solvers'
// common set, ndv2_32 makes two deliberate compromises:
//   1. intra-chassis NVLink is a UNIFORM 200 Gbps (SyCCL allows one
//      link_spec per layer); the hybrid cube-mesh ADJACENCY of NDv2 is
//      kept (each GPU has 4 NVLink neighbors out of 7).
//   2. cross-chassis wiring goes NIC -> IB switch -> NIC (SyCCL's only
//      inter-host mechanism) instead of sparse direct links: 2x 100G IB
//      NICs per chassis, GPU 0-3 -> NIC0, GPU 4-7 -> NIC1 (SyCCL's
//      contiguous block mapping), one non-blocking IB switch.
//
// htsim model: GPUs are the only SmartNodes. Intra-chassis NVLinks are
// point-to-point routes (one pacing port per link, like real NVLink
// serializers). Cross-chassis logical edges share the per-NIC uplink and
// downlink queues (switch-as-wire), so the 2x100G chassis egress cut is
// modeled by queue contention. One IB hop costs 1.3us per segment
// (up + down = 2.6us GPU to GPU), matching the 16-GPU ndv2 alpha flavor.
//
// Note: with a symmetric A2A workload every scheme is bound by the same
// chassis-egress cut, so unlike dgx_rail this topology is for common-set
// executability, not for showcasing solver-vs-ECMP gaps.
// ---------------------------------------------------------------------------

static const uint32_t NDV2_32_CHASSIS = 4;
static const uint32_t NDV2_32_GPUS_PER_CHASSIS = 8;
static const uint32_t NDV2_32_NODE_COUNT =
    NDV2_32_CHASSIS * NDV2_32_GPUS_PER_CHASSIS;
static const uint32_t NDV2_32_NICS_PER_CHASSIS = 2;
static const uint32_t NDV2_32_NIC_COUNT =
    NDV2_32_CHASSIS * NDV2_32_NICS_PER_CHASSIS;
static const uint32_t NDV2_32_PORT_IB = 100;  // distinct from GPU ids 0..31

static const double NDV2_32_NV_GBPS = 200.0;
static const double NDV2_32_IB_GBPS = 100.0;

static uint32_t ndv2_32_chassis(uint32_t gpu) {
    return gpu / NDV2_32_GPUS_PER_CHASSIS;
}

static uint32_t ndv2_32_nic(uint32_t gpu) {
    uint32_t local = gpu % NDV2_32_GPUS_PER_CHASSIS;
    uint32_t gpus_per_nic = NDV2_32_GPUS_PER_CHASSIS / NDV2_32_NICS_PER_CHASSIS;
    return ndv2_32_chassis(gpu) * NDV2_32_NICS_PER_CHASSIS
        + local / gpus_per_nic;
}

// NDv2 hybrid cube-mesh adjacency (local ids), from the 8x8 block of the
// 16-GPU capacity matrix above with speeds flattened to one class.
static const uint32_t NDV2_32_CUBE_EDGES[16][2] = {
    {0, 1}, {0, 2}, {0, 3}, {0, 4},
    {1, 2}, {1, 3}, {1, 5},
    {2, 3}, {2, 6},
    {3, 7},
    {4, 5}, {4, 6}, {4, 7},
    {5, 6}, {5, 7},
    {6, 7}
};

static SmartGraph build_ndv2_32_graph(EventList& eventlist,
                                      Logfile& logfile) {
    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = 1024LL * 1024 * 1024;
    const simtime_picosec nv_delay = timeFromUs(0.7);
    const simtime_picosec ib_delay = timeFromUs(1.3);

    SmartGraph graph;
    graph.adjacency.resize(NDV2_32_NODE_COUNT);

    for (uint32_t id = 0; id < NDV2_32_NODE_COUNT; id++) {
        graph.nodes.push_back(new SmartNode(
            id, NDV2_32_NODE_COUNT, eventlist,
            speedFromGbps(NDV2_32_NV_GBPS)));
    }

    // Intra-chassis NVLinks: point-to-point edges, one pacing port each
    // (add_graph_edge defaults port == neighbor).
    uint64_t nv_edges = 0;
    for (uint32_t chassis = 0; chassis < NDV2_32_CHASSIS; chassis++) {
        uint32_t base = chassis * NDV2_32_GPUS_PER_CHASSIS;
        for (uint32_t e = 0; e < 16; e++) {
            add_graph_edge(graph, eventlist, logfile,
                           base + NDV2_32_CUBE_EDGES[e][0],
                           base + NDV2_32_CUBE_EDGES[e][1],
                           speedFromGbps(NDV2_32_NV_GBPS), egress_buffer,
                           nv_delay);
            nv_edges++;
        }
    }

    // Cross-chassis fabric: shared per-NIC uplink/downlink segments through
    // one non-blocking IB switch (switch-as-wire).
    vector<DgxRailSegment> nic_up(NDV2_32_NIC_COUNT);
    vector<DgxRailSegment> nic_down(NDV2_32_NIC_COUNT);
    for (uint32_t nic = 0; nic < NDV2_32_NIC_COUNT; nic++) {
        nic_up[nic] = make_dgx_rail_segment(
            graph, eventlist, logfile, NDV2_32_IB_GBPS, fabric_buffer,
            ib_delay, "ndv2_32_nic_up_" + ntoa(nic));
        nic_down[nic] = make_dgx_rail_segment(
            graph, eventlist, logfile, NDV2_32_IB_GBPS, fabric_buffer,
            ib_delay, "ndv2_32_nic_down_" + ntoa(nic));
    }

    uint64_t ib_routes = 0;
    for (uint32_t src = 0; src < NDV2_32_NODE_COUNT; src++) {
        for (uint32_t dst = 0; dst < NDV2_32_NODE_COUNT; dst++) {
            if (src == dst || ndv2_32_chassis(src) == ndv2_32_chassis(dst)) {
                continue;
            }

            Route* route = new Route();
            append_dgx_rail_segment(route, nic_up[ndv2_32_nic(src)]);
            append_dgx_rail_segment(route, nic_down[ndv2_32_nic(dst)]);
            route->push_back(graph.nodes[dst]);
            graph.routes.push_back(route);
            graph.adjacency[src].push_back(dst);
            graph.nodes[src]->set_neighbor_rate(
                dst, speedFromGbps(NDV2_32_IB_GBPS));
            graph.nodes[src]->add_route(dst, route, NDV2_32_PORT_IB,
                                        speedFromGbps(NDV2_32_IB_GBPS));
            ib_routes++;
        }
    }

    vector<vector<int> > distances = graph_all_pair_distances(graph);
    install_forwarding_choices(graph, distances);
    uint32_t diameter = graph_diameter_from_distances(distances);

    cout << "built ndv2_32 graph with " << NDV2_32_NODE_COUNT
         << " GPU nodes (" << NDV2_32_CHASSIS << " chassis x "
         << NDV2_32_GPUS_PER_CHASSIS << " GPUs, cube-mesh intra),"
         << " nv_edges " << nv_edges
         << " ib_routes " << ib_routes
         << ", diameter " << diameter
         << ", nvlink " << NDV2_32_NV_GBPS << "Gbps/0.7us"
         << ", ib " << NDV2_32_IB_GBPS << "Gbps/1.3us_per_segment ("
         << NDV2_32_NICS_PER_CHASSIS << " NICs/chassis)" << endl;

    return graph;
}

static SmartGraph build_ndv2_like_graph(EventList& eventlist,
                                        Logfile& logfile,
                                        mem_b buffer,
                                        simtime_picosec delay) {
    (void)delay;
    SmartGraph graph;
    graph.adjacency.resize(NDV2_LIKE_NODE_COUNT);

    linkspeed_bps default_speed = speedFromGbps(400.0);
    for (uint32_t id = 0; id < NDV2_LIKE_NODE_COUNT; id++) {
        graph.nodes.push_back(new SmartNode(id, NDV2_LIKE_NODE_COUNT,
                                            eventlist, default_speed));
    }

    vector<vector<int> > capacity = ndv2_like_raw_capacity_codes();
    uint64_t edge_count = 0;
    map<int, uint64_t> code_counts;
    map<int, uint64_t> delay_counts;
    for (uint32_t left = 0; left < NDV2_LIKE_NODE_COUNT; left++) {
        for (uint32_t right = left + 1; right < NDV2_LIKE_NODE_COUNT; right++) {
            int left_to_right = capacity[left][right];
            int right_to_left = capacity[right][left];
            if (left_to_right == 0 && right_to_left == 0) {
                continue;
            }

            int code = 0;
            if (left_to_right != 0 && right_to_left != 0) {
                code = left_to_right > right_to_left ? right_to_left : left_to_right;
            } else {
                code = left_to_right != 0 ? left_to_right : right_to_left;
            }
            double gbps = ndv2_like_code_to_gbps(code);
            if (gbps <= 0.0) {
                throw runtime_error("invalid NDv2-like capacity code");
            }

            add_graph_edge(graph, eventlist, logfile, left, right,
                           speedFromGbps(gbps), buffer,
                           ndv2_like_code_to_delay(code));
            edge_count++;
            code_counts[code]++;
            delay_counts[code]++;
        }
    }

    vector<vector<int> > distances = graph_all_pair_distances(graph);
    install_forwarding_choices(graph, distances);
    uint32_t diameter = graph_diameter_from_distances(distances);

    cout << "built NDv2-like no-switch graph with "
         << NDV2_LIKE_NODE_COUNT << " GPU nodes, "
         << edge_count << " weighted edges, diameter " << diameter
         << ", rates: 400Gbps_edges " << code_counts[23]
         << ", 200Gbps_edges " << code_counts[46]
         << ", 100Gbps_edges " << code_counts[107]
         << ", alpha_0.7us_edges " << delay_counts[23] + delay_counts[46]
         << ", alpha_1.3us_edges " << delay_counts[107] << endl;

    return graph;
}

#endif
