// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef DGX_RAIL_GRAPH_H
#define DGX_RAIL_GRAPH_H

#include "regular_graph.h"

#include <stdexcept>

// Rail-optimized GPU pod (DGX/SuperPOD style):
//   - DGX_RAIL_HOSTS hosts, each with DGX_RAIL_GPUS_PER_HOST GPUs behind a
//     non-blocking NVSwitch (scale-up).
//   - One NIC per GPU. Rail r = the r-th NIC of every host, all wired to
//     leaf switch r (scale-out). Cross-host same-rail traffic never leaves
//     its leaf; cross-rail traffic must cross the (tapered) spine.
//   - Only GPUs are SmartNodes. NVSwitch/NIC/leaf/spine are modeled as
//     htsim queue+pipe segments shared between routes (switch-as-wire),
//     matching how TE-CCL (buffer==0 switches), TACCL and SyCCL treat
//     switches: contention and latency, but no store-and-forward hop.
//
// Rates/latencies follow the SyCCL A100 profile: NVLink 2400 Gbps / 9 us,
// NIC+leaf 180 Gbps / 25 us per segment. The leaf->spine taper is the
// experiment knob (default 180 Gbps, i.e. 4:1 oversubscription per leaf:
// 4 host NICs x 180G feed one 180G uplink).

static const uint32_t DGX_RAIL_HOSTS = 4;
static const uint32_t DGX_RAIL_GPUS_PER_HOST = 8;
static const uint32_t DGX_RAIL_NODE_COUNT =
    DGX_RAIL_HOSTS * DGX_RAIL_GPUS_PER_HOST;

static const uint32_t DGX_RAIL_PORT_NV = 0;
static const uint32_t DGX_RAIL_PORT_NIC = 1;

static const double DGX_RAIL_NV_GBPS = 2400.0;
static const double DGX_RAIL_NIC_GBPS = 180.0;

static uint32_t dgx_rail_host(uint32_t gpu) {
    return gpu / DGX_RAIL_GPUS_PER_HOST;
}

static uint32_t dgx_rail_rail(uint32_t gpu) {
    return gpu % DGX_RAIL_GPUS_PER_HOST;
}

struct DgxRailSegment {
    SmartQueue* queue;
    Pipe* pipe;
};

static DgxRailSegment make_dgx_rail_segment(SmartGraph& graph,
                                            EventList& eventlist,
                                            Logfile& logfile,
                                            double gbps, mem_b buffer,
                                            simtime_picosec delay,
                                            const string& name) {
    DgxRailSegment segment;
    segment.queue = new SmartQueue(speedFromGbps(gbps), buffer, eventlist,
                                   NULL);
    segment.queue->setName(name + "_queue");
    logfile.writeName(*segment.queue);
    graph.queues.push_back(segment.queue);

    segment.pipe = new Pipe(delay, eventlist);
    segment.pipe->setName(name + "_pipe");
    logfile.writeName(*segment.pipe);
    graph.pipes.push_back(segment.pipe);
    return segment;
}

static void append_dgx_rail_segment(Route* route,
                                    const DgxRailSegment& segment) {
    route->push_back(segment.queue);
    route->push_back(segment.pipe);
}

static SmartGraph build_dgx_rail_graph(EventList& eventlist,
                                       Logfile& logfile,
                                       double taper_gbps) {
    if (taper_gbps <= 0.0) {
        throw runtime_error("dgx_rail taper_gbps must be positive");
    }

    // Buffers are deliberately generous: schedules must never lose packets
    // (there is no retransmission), and pool-side pacing keeps the sender
    // queues shallow anyway. Fabric queues absorb transient incast.
    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = 1024LL * 1024 * 1024;

    SmartGraph graph;
    graph.adjacency.resize(DGX_RAIL_NODE_COUNT);

    for (uint32_t id = 0; id < DGX_RAIL_NODE_COUNT; id++) {
        graph.nodes.push_back(new SmartNode(
            id, DGX_RAIL_NODE_COUNT, eventlist,
            speedFromGbps(DGX_RAIL_NV_GBPS)));
    }

    // Per-GPU segments.
    vector<DgxRailSegment> nv_up(DGX_RAIL_NODE_COUNT);
    vector<DgxRailSegment> nv_down(DGX_RAIL_NODE_COUNT);
    vector<DgxRailSegment> nic_up(DGX_RAIL_NODE_COUNT);
    vector<DgxRailSegment> leaf_down(DGX_RAIL_NODE_COUNT);
    // Per-rail (leaf) segments.
    vector<DgxRailSegment> leaf_up(DGX_RAIL_GPUS_PER_HOST);
    vector<DgxRailSegment> spine_down(DGX_RAIL_GPUS_PER_HOST);

    for (uint32_t g = 0; g < DGX_RAIL_NODE_COUNT; g++) {
        // GPU -> NVSwitch (all NVLink latency charged on the way up).
        nv_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, DGX_RAIL_NV_GBPS, egress_buffer,
            timeFromUs(9.0), "nv_up_" + ntoa(g));
        // NVSwitch egress port -> GPU (contention point for intra-host
        // incast).
        nv_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, DGX_RAIL_NV_GBPS, fabric_buffer,
            timeFromUs(0.0), "nv_down_" + ntoa(g));
        // GPU -> own NIC -> leaf ingress.
        nic_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, DGX_RAIL_NIC_GBPS, egress_buffer,
            timeFromUs(25.0), "nic_up_" + ntoa(g));
        // Leaf egress -> NIC -> GPU (contention point for rail incast).
        leaf_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, DGX_RAIL_NIC_GBPS, fabric_buffer,
            timeFromUs(25.0), "leaf_down_" + ntoa(g));
    }

    for (uint32_t r = 0; r < DGX_RAIL_GPUS_PER_HOST; r++) {
        leaf_up[r] = make_dgx_rail_segment(
            graph, eventlist, logfile, taper_gbps, fabric_buffer,
            timeFromUs(25.0), "leaf_up_" + ntoa(r));
        spine_down[r] = make_dgx_rail_segment(
            graph, eventlist, logfile, taper_gbps, fabric_buffer,
            timeFromUs(25.0), "spine_down_" + ntoa(r));
    }

    uint64_t nv_routes = 0;
    uint64_t rail_routes = 0;
    uint64_t spine_routes = 0;
    for (uint32_t src = 0; src < DGX_RAIL_NODE_COUNT; src++) {
        for (uint32_t dst = 0; dst < DGX_RAIL_NODE_COUNT; dst++) {
            if (src == dst) {
                continue;
            }

            Route* route = new Route();
            linkspeed_bps logical_rate;
            uint32_t port;
            linkspeed_bps port_rate;

            if (dgx_rail_host(src) == dgx_rail_host(dst)) {
                append_dgx_rail_segment(route, nv_up[src]);
                append_dgx_rail_segment(route, nv_down[dst]);
                logical_rate = speedFromGbps(DGX_RAIL_NV_GBPS);
                port = DGX_RAIL_PORT_NV;
                port_rate = speedFromGbps(DGX_RAIL_NV_GBPS);
                nv_routes++;
            } else if (dgx_rail_rail(src) == dgx_rail_rail(dst)) {
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                logical_rate = speedFromGbps(DGX_RAIL_NIC_GBPS);
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(DGX_RAIL_NIC_GBPS);
                rail_routes++;
            } else {
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(route, leaf_up[dgx_rail_rail(src)]);
                append_dgx_rail_segment(route,
                                        spine_down[dgx_rail_rail(dst)]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                double bottleneck_gbps = DGX_RAIL_NIC_GBPS < taper_gbps
                    ? DGX_RAIL_NIC_GBPS : taper_gbps;
                logical_rate = speedFromGbps(bottleneck_gbps);
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(DGX_RAIL_NIC_GBPS);
                spine_routes++;
            }

            route->push_back(graph.nodes[dst]);
            graph.routes.push_back(route);
            graph.adjacency[src].push_back(dst);
            graph.nodes[src]->set_neighbor_rate(dst, logical_rate);
            graph.nodes[src]->add_route(dst, route, port, port_rate);
        }
    }

    vector<vector<int> > distances = graph_all_pair_distances(graph);
    install_forwarding_choices(graph, distances);

    cout << "built dgx_rail graph with " << DGX_RAIL_NODE_COUNT
         << " GPU nodes (" << DGX_RAIL_HOSTS << " hosts x "
         << DGX_RAIL_GPUS_PER_HOST << " GPUs), switch-as-wire fabric,"
         << " nv_routes " << nv_routes
         << " rail_routes " << rail_routes
         << " spine_routes " << spine_routes
         << ", nvlink " << DGX_RAIL_NV_GBPS << "Gbps/9us"
         << ", nic+leaf " << DGX_RAIL_NIC_GBPS << "Gbps/25us"
         << ", leaf_up taper " << taper_gbps << "Gbps/25us" << endl;

    return graph;
}

#endif
