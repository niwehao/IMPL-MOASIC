// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef CLOS_GRAPH_H
#define CLOS_GRAPH_H

#include "regular_graph.h"
#include "dgx_rail_graph.h"

#include <algorithm>
#include <stdexcept>

// Pure two-layer clos: hosts have NO scale-up dimension, every byte crosses
// the switch fabric. Rewritten 2026-07-08 as switch-as-wire: only hosts are
// SmartNodes; leaf/spine live on as shared queue+pipe segments (contention
// and latency, no store-and-forward hop), matching how TE-CCL/TACCL/SyCCL
// model switches.
//
// Structure: leaf l serves the contiguous host block [l*hosts_per_leaf, ...);
// every leaf connects to every spine at `uplink_speed` per link. Same-leaf
// traffic: host -> leaf -> host (2 segments). Cross-leaf: host -> leaf ->
// spine -> leaf -> host (4 segments), spine pinned per (src+dst) % spines
// (fabric-level static ECMP baked into the route; schedules cannot choose
// the spine — with the SyCCL-compatible default of 1 spine this is moot).
//
// Note for experiment design: each host has exactly one uplink (degree 1),
// so under a symmetric A2A every scheme carries identical host-uplink load
// and relaying cannot bypass anything. This is an executability/consistency
// topology, not a solver-advantage topology (see
// Note/solver_topology_capabilities.md).

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

static uint32_t default_clos_count(uint32_t host_count) {
    uint32_t count = (uint32_t)ceil(sqrt((double)host_count));
    return count == 0 ? 1 : count;
}

static const uint32_t CLOS_PORT_UP = 0;  // each host has a single uplink port

static SmartGraph build_two_layer_clos_graph(uint32_t host_count,
                                             uint32_t leaf_count,
                                             uint32_t spine_count,
                                             EventList& eventlist,
                                             Logfile& logfile,
                                             linkspeed_bps speed,
                                             linkspeed_bps uplink_speed,
                                             simtime_picosec delay,
                                             uint32_t fabric_buffer_mb = 1024) {
    if (host_count == 0) {
        throw runtime_error("host count must be greater than zero");
    }
    if (leaf_count == 0) {
        leaf_count = default_clos_count(host_count);
    }
    if (spine_count == 0) {
        spine_count = 1;
    }

    uint32_t hosts_per_leaf = ceil_div_u32(host_count, leaf_count);
    if (uplink_speed == 0) {
        // Default 2:1 oversubscription per leaf, split across spine links.
        uplink_speed = (linkspeed_bps)(
            (double)speed * hosts_per_leaf / (2.0 * spine_count));
    }

    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = (mem_b)fabric_buffer_mb * 1024 * 1024;

    SmartGraph graph;
    graph.adjacency.resize(host_count);
    for (uint32_t id = 0; id < host_count; id++) {
        graph.nodes.push_back(new SmartNode(id, host_count, eventlist,
                                            speed));
    }

    double speed_gbps = (double)speed / 1e9;
    double uplink_gbps = (double)uplink_speed / 1e9;
    double delay_us = (double)delay / 1e6;

    vector<DgxRailSegment> host_up(host_count);
    vector<DgxRailSegment> leaf_down(host_count);
    // leaf_up[l][s]: leaf l -> spine s; spine_down[s][l]: spine s -> leaf l.
    vector<vector<DgxRailSegment> > leaf_up(
        leaf_count, vector<DgxRailSegment>(spine_count));
    vector<vector<DgxRailSegment> > spine_down(
        spine_count, vector<DgxRailSegment>(leaf_count));

    for (uint32_t h = 0; h < host_count; h++) {
        host_up[h] = make_dgx_rail_segment(
            graph, eventlist, logfile, speed_gbps, egress_buffer, delay,
            "clos_host_up_" + ntoa(h));
        leaf_down[h] = make_dgx_rail_segment(
            graph, eventlist, logfile, speed_gbps, fabric_buffer, delay,
            "clos_leaf_down_" + ntoa(h));
    }
    for (uint32_t l = 0; l < leaf_count; l++) {
        for (uint32_t s = 0; s < spine_count; s++) {
            leaf_up[l][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, uplink_gbps, fabric_buffer, delay,
                "clos_leaf_up_" + ntoa(l) + "_" + ntoa(s));
            spine_down[s][l] = make_dgx_rail_segment(
                graph, eventlist, logfile, uplink_gbps, fabric_buffer, delay,
                "clos_spine_down_" + ntoa(s) + "_" + ntoa(l));
        }
    }

    uint64_t leaf_routes = 0;
    uint64_t spine_routes = 0;
    for (uint32_t src = 0; src < host_count; src++) {
        uint32_t src_leaf = min(src / hosts_per_leaf, leaf_count - 1);
        for (uint32_t dst = 0; dst < host_count; dst++) {
            if (src == dst) {
                continue;
            }
            uint32_t dst_leaf = min(dst / hosts_per_leaf, leaf_count - 1);

            Route* route = new Route();
            append_dgx_rail_segment(route, host_up[src]);
            if (src_leaf != dst_leaf) {
                uint32_t spine = (src + dst) % spine_count;
                append_dgx_rail_segment(route, leaf_up[src_leaf][spine]);
                append_dgx_rail_segment(route, spine_down[spine][dst_leaf]);
                spine_routes++;
            } else {
                leaf_routes++;
            }
            append_dgx_rail_segment(route, leaf_down[dst]);
            route->push_back(graph.nodes[dst]);
            graph.routes.push_back(route);
            graph.adjacency[src].push_back(dst);
            graph.nodes[src]->set_neighbor_rate(
                dst, src_leaf == dst_leaf
                    ? speed
                    : (speed < uplink_speed ? speed : uplink_speed));
            graph.nodes[src]->add_route(dst, route, CLOS_PORT_UP, speed);
        }
    }

    vector<vector<int> > distances = graph_all_pair_distances(graph);
    install_forwarding_choices(graph, distances);

    cout << "built two-layer clos (switch-as-wire) with "
         << host_count << " hosts, " << leaf_count << " leaves, "
         << spine_count << " spines, leaf_routes " << leaf_routes
         << " spine_routes " << spine_routes
         << ", host_link " << speed_gbps << "Gbps"
         << ", leaf_uplink " << uplink_gbps << "Gbps/link"
         << " (oversub " << ((double)hosts_per_leaf * speed_gbps
                             / (uplink_gbps * spine_count)) << ":1)"
         << ", delay_us_per_segment " << delay_us << endl;

    return graph;
}

#endif
