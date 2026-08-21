// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef FLEX_POD_GRAPH_H
#define FLEX_POD_GRAPH_H

#include "dgx_rail_graph.h"

#include <algorithm>
#include <stdexcept>

// flex_rail: knob-generalized rail-optimized pod (MOSAIC exploration
// topology X1, Note/idea_moe_subschedule_assembly.md §4.2).
//   - FLEX hosts x gpus_per_host, NVSwitch scale-up at nv_gbps (the X1
//     knob: 2400 = dgx_rail's A100 profile, 180 = PCIe-class 1:1 with the
//     NIC, 90 = 0.5:1).
//   - One NIC per GPU at nic_gbps, rail-wired to gpus_per_host leaf
//     switches; leaf uplinks total taper_gbps per leaf, split evenly over
//     `spines` spine switches (cross-rail route pinned to spine
//     (src+dst) % spines, fabric-level static ECMP like clos_graph).
//   - Same switch-as-wire segment structure and latencies as dgx_rail
//     (NVLink 9us, NIC/leaf/spine segments 25us), so results are directly
//     comparable: build_dgx_rail_graph == flex_rail(4, 8, 2400, 180, 180, 1).

static uint32_t FLEX_RAIL_HOSTS = 4;
static uint32_t FLEX_RAIL_GPUS_PER_HOST = 8;

static uint32_t flex_rail_host(uint32_t gpu) {
    return gpu / FLEX_RAIL_GPUS_PER_HOST;
}

static uint32_t flex_rail_rail(uint32_t gpu) {
    return gpu % FLEX_RAIL_GPUS_PER_HOST;
}

static SmartGraph build_flex_rail_graph(EventList& eventlist,
                                        Logfile& logfile,
                                        uint32_t hosts,
                                        uint32_t gpus_per_host,
                                        double nv_gbps,
                                        double nic_gbps,
                                        double taper_gbps,
                                        uint32_t spine_count,
                                        mem_b fabric_buffer_mb = 1024,
                                        uint32_t degrade_gpu = UINT32_MAX,
                                        double degrade_factor = 1.0,
                                        uint32_t degrade_dir = 0,
                                        const vector<char>& degrade_mask =
                                            vector<char>()) {
    if (hosts == 0 || gpus_per_host == 0) {
        throw runtime_error("flex_rail: hosts and gpus_per_host must be positive");
    }
    if (nv_gbps <= 0.0 || nic_gbps <= 0.0 || taper_gbps <= 0.0) {
        throw runtime_error("flex_rail: link rates must be positive");
    }
    if (spine_count == 0) {
        spine_count = 1;
    }
    FLEX_RAIL_HOSTS = hosts;
    FLEX_RAIL_GPUS_PER_HOST = gpus_per_host;
    uint32_t node_count = hosts * gpus_per_host;
    double spine_gbps = taper_gbps / spine_count;

    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = fabric_buffer_mb * 1024 * 1024;

    SmartGraph graph;
    graph.adjacency.resize(node_count);
    for (uint32_t id = 0; id < node_count; id++) {
        graph.nodes.push_back(new SmartNode(id, node_count, eventlist,
                                            speedFromGbps(nv_gbps)));
    }

    // Per-GPU segments (same roles/latencies as dgx_rail).
    vector<DgxRailSegment> nv_up(node_count);
    vector<DgxRailSegment> nv_down(node_count);
    vector<DgxRailSegment> nic_up(node_count);
    vector<DgxRailSegment> leaf_down(node_count);
    // Per (rail, spine) fabric segments.
    vector<vector<DgxRailSegment> > leaf_up(
        gpus_per_host, vector<DgxRailSegment>(spine_count));
    vector<vector<DgxRailSegment> > spine_down(
        gpus_per_host, vector<DgxRailSegment>(spine_count));

    if (degrade_factor < 1.0) {
        throw runtime_error("flex_rail: degrade_factor must be >= 1");
    }
    // Unannounced interference model: a set of GPUs' NICs run at
    // nic/degrade_factor. Schedulers are NOT told -- static plans keep
    // assuming the nominal rate; adaptive ones must discover it online.
    // 32 组 instruments: degrade_dir picks the slowed half (0 = both, the
    // 19 组 behavior; 1 = egress/TX only; 2 = ingress/RX only), and
    // degrade_mask degrades a SET of GPUs (oracle throttling experiments).
    auto is_degraded = [&](uint32_t g) {
        return g == degrade_gpu
            || (g < degrade_mask.size() && degrade_mask[g]);
    };
    auto nic_tx_of = [&](uint32_t g) {
        return (is_degraded(g) && degrade_dir != 2)
            ? nic_gbps / degrade_factor : nic_gbps;
    };
    auto nic_rx_of = [&](uint32_t g) {
        return (is_degraded(g) && degrade_dir != 1)
            ? nic_gbps / degrade_factor : nic_gbps;
    };

    for (uint32_t g = 0; g < node_count; g++) {
        nv_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, egress_buffer,
            timeFromUs(9.0), "flex_nv_up_" + ntoa(g));
        nv_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, fabric_buffer,
            timeFromUs(0.0), "flex_nv_down_" + ntoa(g));
        nic_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_tx_of(g), egress_buffer,
            timeFromUs(25.0), "flex_nic_up_" + ntoa(g));
        leaf_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_rx_of(g), fabric_buffer,
            timeFromUs(25.0), "flex_leaf_down_" + ntoa(g));
    }
    for (uint32_t r = 0; r < gpus_per_host; r++) {
        for (uint32_t s = 0; s < spine_count; s++) {
            leaf_up[r][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0), "flex_leaf_up_" + ntoa(r) + "_" + ntoa(s));
            spine_down[r][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0),
                "flex_spine_down_" + ntoa(r) + "_" + ntoa(s));
        }
    }

    uint64_t nv_routes = 0;
    uint64_t rail_routes = 0;
    uint64_t spine_routes = 0;
    for (uint32_t src = 0; src < node_count; src++) {
        for (uint32_t dst = 0; dst < node_count; dst++) {
            if (src == dst) {
                continue;
            }

            Route* route = new Route();
            linkspeed_bps logical_rate;
            uint32_t port;
            linkspeed_bps port_rate;

            if (src / gpus_per_host == dst / gpus_per_host) {
                append_dgx_rail_segment(route, nv_up[src]);
                append_dgx_rail_segment(route, nv_down[dst]);
                logical_rate = speedFromGbps(nv_gbps);
                port = DGX_RAIL_PORT_NV;
                port_rate = speedFromGbps(nv_gbps);
                nv_routes++;
            } else if (src % gpus_per_host == dst % gpus_per_host) {
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                logical_rate =
                    speedFromGbps(min(nic_tx_of(src), nic_rx_of(dst)));
                port = DGX_RAIL_PORT_NIC;
                // The host paces its own (possibly degraded) NIC; only the
                // plan-time schedulers are blind to the degradation.
                port_rate = speedFromGbps(nic_tx_of(src));
                rail_routes++;
            } else {
                uint32_t spine = (src + dst) % spine_count;
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(
                    route, leaf_up[src % gpus_per_host][spine]);
                append_dgx_rail_segment(
                    route, spine_down[dst % gpus_per_host][spine]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                double bottleneck_gbps =
                    min(min(nic_tx_of(src), nic_rx_of(dst)), spine_gbps);
                logical_rate = speedFromGbps(bottleneck_gbps);
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(nic_tx_of(src));
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

    cout << "built flex_rail graph with " << node_count
         << " GPU nodes (" << hosts << " hosts x " << gpus_per_host
         << " GPUs), nv " << nv_gbps << "G nic " << nic_gbps
         << "G taper " << taper_gbps << "G spines " << spine_count
         << " (nv:nic " << (nv_gbps / nic_gbps) << ":1),"
         << (degrade_gpu != UINT32_MAX || !degrade_mask.empty()
             ? " degraded_gpu " + ntoa(degrade_gpu) + " factor "
               + ntoa((uint32_t)degrade_factor) + " dir "
               + ntoa(degrade_dir) + " mask "
               + ntoa((uint32_t)count(degrade_mask.begin(),
                                      degrade_mask.end(), (char)1))
             : "")
         << " nv_routes " << nv_routes
         << " rail_routes " << rail_routes
         << " spine_routes " << spine_routes << endl;

    return graph;
}

// flex_pod: TRUE pod wiring (X2, report §8.2). Every host's G NICs share
// ONE host leaf; leaves connect to `spine_count` spines, each leaf-spine
// link at taper_total/spine_count. Cross-host routes are PINNED to spine
// (src+dst) % spine_count (fabric-level static ECMP -> hash collisions on
// hot destinations are structural). Unlike flex_rail there is no rail
// semantics: TX-style re-railing is meaningless; the only spreading lever
// is the receiver-side landing GPU, which selects BOTH the ingress NIC
// and (via the pin) the spine.
static SmartGraph build_flex_pod_graph(EventList& eventlist,
                                       Logfile& logfile,
                                       uint32_t hosts,
                                       uint32_t gpus_per_host,
                                       double nv_gbps,
                                       double nic_gbps,
                                       double taper_total_gbps,
                                       uint32_t spine_count,
                                       mem_b fabric_buffer_mb = 1024) {
    if (hosts == 0 || gpus_per_host == 0) {
        throw runtime_error("flex_pod: hosts and gpus_per_host must be positive");
    }
    if (nv_gbps <= 0.0 || nic_gbps <= 0.0 || taper_total_gbps <= 0.0) {
        throw runtime_error("flex_pod: link rates must be positive");
    }
    if (spine_count == 0) {
        spine_count = 1;
    }
    FLEX_RAIL_HOSTS = hosts;
    FLEX_RAIL_GPUS_PER_HOST = gpus_per_host;
    uint32_t node_count = hosts * gpus_per_host;
    double spine_gbps = taper_total_gbps / spine_count;

    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = fabric_buffer_mb * 1024 * 1024;

    SmartGraph graph;
    graph.adjacency.resize(node_count);
    for (uint32_t id = 0; id < node_count; id++) {
        graph.nodes.push_back(new SmartNode(id, node_count, eventlist,
                                            speedFromGbps(nv_gbps)));
    }

    vector<DgxRailSegment> nv_up(node_count);
    vector<DgxRailSegment> nv_down(node_count);
    vector<DgxRailSegment> nic_up(node_count);
    vector<DgxRailSegment> leaf_down(node_count);
    // Per (host, spine) fabric segments.
    vector<vector<DgxRailSegment> > leaf_up(
        hosts, vector<DgxRailSegment>(spine_count));
    vector<vector<DgxRailSegment> > spine_down(
        hosts, vector<DgxRailSegment>(spine_count));

    for (uint32_t g = 0; g < node_count; g++) {
        nv_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, egress_buffer,
            timeFromUs(9.0), "pod_nv_up_" + ntoa(g));
        nv_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, fabric_buffer,
            timeFromUs(0.0), "pod_nv_down_" + ntoa(g));
        nic_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_gbps, egress_buffer,
            timeFromUs(25.0), "pod_nic_up_" + ntoa(g));
        leaf_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_gbps, fabric_buffer,
            timeFromUs(25.0), "pod_leaf_down_" + ntoa(g));
    }
    for (uint32_t h = 0; h < hosts; h++) {
        for (uint32_t s = 0; s < spine_count; s++) {
            leaf_up[h][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0), "pod_leaf_up_" + ntoa(h) + "_" + ntoa(s));
            spine_down[h][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0),
                "pod_spine_down_" + ntoa(h) + "_" + ntoa(s));
        }
    }

    uint64_t nv_routes = 0;
    uint64_t fabric_routes = 0;
    for (uint32_t src = 0; src < node_count; src++) {
        for (uint32_t dst = 0; dst < node_count; dst++) {
            if (src == dst) {
                continue;
            }

            Route* route = new Route();
            linkspeed_bps logical_rate;
            uint32_t port;
            linkspeed_bps port_rate;

            if (src / gpus_per_host == dst / gpus_per_host) {
                append_dgx_rail_segment(route, nv_up[src]);
                append_dgx_rail_segment(route, nv_down[dst]);
                logical_rate = speedFromGbps(nv_gbps);
                port = DGX_RAIL_PORT_NV;
                port_rate = speedFromGbps(nv_gbps);
                nv_routes++;
            } else {
                uint32_t spine = (src + dst) % spine_count;
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(
                    route, leaf_up[src / gpus_per_host][spine]);
                append_dgx_rail_segment(
                    route, spine_down[dst / gpus_per_host][spine]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                logical_rate = speedFromGbps(min(nic_gbps, spine_gbps));
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(nic_gbps);
                fabric_routes++;
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

    cout << "built flex_pod graph with " << node_count
         << " GPU nodes (" << hosts << " hosts x " << gpus_per_host
         << " GPUs), nv " << nv_gbps << "G nic " << nic_gbps
         << "G leaf-uplink-total " << taper_total_gbps
         << "G spines " << spine_count
         << " nv_routes " << nv_routes
         << " fabric_routes " << fabric_routes << endl;

    return graph;
}

// flex_rail2p: DUAL-PLANE rail wiring (X3, report §8.2). Each GPU has TWO
// NICs at nic_gbps/2 each, wired into two INDEPENDENT rail fabrics (own
// leaves + own spine at taper/2). Cross-host routes are PINNED to plane
// (src+dst) % 2. Total per-GPU egress and fabric capacity equal flex_rail,
// so results are budget-comparable. planeb_factor > 1 degrades ALL of
// plane B's fabric (unannounced asymmetry: flapping links, oversubscribed
// plane); relaying through a different GPU flips the pin parity, so the
// existing rail corridors double as plane-selection levers with zero
// scheduler changes.
static const uint32_t DGX_RAIL_PORT_NIC_B = 2;

static SmartGraph build_flex_rail2p_graph(EventList& eventlist,
                                          Logfile& logfile,
                                          uint32_t hosts,
                                          uint32_t gpus_per_host,
                                          double nv_gbps,
                                          double nic_gbps,
                                          double taper_gbps,
                                          double planeb_factor = 1.0,
                                          mem_b fabric_buffer_mb = 1024) {
    if (hosts == 0 || gpus_per_host == 0) {
        throw runtime_error("flex_rail2p: hosts/gpus_per_host must be positive");
    }
    if (planeb_factor < 1.0) {
        throw runtime_error("flex_rail2p: planeb_factor must be >= 1");
    }
    FLEX_RAIL_HOSTS = hosts;
    FLEX_RAIL_GPUS_PER_HOST = gpus_per_host;
    uint32_t node_count = hosts * gpus_per_host;
    double plane_nic = nic_gbps / 2.0;
    double plane_taper = taper_gbps / 2.0;

    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = fabric_buffer_mb * 1024 * 1024;

    SmartGraph graph;
    graph.adjacency.resize(node_count);
    for (uint32_t id = 0; id < node_count; id++) {
        graph.nodes.push_back(new SmartNode(id, node_count, eventlist,
                                            speedFromGbps(nv_gbps)));
    }

    vector<DgxRailSegment> nv_up(node_count);
    vector<DgxRailSegment> nv_down(node_count);
    // Per plane: per-GPU NIC segments + per-rail fabric segments.
    vector<vector<DgxRailSegment> > nic_up(2, vector<DgxRailSegment>(node_count));
    vector<vector<DgxRailSegment> > leaf_down(2, vector<DgxRailSegment>(node_count));
    vector<vector<DgxRailSegment> > leaf_up(2, vector<DgxRailSegment>(gpus_per_host));
    vector<vector<DgxRailSegment> > spine_down(2, vector<DgxRailSegment>(gpus_per_host));

    auto plane_rate = [&](uint32_t p, double base) {
        return p == 1 ? base / planeb_factor : base;
    };

    for (uint32_t g = 0; g < node_count; g++) {
        nv_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, egress_buffer,
            timeFromUs(9.0), "r2p_nv_up_" + ntoa(g));
        nv_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nv_gbps, fabric_buffer,
            timeFromUs(0.0), "r2p_nv_down_" + ntoa(g));
        for (uint32_t p = 0; p < 2; p++) {
            nic_up[p][g] = make_dgx_rail_segment(
                graph, eventlist, logfile, plane_rate(p, plane_nic),
                egress_buffer, timeFromUs(25.0),
                "r2p_nic_up_" + ntoa(p) + "_" + ntoa(g));
            leaf_down[p][g] = make_dgx_rail_segment(
                graph, eventlist, logfile, plane_rate(p, plane_nic),
                fabric_buffer, timeFromUs(25.0),
                "r2p_leaf_down_" + ntoa(p) + "_" + ntoa(g));
        }
    }
    for (uint32_t r = 0; r < gpus_per_host; r++) {
        for (uint32_t p = 0; p < 2; p++) {
            leaf_up[p][r] = make_dgx_rail_segment(
                graph, eventlist, logfile, plane_rate(p, plane_taper),
                fabric_buffer, timeFromUs(25.0),
                "r2p_leaf_up_" + ntoa(p) + "_" + ntoa(r));
            spine_down[p][r] = make_dgx_rail_segment(
                graph, eventlist, logfile, plane_rate(p, plane_taper),
                fabric_buffer, timeFromUs(25.0),
                "r2p_spine_down_" + ntoa(p) + "_" + ntoa(r));
        }
    }

    for (uint32_t src = 0; src < node_count; src++) {
        for (uint32_t dst = 0; dst < node_count; dst++) {
            if (src == dst) {
                continue;
            }
            Route* route = new Route();
            linkspeed_bps logical_rate;
            uint32_t port;
            linkspeed_bps port_rate;
            if (src / gpus_per_host == dst / gpus_per_host) {
                append_dgx_rail_segment(route, nv_up[src]);
                append_dgx_rail_segment(route, nv_down[dst]);
                logical_rate = speedFromGbps(nv_gbps);
                port = DGX_RAIL_PORT_NV;
                port_rate = speedFromGbps(nv_gbps);
            } else {
                uint32_t p = (src + dst) % 2;
                append_dgx_rail_segment(route, nic_up[p][src]);
                if (src % gpus_per_host != dst % gpus_per_host) {
                    append_dgx_rail_segment(
                        route, leaf_up[p][src % gpus_per_host]);
                    append_dgx_rail_segment(
                        route, spine_down[p][dst % gpus_per_host]);
                }
                append_dgx_rail_segment(route, leaf_down[p][dst]);
                logical_rate = speedFromGbps(plane_rate(p, plane_nic));
                port = p == 0 ? DGX_RAIL_PORT_NIC : DGX_RAIL_PORT_NIC_B;
                port_rate = speedFromGbps(plane_rate(p, plane_nic));
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

    cout << "built flex_rail2p graph with " << node_count
         << " GPU nodes (" << hosts << " hosts x " << gpus_per_host
         << "), nv " << nv_gbps << "G nic 2x" << plane_nic
         << "G taper 2x" << plane_taper << "G planeb_factor "
         << planeb_factor << endl;
    return graph;
}

#endif
