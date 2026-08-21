// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef FLEX_CUBE_GRAPH_H
#define FLEX_CUBE_GRAPH_H

#include "ndv2_graph.h"
#include "flex_pod_graph.h"

#include <sstream>
#include <stdexcept>

// flex_cube: DGX-shaped pod whose INTRA-HOST fabric is an NDv2-style
// hybrid cube-mesh (degree 4, 16 point-to-point links per 8-GPU host)
// instead of an NVSwitch crossbar, with HETEROGENEOUS per-link speeds
// (e.g. 400/200/100G tiers mixed by a knob). Inter-host wiring is the
// same rail fabric as flex_rail (one NIC per GPU at nic_gbps, same-rail
// leaf, tapered spines, identical 25us segment latencies), so the
// scale-out side of every result is directly comparable to flex_rail.
//
// Modeling choices (consistent with the flex family):
//   - Each DIRECTED cube link is one switch-as-wire segment (queue+pipe)
//     at its own speed; the sending GPU paces each outgoing link with a
//     dedicated egress port (port id 100+neighbor), like a real NVLink
//     serializer. Link contention shows up in the shared link queue.
//   - Every intra-host ordered pair gets ONE pinned route: BFS shortest
//     path, ties broken by max bottleneck capacity then lowest next-hop
//     id (deterministic). Non-adjacent pairs therefore ride through
//     transit GPUs' links without a scheduler hop -- schedulers that
//     want chunk-level relays (chunk availability semantics) emit them
//     explicitly, exactly as on flex_rail.
//   - Degenerate mode: full crossbar + single speed, for calibration
//     against flex_rail (aggregate ingress (G-1)*v vs nv pipe).
//
// The intra-host speed layout is exposed via FLEX_CUBE_LOCAL_GBPS so the
// planners (MOSAIC prior, FAST guard) can read the SAME ground truth the
// simulator executes -- schedule-time and run-time topologies must match.

static const uint32_t FLEX_CUBE_PORT_BASE = 100;  // intra port = base+nbr

// Set by build_flex_cube_graph: local G x G symmetric gbps matrix
// (0 = no link) and pinned local paths [src][dst] = node sequence.
static vector<vector<double> > FLEX_CUBE_LOCAL_GBPS;
static vector<vector<vector<uint32_t> > > FLEX_CUBE_LOCAL_PATH;

// Parse a tier spec like "400x6,200x6,100x4" into per-edge speeds for
// the 16 NDv2 cube edges (NDV2_32_CUBE_EDGES order), rotated by `rot`.
static vector<double> flex_cube_edge_speeds(const string& spec,
                                            uint32_t edge_count,
                                            uint32_t rot) {
    vector<double> tiers;
    stringstream ss(spec);
    string item;
    while (getline(ss, item, ',')) {
        size_t x = item.find('x');
        if (x == string::npos) {
            throw runtime_error("flex_cube speeds: expect GBPSxCOUNT,...");
        }
        double gbps = atof(item.substr(0, x).c_str());
        int count = atoi(item.substr(x + 1).c_str());
        if (gbps <= 0.0 || count <= 0) {
            throw runtime_error("flex_cube speeds: bad tier " + item);
        }
        for (int i = 0; i < count; i++) {
            tiers.push_back(gbps);
        }
    }
    if (tiers.size() != edge_count) {
        throw runtime_error("flex_cube speeds: tier counts must sum to "
                            + ntoa(edge_count) + " edges");
    }
    vector<double> speeds(edge_count);
    for (uint32_t e = 0; e < edge_count; e++) {
        speeds[e] = tiers[(e + rot) % edge_count];
    }
    return speeds;
}

// Local adjacency matrix: cube mode uses the NDv2 hybrid cube-mesh edge
// list (requires G == 8); full mode is a complete graph at full_gbps.
static vector<vector<double> > flex_cube_local_matrix(uint32_t gpus,
                                                      const string& spec,
                                                      uint32_t rot,
                                                      double full_gbps) {
    vector<vector<double> > local(gpus, vector<double>(gpus, 0.0));
    if (full_gbps > 0.0) {
        for (uint32_t a = 0; a < gpus; a++) {
            for (uint32_t b = 0; b < gpus; b++) {
                if (a != b) {
                    local[a][b] = full_gbps;
                }
            }
        }
        return local;
    }
    if (gpus != 8) {
        throw runtime_error("flex_cube: cube-mesh mode requires "
                            "gpus_per_host = 8 (NDv2 adjacency)");
    }
    vector<double> speeds = flex_cube_edge_speeds(spec, 16, rot);
    for (uint32_t e = 0; e < 16; e++) {
        uint32_t a = NDV2_32_CUBE_EDGES[e][0];
        uint32_t b = NDV2_32_CUBE_EDGES[e][1];
        local[a][b] = speeds[e];
        local[b][a] = speeds[e];
    }
    return local;
}

// Pinned local path a -> b: BFS shortest hops; among shortest paths take
// max bottleneck capacity, then lexicographically smallest sequence.
static vector<uint32_t> flex_cube_local_path(
        const vector<vector<double> >& local, uint32_t a, uint32_t b) {
    uint32_t g = (uint32_t)local.size();
    if (local[a][b] > 0.0) {
        vector<uint32_t> direct;
        direct.push_back(a);
        direct.push_back(b);
        return direct;
    }
    // BFS levels from a.
    vector<int> dist(g, -1);
    dist[a] = 0;
    vector<uint32_t> frontier(1, a);
    while (!frontier.empty() && dist[b] < 0) {
        vector<uint32_t> next;
        for (uint32_t i = 0; i < frontier.size(); i++) {
            uint32_t u = frontier[i];
            for (uint32_t v = 0; v < g; v++) {
                if (local[u][v] > 0.0 && dist[v] < 0) {
                    dist[v] = dist[u] + 1;
                    next.push_back(v);
                }
            }
        }
        frontier = next;
    }
    if (dist[b] < 0) {
        throw runtime_error("flex_cube: intra graph is disconnected");
    }
    // Depth-limited DFS over shortest paths, tracking the best.
    vector<uint32_t> best, cur(1, a);
    double best_bn = -1.0;
    struct Rec {
        static void go(const vector<vector<double> >& local,
                       const vector<int>& dist, uint32_t u, uint32_t b,
                       double bn, vector<uint32_t>& cur,
                       vector<uint32_t>& best, double& best_bn) {
            if (u == b) {
                if (bn > best_bn || (bn == best_bn
                                     && (best.empty() || cur < best))) {
                    best = cur;
                    best_bn = bn;
                }
                return;
            }
            uint32_t g = (uint32_t)local.size();
            for (uint32_t v = 0; v < g; v++) {
                if (local[u][v] <= 0.0 || dist[v] != dist[u] + 1) {
                    continue;
                }
                cur.push_back(v);
                Rec::go(local, dist, v, b, min(bn, local[u][v]), cur,
                        best, best_bn);
                cur.pop_back();
            }
        }
    };
    Rec::go(local, dist, a, b, 1e18, cur, best, best_bn);
    return best;
}

static SmartGraph build_flex_cube_graph(EventList& eventlist,
                                        Logfile& logfile,
                                        uint32_t hosts,
                                        uint32_t gpus_per_host,
                                        const vector<vector<double> >& local,
                                        double nic_gbps,
                                        double taper_gbps,
                                        uint32_t spine_count,
                                        double link_delay_us,
                                        mem_b fabric_buffer_mb = 1024) {
    if (hosts == 0 || gpus_per_host == 0) {
        throw runtime_error("flex_cube: hosts and gpus_per_host must be positive");
    }
    if (nic_gbps <= 0.0 || taper_gbps <= 0.0) {
        throw runtime_error("flex_cube: link rates must be positive");
    }
    if (spine_count == 0) {
        spine_count = 1;
    }
    FLEX_RAIL_HOSTS = hosts;
    FLEX_RAIL_GPUS_PER_HOST = gpus_per_host;
    uint32_t node_count = hosts * gpus_per_host;
    uint32_t gpus = gpus_per_host;
    double spine_gbps = taper_gbps / spine_count;

    const mem_b egress_buffer = 32 * 1024 * 1024;
    const mem_b fabric_buffer = fabric_buffer_mb * 1024 * 1024;
    const simtime_picosec link_delay = timeFromUs(link_delay_us);

    FLEX_CUBE_LOCAL_GBPS = local;
    FLEX_CUBE_LOCAL_PATH.assign(
        gpus, vector<vector<uint32_t> >(gpus));
    for (uint32_t a = 0; a < gpus; a++) {
        for (uint32_t b = 0; b < gpus; b++) {
            if (a != b) {
                FLEX_CUBE_LOCAL_PATH[a][b] =
                    flex_cube_local_path(local, a, b);
            }
        }
    }

    SmartGraph graph;
    graph.adjacency.resize(node_count);
    for (uint32_t id = 0; id < node_count; id++) {
        double bin = 0.0;
        for (uint32_t v = 0; v < gpus; v++) {
            bin += local[id % gpus][v];
        }
        graph.nodes.push_back(new SmartNode(id, node_count, eventlist,
                                            speedFromGbps(bin)));
    }

    // Per-host directed cube link segments [host][u][v].
    vector<vector<vector<DgxRailSegment> > > link_seg(
        hosts, vector<vector<DgxRailSegment> >(
            gpus, vector<DgxRailSegment>(gpus)));
    uint64_t cube_links = 0;
    for (uint32_t h = 0; h < hosts; h++) {
        for (uint32_t u = 0; u < gpus; u++) {
            for (uint32_t v = 0; v < gpus; v++) {
                if (local[u][v] <= 0.0) {
                    continue;
                }
                link_seg[h][u][v] = make_dgx_rail_segment(
                    graph, eventlist, logfile, local[u][v], fabric_buffer,
                    link_delay, "cube_" + ntoa(h) + "_" + ntoa(u) + "_"
                    + ntoa(v));
                cube_links++;
            }
        }
    }

    // Rail fabric segments, exactly as flex_rail.
    vector<DgxRailSegment> nic_up(node_count);
    vector<DgxRailSegment> leaf_down(node_count);
    vector<vector<DgxRailSegment> > leaf_up(
        gpus, vector<DgxRailSegment>(spine_count));
    vector<vector<DgxRailSegment> > spine_down(
        gpus, vector<DgxRailSegment>(spine_count));
    for (uint32_t g = 0; g < node_count; g++) {
        nic_up[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_gbps, egress_buffer,
            timeFromUs(25.0), "cube_nic_up_" + ntoa(g));
        leaf_down[g] = make_dgx_rail_segment(
            graph, eventlist, logfile, nic_gbps, fabric_buffer,
            timeFromUs(25.0), "cube_leaf_down_" + ntoa(g));
    }
    for (uint32_t r = 0; r < gpus; r++) {
        for (uint32_t s = 0; s < spine_count; s++) {
            leaf_up[r][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0), "cube_leaf_up_" + ntoa(r) + "_" + ntoa(s));
            spine_down[r][s] = make_dgx_rail_segment(
                graph, eventlist, logfile, spine_gbps, fabric_buffer,
                timeFromUs(25.0),
                "cube_spine_down_" + ntoa(r) + "_" + ntoa(s));
        }
    }

    uint64_t intra_routes = 0;
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

            if (src / gpus == dst / gpus) {
                uint32_t h = src / gpus;
                const vector<uint32_t>& path =
                    FLEX_CUBE_LOCAL_PATH[src % gpus][dst % gpus];
                double bottleneck = 1e18;
                for (uint32_t i = 0; i + 1 < path.size(); i++) {
                    append_dgx_rail_segment(
                        route, link_seg[h][path[i]][path[i + 1]]);
                    bottleneck = min(bottleneck,
                                     local[path[i]][path[i + 1]]);
                }
                logical_rate = speedFromGbps(bottleneck);
                port = FLEX_CUBE_PORT_BASE + h * gpus + path[1];
                port_rate = speedFromGbps(local[path[0]][path[1]]);
                intra_routes++;
            } else if (src % gpus == dst % gpus) {
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                logical_rate = speedFromGbps(nic_gbps);
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(nic_gbps);
                rail_routes++;
            } else {
                uint32_t spine = (src + dst) % spine_count;
                append_dgx_rail_segment(route, nic_up[src]);
                append_dgx_rail_segment(route, leaf_up[src % gpus][spine]);
                append_dgx_rail_segment(route,
                                        spine_down[dst % gpus][spine]);
                append_dgx_rail_segment(route, leaf_down[dst]);
                logical_rate = speedFromGbps(min(nic_gbps, spine_gbps));
                port = DGX_RAIL_PORT_NIC;
                port_rate = speedFromGbps(nic_gbps);
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

    double edge_sum = 0.0;
    uint32_t edge_cnt = 0;
    for (uint32_t u = 0; u < gpus; u++) {
        for (uint32_t v = u + 1; v < gpus; v++) {
            if (local[u][v] > 0.0) {
                edge_sum += local[u][v];
                edge_cnt++;
            }
        }
    }
    cout << "built flex_cube graph with " << node_count
         << " GPU nodes (" << hosts << " hosts x " << gpus
         << " GPUs cube-mesh), intra_edges " << edge_cnt
         << " mean_gbps " << (edge_cnt ? edge_sum / edge_cnt : 0.0)
         << " nic " << nic_gbps << "G taper " << taper_gbps
         << "G spines " << spine_count
         << " link_delay_us " << link_delay_us
         << " cube_link_segs " << cube_links
         << " intra_routes " << intra_routes
         << " rail_routes " << rail_routes
         << " spine_routes " << spine_routes << endl;
    for (uint32_t g = 0; g < gpus; g++) {
        double bin = 0.0;
        for (uint32_t v = 0; v < gpus; v++) {
            bin += local[g][v];
        }
        cout << "flex_cube_local gpu " << g << " bin_gbps " << bin
             << " nbrs";
        for (uint32_t v = 0; v < gpus; v++) {
            if (local[g][v] > 0.0) {
                cout << " " << v << ":" << local[g][v];
            }
        }
        cout << endl;
    }

    return graph;
}

#endif
