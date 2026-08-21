// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef REGULAR_GRAPH_H
#define REGULAR_GRAPH_H

#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "pipe.h"
#include "queue.h"
#include "route.h"
#include "node.h"
#include "wait_queue.h"

#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

struct SmartGraph {
    vector<SmartNode*> nodes;
    vector<BaseQueue*> queues;
    vector<Pipe*> pipes;
    vector<Route*> routes;
    vector<vector<uint32_t> > adjacency;
};
#define SMART_GRAPH_DEFINED 1

static vector<uint32_t> choose_low_diameter_offsets(uint32_t node_count,
                                                    uint32_t offset_count) {
    vector<uint32_t> offsets;
    if (offset_count == 0) {
        return offsets;
    }

    uint32_t max_offset = (node_count - 1) / 2;
    set<uint32_t> seen;
    for (uint32_t i = 0; i < offset_count; i++) {
        double exponent = (double)i / (double)offset_count;
        uint32_t offset = (uint32_t)(pow((double)node_count, exponent) + 0.5);
        if (offset < 1) {
            offset = 1;
        }
        if (offset > max_offset) {
            offset = max_offset;
        }
        seen.insert(offset);
    }

    for (uint32_t offset = 1; seen.size() < offset_count && offset <= max_offset;
         offset++) {
        seen.insert(offset);
    }

    for (set<uint32_t>::const_iterator it = seen.begin(); it != seen.end(); ++it) {
        offsets.push_back(*it);
    }
    return offsets;
}

static vector<int> graph_distances_from(const SmartGraph& graph, uint32_t src) {
    uint32_t node_count = (uint32_t)graph.nodes.size();
    vector<int> distance(node_count, -1);
    queue<uint32_t> pending;

    distance[src] = 0;
    pending.push(src);

    while (!pending.empty()) {
        uint32_t current = pending.front();
        pending.pop();

        for (uint32_t neighbor : graph.adjacency[current]) {
            if (distance[neighbor] != -1) {
                continue;
            }
            distance[neighbor] = distance[current] + 1;
            pending.push(neighbor);
        }
    }

    return distance;
}

static vector<vector<int> > graph_all_pair_distances(const SmartGraph& graph) {
    uint32_t node_count = (uint32_t)graph.nodes.size();
    vector<vector<int> > distances;
    distances.reserve(node_count);

    for (uint32_t src = 0; src < node_count; src++) {
        distances.push_back(graph_distances_from(graph, src));
    }
    return distances;
}

static uint32_t graph_diameter_from_distances(
        const vector<vector<int> >& distances) {
    uint32_t node_count = (uint32_t)distances.size();
    uint32_t diameter = 0;

    for (uint32_t src = 0; src < node_count; src++) {
        for (uint32_t dst = 0; dst < node_count; dst++) {
            if (distances[src][dst] < 0) {
                throw runtime_error("generated graph is disconnected");
            }
            if ((uint32_t)distances[src][dst] > diameter) {
                diameter = (uint32_t)distances[src][dst];
            }
        }
    }

    return diameter;
}

static Route* make_graph_route(EventList& eventlist, Logfile& logfile,
                               SmartGraph& graph, SmartNode* owner,
                               SmartNode* dst,
                               linkspeed_bps speed, mem_b buffer,
                               simtime_picosec delay, const string& name) {
    SmartQueue* queue = new SmartQueue(speed, buffer, eventlist, NULL);
    queue->set_owner(owner);
    queue->setName(name + "_queue");
    logfile.writeName(*queue);
    graph.queues.push_back(queue);

    Pipe* pipe = new Pipe(delay, eventlist);
    pipe->setName(name + "_pipe");
    logfile.writeName(*pipe);
    graph.pipes.push_back(pipe);

    Route* route = new Route();
    route->push_back(queue);
    route->push_back(pipe);
    route->push_back(dst);
    graph.routes.push_back(route);
    return route;
}

static void add_graph_edge(SmartGraph& graph, EventList& eventlist,
                           Logfile& logfile, uint32_t left, uint32_t right,
                           linkspeed_bps speed, mem_b buffer,
                           simtime_picosec delay) {
    graph.adjacency[left].push_back(right);
    graph.adjacency[right].push_back(left);
    graph.nodes[left]->set_neighbor_rate(right, speed);
    graph.nodes[right]->set_neighbor_rate(left, speed);

    graph.nodes[left]->add_route(
        right,
        make_graph_route(eventlist, logfile, graph, graph.nodes[left],
                         graph.nodes[right], speed, buffer, delay,
                         "link_" + ntoa(left) + "_to_" + ntoa(right)));

    graph.nodes[right]->add_route(
        left,
        make_graph_route(eventlist, logfile, graph, graph.nodes[right],
                         graph.nodes[left], speed, buffer, delay,
                         "link_" + ntoa(right) + "_to_" + ntoa(left)));
}

static bool is_regular_ring_edge(uint32_t node_count, uint32_t left,
                                 uint32_t right) {
    if (node_count <= 1) {
        return false;
    }
    uint32_t diff = left < right ? right - left : left - right;
    return diff == 1 || diff == node_count - 1;
}

static void install_forwarding_choices(
        SmartGraph& graph, const vector<vector<int> >& distances) {
    uint32_t node_count = (uint32_t)graph.nodes.size();
    uint64_t minus_one_count = 0;
    uint64_t zero_count = 0;
    uint64_t plus_one_count = 0;

    for (uint32_t node = 0; node < node_count; node++) {
        for (uint32_t dst = 0; dst < node_count; dst++) {
            if (node == dst) {
                continue;
            }

            int current_distance = distances[node][dst];
            if (current_distance < 0) {
                throw runtime_error("generated graph is disconnected");
            }

            vector<uint32_t> minus_one_hops;
            vector<uint32_t> zero_hops;
            vector<uint32_t> plus_one_hops;

            for (uint32_t neighbor : graph.adjacency[node]) {
                int neighbor_distance = distances[neighbor][dst];
                int delta = neighbor_distance - current_distance;
                if (delta == -1) {
                    minus_one_hops.push_back(neighbor);
                } else if (delta == 0) {
                    zero_hops.push_back(neighbor);
                } else if (delta == 1) {
                    plus_one_hops.push_back(neighbor);
                } else {
                    throw runtime_error("neighbor distance delta is outside -1..1");
                }
            }

            if (minus_one_hops.empty()) {
                throw runtime_error("no shortest-path next hop found");
            }
            minus_one_count += minus_one_hops.size();
            zero_count += zero_hops.size();
            plus_one_count += plus_one_hops.size();

            graph.nodes[node]->set_forwarding_choices(
                dst, current_distance, minus_one_hops, zero_hops,
                plus_one_hops);
        }
    }

    cout << "installed forwarding choices: minus_one " << minus_one_count
         << ", zero " << zero_count
         << ", plus_one " << plus_one_count << endl;
}

static SmartGraph build_d_regular_graph(uint32_t node_count, uint32_t degree,
                                        EventList& eventlist, Logfile& logfile,
                                        linkspeed_bps speed, mem_b buffer,
                                        simtime_picosec delay,
                                        bool hetero_ring = false,
                                        double hetero_multiplier = 2.0) {
    if (node_count == 0) {
        throw runtime_error("node count must be greater than zero");
    }
    if (degree >= node_count) {
        throw runtime_error("degree must be smaller than node count");
    }
    if ((((uint64_t)node_count) * degree) % 2 != 0) {
        throw runtime_error("N * D must be even for a D-regular graph");
    }
    if (degree % 2 == 1 && node_count % 2 == 1) {
        throw runtime_error("odd degree requires an even node count");
    }
    if (hetero_ring && hetero_multiplier <= 0.0) {
        throw runtime_error("hetero multiplier must be positive");
    }

    SmartGraph graph;
    graph.adjacency.resize(node_count);

    for (uint32_t id = 0; id < node_count; id++) {
        graph.nodes.push_back(new SmartNode(id, node_count, eventlist, speed));
    }

    set<pair<uint32_t, uint32_t> > edges;
    vector<uint32_t> offsets = choose_low_diameter_offsets(node_count, degree / 2);
    for (uint32_t offset : offsets) {
        for (uint32_t node = 0; node < node_count; node++) {
            uint32_t peer = (node + offset) % node_count;
            uint32_t left = node < peer ? node : peer;
            uint32_t right = node < peer ? peer : node;
            edges.insert(make_pair(left, right));
        }
    }

    if (degree % 2 == 1) {
        uint32_t offset = node_count / 2;
        for (uint32_t node = 0; node < node_count; node++) {
            uint32_t peer = (node + offset) % node_count;
            uint32_t left = node < peer ? node : peer;
            uint32_t right = node < peer ? peer : node;
            edges.insert(make_pair(left, right));
        }
    }

    for (set<pair<uint32_t, uint32_t> >::const_iterator it = edges.begin();
         it != edges.end(); ++it) {
        linkspeed_bps edge_speed = speed;
        if (hetero_ring
            && is_regular_ring_edge(node_count, it->first, it->second)) {
            edge_speed = (linkspeed_bps)((double)speed * hetero_multiplier);
        }
        add_graph_edge(graph, eventlist, logfile, it->first, it->second,
                       edge_speed, buffer, delay);
    }

    vector<vector<int> > distances = graph_all_pair_distances(graph);
    install_forwarding_choices(graph, distances);
    uint32_t diameter = graph_diameter_from_distances(distances);
    cout << "built " << degree << "-regular graph with "
         << node_count << " nodes, " << edges.size()
         << " edges, diameter " << diameter;
    if (hetero_ring) {
        cout << ", hetero_ring offset=1 multiplier " << hetero_multiplier;
    }
    cout << endl;
    return graph;
}

#endif
