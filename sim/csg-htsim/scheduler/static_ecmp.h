// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef STATIC_ECMP_SCHEDULER_H
#define STATIC_ECMP_SCHEDULER_H

#include "common.h"

#include <set>
#include <stdexcept>

class StaticEcmpScheduler {
public:
    template <class Graph>
    static void install_paths(Graph& graph) {
        uint32_t node_count = (uint32_t)graph.nodes.size();
        uint64_t installed = 0;

        for (uint32_t src = 0; src < node_count; src++) {
            for (uint32_t dst = 0; dst < node_count; dst++) {
                if (src == dst) {
                    continue;
                }
                vector<uint32_t> path = build_path(graph, src, dst);
                graph.nodes[src]->set_static_ecmp_path(dst, path);
                installed++;
            }
        }

        cout << "installed static ECMP paths: " << installed << endl;
    }

    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<StaticEcmpScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk<StaticEcmpScheduler>(
            node, chunk);
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
        SchedulerCommon::do_next_event<StaticEcmpScheduler>(node);
    }

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
        if (next == UINT32_MAX) {
            return NULL;
        }
        return route_for_next_hop(node, next);
    }

    static Route* route_for_split_chunk(SmartNode* node,
                                        SmartNode::PendingChunk& chunk) {
        ensure_chunk_path(node, chunk);
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

    static uint32_t choose_next_hop_to(SmartNode* node, uint32_t dst) {
        const vector<uint32_t>* path = node->static_ecmp_path_to(dst);
        if (!path || path->size() < 2 || path->front() != node->_id) {
            return UINT32_MAX;
        }
        return path->at(1);
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

private:
    static void ensure_chunk_path(SmartNode* node, SmartNode::PendingChunk& chunk) {
        if (!chunk.meta->path.empty()) {
            return;
        }
        const vector<uint32_t>* path = node->static_ecmp_path_to(chunk.meta->dst);
        if (!path || path->size() < 2 || path->front() != chunk.meta->src) {
            throw runtime_error("missing static ECMP path");
        }
        chunk.meta->path = *path;
    }

    template <class Graph>
    static vector<uint32_t> build_path(const Graph& graph,
                                       uint32_t src, uint32_t dst) {
        vector<uint32_t> path;
        path.push_back(src);

        uint32_t current = src;
        uint32_t salt = src * 1103515245u + dst * 12345u;
        set<uint32_t> visited;
        visited.insert(current);

        while (current != dst) {
            const SmartNode::ForwardingChoices* choices =
                graph.nodes[current]->forwarding_choices_to(dst);
            if (!choices || choices->minus_one_hops.empty()) {
                throw runtime_error("static ECMP path has no shortest next hop");
            }

            uint32_t index = salt % choices->minus_one_hops.size();
            uint32_t next = choices->minus_one_hops.at(index);
            if (visited.find(next) != visited.end()) {
                throw runtime_error("static ECMP path looped");
            }
            path.push_back(next);
            visited.insert(next);
            current = next;
            salt = salt * 1664525u + 1013904223u;
        }

        return path;
    }
};

#endif
