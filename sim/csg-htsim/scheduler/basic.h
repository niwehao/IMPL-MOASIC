// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef BASIC_SCHEDULER_H
#define BASIC_SCHEDULER_H

#include "common.h"

class BasicScheduler {
public:
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        return SchedulerCommon::try_send_one_packet<BasicScheduler>(
            node, packet_size);
    }

    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        return SchedulerCommon::sendable_route_for_chunk<BasicScheduler>(
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
        SchedulerCommon::do_next_event<BasicScheduler>(node);
    }

    static simtime_picosec pacing_interval(SmartNode* node, int packet_bytes,
                                           uint32_t next_hop) {
        return SchedulerCommon::pacing_interval(node, packet_bytes, next_hop);
    }

    static void schedule_send_event(SmartNode* node, simtime_picosec delay) {
        SchedulerCommon::schedule_send_event(node, delay);
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
        return route_for_next_hop(node, next);
    }

    static Route* route_for_next_hop(SmartNode* node, uint32_t next) {
        return SchedulerCommon::route_for_next_hop(node, next);
    }

    static Route* route_for_split_chunk(SmartNode* node,
                                        SmartNode::PendingChunk& chunk) {
        const SmartNode::ForwardingChoices* choices =
            node->forwarding_choices_to(chunk.meta->dst);
        if (!choices) {
            uint32_t next = select_next_hop_for_chunk(node, chunk);
            chunk.selected_split_class = 0;
            chunk.selected_next_hop = next;
            return route_for_next_hop(node, next);
        }

        return SchedulerCommon::route_for_choices_on_link(node, chunk,
                                                          choices);
    }

    static void record_split_packet(uint32_t split_class) {
        SchedulerCommon::record_split_packet(split_class);
    }

    static BaseQueue* output_queue(Route* route) {
        return SchedulerCommon::output_queue(route);
    }

    static uint32_t select_next_hop_for_chunk(
        SmartNode* node, const SmartNode::PendingChunk& chunk) {
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

        uint32_t clockwise = (dst + node->_node_count - node->_id)
            % node->_node_count;
        uint32_t counterclockwise = (node->_id + node->_node_count - dst)
            % node->_node_count;

        if (clockwise <= counterclockwise) {
            return (node->_id + 1) % node->_node_count;
        }
        return (node->_id + node->_node_count - 1) % node->_node_count;
    }

    static uint32_t split_class_for_bucket(uint32_t bucket) {
        return SchedulerCommon::split_class_for_bucket(bucket);
    }

    static uint32_t split_weight_total() {
        return SchedulerCommon::split_weight_total();
    }

    static linkspeed_bps rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        return SchedulerCommon::rate_to_neighbor(node, neighbor);
    }

    static void reduce_rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        SchedulerCommon::reduce_rate_to_neighbor(node, neighbor);
    }
};

#endif
