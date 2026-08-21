// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef COMMON_SCHEDULER_H
#define COMMON_SCHEDULER_H

#include "../src/node.h"

#include <algorithm>

class SchedulerCommon {
public:
    template <class Strategy>
    static bool try_send_one_packet(SmartNode* node, int packet_size) {
        Route* selected_route = NULL;
        SmartNode::ChunkIter chunk =
            Strategy::select_chunk_to_send(node, selected_route);
        if (chunk == node->_chunk_pool.end()) {
            return false;
        }

        return send_packet_from_chunk(node, chunk, selected_route, packet_size);
    }

    template <class Strategy>
    static Route* sendable_route_for_chunk(SmartNode* node,
                                           SmartNode::PendingChunk& chunk) {
        Route* route = Strategy::route_for_split_chunk(node, chunk);
        if (!route) {
            return NULL;
        }

        // Each send event belongs to one physical egress port; a chunk is
        // only sendable in this event if its next hop leaves via that port.
        if (node->_current_send_link != UINT32_MAX
            && node->port_for_neighbor(chunk.selected_next_hop)
               != node->_current_send_link) {
            return NULL;
        }

        BaseQueue* queue = output_queue(route);
        if (!queue) {
            return NULL;
        }

        SmartQueue* smart_queue = dynamic_cast<SmartQueue*>(queue);
        if (smart_queue && smart_queue->is_paused()) {
            return NULL;
        }
        return route;
    }

    template <class Strategy>
    static void do_next_event(SmartNode* node) {
        uint32_t link = node->_current_send_link;

        bool sent = Strategy::try_send_one_packet(
            node, Packet::data_packet_size());
        if (node->_chunk_pool.empty()) {
            return;
        }

        simtime_picosec delay = sent
            ? pacing_interval(node, Packet::data_packet_size(), link)
            : timeFromUs((uint32_t)1);
        node->schedule_link_event(link, delay);
    }

    // For schedulers with multiple candidate next hops (basic/syccl
    // fallback): the port whose send event is running pulls the chunk if
    // one of its next hops appears in an allowed split class. Idle ports
    // thereby drain the pool in parallel instead of waiting for a
    // round-robin turn.
    static Route* route_for_choices_on_link(
        SmartNode* node, SmartNode::PendingChunk& chunk,
        const SmartNode::ForwardingChoices* choices) {
        uint32_t port = node->_current_send_link;
        const vector<uint32_t>* class_hops[3] = {
            &choices->minus_one_hops, &choices->zero_hops,
            &choices->plus_one_hops};
        uint32_t class_weights[3] = {
            SmartNode::split_minus_weight_ref(),
            SmartNode::split_zero_weight_ref(),
            SmartNode::split_plus_weight_ref()};

        for (uint32_t split_class = 0; split_class < 3; split_class++) {
            if (class_weights[split_class] == 0) {
                continue;
            }
            const vector<uint32_t>& hops = *class_hops[split_class];
            if (hops.empty()) {
                continue;
            }

            vector<uint32_t> candidates;
            if (port == UINT32_MAX) {
                candidates = hops;
            } else {
                for (uint32_t hop : hops) {
                    if (node->port_for_neighbor(hop) == port) {
                        candidates.push_back(hop);
                    }
                }
            }
            if (candidates.empty()) {
                continue;
            }

            uint32_t next =
                candidates[chunk.packet_index % candidates.size()];
            Route* route = route_for_next_hop(node, next);
            if (!route) {
                continue;
            }
            chunk.selected_split_class = split_class;
            chunk.selected_next_hop = next;
            return route;
        }
        return NULL;
    }

    static void send_chunk(SmartNode* node,
                           const SmartNode::PendingChunk& chunk,
                           Route* route, int packet_size) {
        assert(route);

        mem_b remaining = chunk.meta->chunk_size;
        uint32_t chunk_packet_index = 0;
        while (remaining > 0) {
            int bytes = packet_size;
            if (remaining < bytes) {
                bytes = (int)remaining;
            }

            SmartPacket* pkt = SmartPacket::newpkt(chunk.meta, bytes,
                                                   (*chunk.next_packet_id)++,
                                                   chunk.ttl,
                                                   chunk_packet_index);
            pkt->set_path(chunk.path);
            pkt->set_route(*route);
            pkt->sendOn();
            remaining -= bytes;
            chunk_packet_index++;
        }

        node->sent_chunks_ref()++;
        print_sent_chunk(node, chunk, chunk.meta->chunk_size);
    }

    static bool send_packet_from_chunk(
        SmartNode* node, SmartNode::ChunkIter chunk,
        Route* route, int packet_size) {
        assert(route);
        assert(chunk != node->_chunk_pool.end());

        mem_b remaining = chunk->bytes_total - chunk->bytes_sent;
        int bytes = packet_size;
        if (remaining < bytes) {
            bytes = (int)remaining;
        }

        SmartPacket* pkt = SmartPacket::newpkt(chunk->meta, bytes,
                                               (*chunk->next_packet_id)++,
                                               chunk->ttl,
                                               chunk->packet_index);
        pkt->set_path(chunk->path);
        pkt->set_route(*route);
        pkt->sendOn();

        record_split_packet(chunk->selected_split_class);
        chunk->bytes_sent += bytes;
        chunk->packet_index++;
        uint32_t total_weight = split_weight_total();
        if (total_weight > 0) {
            chunk->split_cursor = (chunk->split_cursor + 1) % total_weight;
        }
        if (chunk->bytes_sent < chunk->bytes_total) {
            node->pool_requeue(chunk);
            return true;
        }

        node->sent_chunks_ref()++;
        print_sent_chunk(node, *chunk, chunk->bytes_total);
        node->pool_erase(chunk);
        return true;
    }

    static Route* route_for_next_hop(SmartNode* node, uint32_t next) {
        map<uint32_t, vector<Route*> >::const_iterator it =
            node->_routes.find(next);
        if (it == node->_routes.end() || it->second.empty()) {
            return NULL;
        }
        return it->second.at(0);
    }

    static BaseQueue* output_queue(Route* route) {
        if (!route || route->size() == 0) {
            return NULL;
        }
        return dynamic_cast<BaseQueue*>(route->at(0));
    }

    static void record_split_packet(uint32_t split_class) {
        if (split_class == 0) {
            SmartNode::split_minus_packets_ref()++;
        } else if (split_class == 1) {
            SmartNode::split_zero_packets_ref()++;
        } else {
            SmartNode::split_plus_packets_ref()++;
        }
    }

    static uint32_t split_class_for_bucket(uint32_t bucket) {
        if (bucket < SmartNode::split_minus_weight_ref()) {
            return 0;
        }
        bucket -= SmartNode::split_minus_weight_ref();
        if (bucket < SmartNode::split_zero_weight_ref()) {
            return 1;
        }
        return 2;
    }

    static uint32_t split_weight_total() {
        return SmartNode::split_minus_weight_ref()
            + SmartNode::split_zero_weight_ref()
            + SmartNode::split_plus_weight_ref();
    }

    static simtime_picosec pacing_interval(SmartNode* node, int packet_bytes,
                                           uint32_t port) {
        linkspeed_bps rate = node->port_rate(port);
        if (rate <= 0 || packet_bytes <= 0) {
            return 0;
        }
        return (simtime_picosec)(((double)packet_bytes * 8.0 * 1000000000000.0)
                                 / (double)rate);
    }

    static void schedule_send_event(SmartNode* node, simtime_picosec delay) {
        node->schedule_link_event(node->_current_send_link, delay);
    }

    static linkspeed_bps rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        map<uint32_t, linkspeed_bps>::const_iterator it =
            node->_neighbor_rates.find(neighbor);
        if (it == node->_neighbor_rates.end()) {
            return node->_pacing_rate;
        }
        return it->second;
    }

    static void reduce_rate_to_neighbor(SmartNode* node, uint32_t neighbor) {
        linkspeed_bps current = rate_to_neighbor(node, neighbor);
        linkspeed_bps minimum = node->_pacing_rate / 100;
        if (minimum < 1) {
            minimum = 1;
        }
        linkspeed_bps reduced = current / 2;
        if (reduced < minimum) {
            reduced = minimum;
        }
        if (reduced < current) {
            node->_neighbor_rates[neighbor] = reduced;
            SmartNode::ecn_rate_cuts_ref()++;
            if (node->verbose_ref()) {
                cout << "ecn rate cut node " << node->_id
                     << " next_hop " << neighbor
                     << " rate " << current
                     << " -> " << reduced << endl;
            }
        }
    }

private:
    static void print_sent_chunk(SmartNode* node,
                                 const SmartNode::PendingChunk& chunk,
                                 mem_b bytes) {
        if (node->verbose_ref()) {
            cout << "sent " << node->chunk_kind_name(chunk.kind)
                 << " round " << chunk.meta->round
                 << " flow " << chunk.meta->app_flow_id
                 << " chunk " << chunk.meta->chunk_id
                 << " at node " << node->_id
                 << " src " << chunk.meta->src
                 << " dst " << chunk.meta->dst
                 << " bytes " << bytes << endl;
        }
    }
};

#endif
