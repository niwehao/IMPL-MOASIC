// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "node.h"
#include "../scheduler/basic.h"
#include "../scheduler/fast_schedule.h"
#include "../scheduler/static_ecmp.h"
#include "../scheduler/syccl_schedule.h"
#include "../scheduler/taccl_schedule.h"
#include "../scheduler/teccl_schedule.h"

inline bool SmartNode::scheduler_is_teccl_schedule() const {
    return scheduler_algorithm() == "teccl_schedule";
}

inline bool SmartNode::scheduler_is_syccl_schedule() const {
    return scheduler_algorithm() == "syccl_schedule";
}

inline bool SmartNode::scheduler_is_taccl_schedule() const {
    return scheduler_algorithm() == "taccl_schedule";
}

inline bool SmartNode::scheduler_is_fast_schedule() const {
    // mosaic_schedule shares the fast node-level behavior (min-epoch
    // priority over explicit-path chunks); only the runtime differs.
    return scheduler_algorithm() == "fast_schedule"
        || scheduler_algorithm() == "mosaic_schedule";
}

inline bool SmartNode::scheduler_is_static_ecmp() const {
    return scheduler_algorithm() == "static_ecmp";
}

inline bool SmartNode::try_send_one_chunk(int packet_size) {
    return try_send_one_packet(packet_size);
}

inline bool SmartNode::try_send_one_packet(int packet_size) {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::try_send_one_packet(this,
                                                                packet_size);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::try_send_one_packet(this,
                                                                packet_size);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::try_send_one_packet(this,
                                                                packet_size);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::try_send_one_packet(this,
                                                                packet_size);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::try_send_one_packet(this, packet_size);
    }
    return BasicScheduler::try_send_one_packet(this, packet_size);
}

inline Route* SmartNode::sendable_route_for_chunk(PendingChunk& chunk) {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::sendable_route_for_chunk(this,
                                                                     chunk);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::sendable_route_for_chunk(this,
                                                                     chunk);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::sendable_route_for_chunk(this,
                                                                     chunk);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::sendable_route_for_chunk(this,
                                                                     chunk);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::sendable_route_for_chunk(this, chunk);
    }
    return BasicScheduler::sendable_route_for_chunk(this, chunk);
}

inline Route* SmartNode::route_for_split_chunk(PendingChunk& chunk) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::route_for_split_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::route_for_split_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::route_for_split_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::route_for_split_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::route_for_split_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    return BasicScheduler::route_for_split_chunk(
        const_cast<SmartNode*>(this), chunk);
}

inline Route* SmartNode::route_for_next_hop(uint32_t next) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::route_for_next_hop(
            const_cast<SmartNode*>(this), next);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::route_for_next_hop(
            const_cast<SmartNode*>(this), next);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::route_for_next_hop(
            const_cast<SmartNode*>(this), next);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::route_for_next_hop(
            const_cast<SmartNode*>(this), next);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::route_for_next_hop(
            const_cast<SmartNode*>(this), next);
    }
    return BasicScheduler::route_for_next_hop(const_cast<SmartNode*>(this), next);
}

inline void SmartNode::record_split_packet(uint32_t split_class) const {
    if (scheduler_is_fast_schedule()) {
        SmartFastScheduleScheduler::record_split_packet(split_class);
        return;
    }
    if (scheduler_is_taccl_schedule()) {
        SmartTacclScheduleScheduler::record_split_packet(split_class);
        return;
    }
    if (scheduler_is_syccl_schedule()) {
        SmartSycclScheduleScheduler::record_split_packet(split_class);
        return;
    }
    if (scheduler_is_teccl_schedule()) {
        SmartTecclScheduleScheduler::record_split_packet(split_class);
        return;
    }
    if (scheduler_is_static_ecmp()) {
        StaticEcmpScheduler::record_split_packet(split_class);
        return;
    }
    BasicScheduler::record_split_packet(split_class);
}

inline void SmartNode::send_chunk(const PendingChunk& chunk, Route* route,
                                  int packet_size) {
    if (scheduler_is_fast_schedule()) {
        SmartFastScheduleScheduler::send_chunk(this, chunk, route,
                                                packet_size);
        return;
    }
    if (scheduler_is_taccl_schedule()) {
        SmartTacclScheduleScheduler::send_chunk(this, chunk, route,
                                                packet_size);
        return;
    }
    if (scheduler_is_syccl_schedule()) {
        SmartSycclScheduleScheduler::send_chunk(this, chunk, route,
                                                packet_size);
        return;
    }
    if (scheduler_is_teccl_schedule()) {
        SmartTecclScheduleScheduler::send_chunk(this, chunk, route,
                                                packet_size);
        return;
    }
    if (scheduler_is_static_ecmp()) {
        StaticEcmpScheduler::send_chunk(this, chunk, route, packet_size);
        return;
    }
    BasicScheduler::send_chunk(this, chunk, route, packet_size);
}

inline bool SmartNode::send_packet_from_chunk(
    SmartNode::ChunkPool::iterator chunk, Route* route, int packet_size) {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::send_packet_from_chunk(
            this, chunk, route, packet_size);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::send_packet_from_chunk(
            this, chunk, route, packet_size);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::send_packet_from_chunk(
            this, chunk, route, packet_size);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::send_packet_from_chunk(
            this, chunk, route, packet_size);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::send_packet_from_chunk(
            this, chunk, route, packet_size);
    }
    return BasicScheduler::send_packet_from_chunk(this, chunk, route,
                                                  packet_size);
}

inline void SmartNode::doNextEvent() {
    if (scheduler_is_fast_schedule()) {
        SmartFastScheduleScheduler::do_next_event(this);
        return;
    }
    if (scheduler_is_taccl_schedule()) {
        SmartTacclScheduleScheduler::do_next_event(this);
        return;
    }
    if (scheduler_is_syccl_schedule()) {
        SmartSycclScheduleScheduler::do_next_event(this);
        return;
    }
    if (scheduler_is_teccl_schedule()) {
        SmartTecclScheduleScheduler::do_next_event(this);
        return;
    }
    if (scheduler_is_static_ecmp()) {
        StaticEcmpScheduler::do_next_event(this);
        return;
    }
    BasicScheduler::do_next_event(this);
}

inline const char* SmartNode::chunk_kind_name(ChunkKind kind) const {
    return kind == LOCAL_CHUNK ? "local" : "transit";
}

inline SmartNode::ChunkIter
SmartNode::select_chunk_to_send(Route*& selected_route) {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::select_chunk_to_send(
            this, selected_route);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::select_chunk_to_send(
            this, selected_route);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::select_chunk_to_send(
            this, selected_route);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::select_chunk_to_send(
            this, selected_route);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::select_chunk_to_send(this, selected_route);
    }
    return BasicScheduler::select_chunk_to_send(this, selected_route);
}

inline Route* SmartNode::primary_route_for_chunk(const PendingChunk& chunk) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::primary_route_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::primary_route_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::primary_route_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::primary_route_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::primary_route_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    return BasicScheduler::primary_route_for_chunk(
        const_cast<SmartNode*>(this), chunk);
}

inline BaseQueue* SmartNode::output_queue(Route* route) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::output_queue(route);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::output_queue(route);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::output_queue(route);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::output_queue(route);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::output_queue(route);
    }
    return BasicScheduler::output_queue(route);
}

inline uint32_t SmartNode::select_next_hop_for_chunk(
    const PendingChunk& chunk) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::select_next_hop_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::select_next_hop_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::select_next_hop_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::select_next_hop_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::select_next_hop_for_chunk(
            const_cast<SmartNode*>(this), chunk);
    }
    return BasicScheduler::select_next_hop_for_chunk(
        const_cast<SmartNode*>(this), chunk);
}

inline uint32_t SmartNode::choose_next_hop_to(uint32_t dst) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::choose_next_hop_to(
            const_cast<SmartNode*>(this), dst);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::choose_next_hop_to(
            const_cast<SmartNode*>(this), dst);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::choose_next_hop_to(
            const_cast<SmartNode*>(this), dst);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::choose_next_hop_to(
            const_cast<SmartNode*>(this), dst);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::choose_next_hop_to(
            const_cast<SmartNode*>(this), dst);
    }
    return BasicScheduler::choose_next_hop_to(const_cast<SmartNode*>(this), dst);
}

inline uint32_t SmartNode::split_class_for_bucket(uint32_t bucket) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::split_class_for_bucket(bucket);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::split_class_for_bucket(bucket);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::split_class_for_bucket(bucket);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::split_class_for_bucket(bucket);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::split_class_for_bucket(bucket);
    }
    return BasicScheduler::split_class_for_bucket(bucket);
}

inline uint32_t SmartNode::split_weight_total() const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::split_weight_total();
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::split_weight_total();
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::split_weight_total();
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::split_weight_total();
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::split_weight_total();
    }
    return BasicScheduler::split_weight_total();
}

inline simtime_picosec SmartNode::pacing_interval(int packet_bytes,
                                                  uint32_t next_hop) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::pacing_interval(
            const_cast<SmartNode*>(this), packet_bytes, next_hop);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::pacing_interval(
            const_cast<SmartNode*>(this), packet_bytes, next_hop);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::pacing_interval(
            const_cast<SmartNode*>(this), packet_bytes, next_hop);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::pacing_interval(
            const_cast<SmartNode*>(this), packet_bytes, next_hop);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::pacing_interval(
            const_cast<SmartNode*>(this), packet_bytes, next_hop);
    }
    return BasicScheduler::pacing_interval(const_cast<SmartNode*>(this),
                                           packet_bytes, next_hop);
}

inline void SmartNode::schedule_send_event(simtime_picosec delay) {
    if (scheduler_is_fast_schedule()) {
        SmartFastScheduleScheduler::schedule_send_event(this, delay);
        return;
    }
    if (scheduler_is_taccl_schedule()) {
        SmartTacclScheduleScheduler::schedule_send_event(this, delay);
        return;
    }
    if (scheduler_is_syccl_schedule()) {
        SmartSycclScheduleScheduler::schedule_send_event(this, delay);
        return;
    }
    if (scheduler_is_teccl_schedule()) {
        SmartTecclScheduleScheduler::schedule_send_event(this, delay);
        return;
    }
    if (scheduler_is_static_ecmp()) {
        StaticEcmpScheduler::schedule_send_event(this, delay);
        return;
    }
    BasicScheduler::schedule_send_event(this, delay);
}

inline linkspeed_bps SmartNode::rate_to_neighbor(uint32_t neighbor) const {
    if (scheduler_is_fast_schedule()) {
        return SmartFastScheduleScheduler::rate_to_neighbor(
            const_cast<SmartNode*>(this), neighbor);
    }
    if (scheduler_is_taccl_schedule()) {
        return SmartTacclScheduleScheduler::rate_to_neighbor(
            const_cast<SmartNode*>(this), neighbor);
    }
    if (scheduler_is_syccl_schedule()) {
        return SmartSycclScheduleScheduler::rate_to_neighbor(
            const_cast<SmartNode*>(this), neighbor);
    }
    if (scheduler_is_teccl_schedule()) {
        return SmartTecclScheduleScheduler::rate_to_neighbor(
            const_cast<SmartNode*>(this), neighbor);
    }
    if (scheduler_is_static_ecmp()) {
        return StaticEcmpScheduler::rate_to_neighbor(
            const_cast<SmartNode*>(this), neighbor);
    }
    return BasicScheduler::rate_to_neighbor(const_cast<SmartNode*>(this),
                                            neighbor);
}

inline void SmartNode::reduce_rate_to_neighbor(uint32_t neighbor) {
    if (scheduler_is_fast_schedule()) {
        SmartFastScheduleScheduler::reduce_rate_to_neighbor(this, neighbor);
        return;
    }
    if (scheduler_is_taccl_schedule()) {
        SmartTacclScheduleScheduler::reduce_rate_to_neighbor(this, neighbor);
        return;
    }
    if (scheduler_is_syccl_schedule()) {
        SmartSycclScheduleScheduler::reduce_rate_to_neighbor(this, neighbor);
        return;
    }
    if (scheduler_is_teccl_schedule()) {
        SmartTecclScheduleScheduler::reduce_rate_to_neighbor(this, neighbor);
        return;
    }
    if (scheduler_is_static_ecmp()) {
        StaticEcmpScheduler::reduce_rate_to_neighbor(this, neighbor);
        return;
    }
    BasicScheduler::reduce_rate_to_neighbor(this, neighbor);
}

#endif
