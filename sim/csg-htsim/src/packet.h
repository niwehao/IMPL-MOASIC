// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef PACKET_H
#define PACKET_H

#include "network.h"

#include <vector>

struct SmartChunkMeta {
    PacketFlow* flow;
    uint32_t src;
    uint32_t dst;
    uint32_t round;
    uint32_t app_flow_id;
    uint32_t chunk_id;
    mem_b chunk_size;
    vector<uint32_t> path;
    bool count_delivery;
};

class SmartPacket : public Packet {
public:
    static SmartPacket* newpkt(SmartChunkMeta* meta, int pkt_size,
                               packetid_t id, uint32_t ttl,
                               uint32_t chunk_packet_index) {
        SmartPacket* p;
        if (packet_pool().empty()) {
            p = new SmartPacket();
        } else {
            p = packet_pool().back();
            packet_pool().pop_back();
        }

        p->set_attrs(*meta->flow, pkt_size, id);
        p->_type = IP;
        p->set_dst(meta->dst);
        p->_ttl = ttl;
        p->_chunk_packet_index = chunk_packet_index;
        p->_meta = meta;
        p->_ecn_feedback = false;
        p->_feedback_hop_index = 0;
        p->_ecn_upstream = UINT32_MAX;
        p->_ecn_downstream = UINT32_MAX;
        p->_path.clear();
        return p;
    }

    static SmartPacket* new_ecn_feedback(SmartChunkMeta* meta,
                                         packetid_t id,
                                         const vector<uint32_t>& path,
                                         uint32_t ecn_upstream,
                                         uint32_t ecn_downstream) {
        SmartPacket* p;
        if (packet_pool().empty()) {
            p = new SmartPacket();
        } else {
            p = packet_pool().back();
            packet_pool().pop_back();
        }

        p->set_attrs(*meta->flow, 64, id);
        p->_type = IP;
        p->_ttl = 0;
        p->_chunk_packet_index = 0;
        p->_meta = meta;
        p->_ecn_feedback = true;
        p->_path = path;
        p->_feedback_hop_index = path.empty() ? 0 : (uint32_t)path.size() - 1;
        p->_ecn_upstream = ecn_upstream;
        p->_ecn_downstream = ecn_downstream;
        if (!path.empty()) {
            p->set_dst(path.back());
        }
        return p;
    }

    void free() override {
        _meta = NULL;
        _path.clear();
        _ecn_feedback = false;
        _feedback_hop_index = 0;
        _ecn_upstream = UINT32_MAX;
        _ecn_downstream = UINT32_MAX;
        packet_pool().push_back(this);
    }

    PktPriority priority() const override {
        return Packet::PRIO_LO;
    }

    uint32_t src() const {
        return _meta->src;
    }

    uint32_t chunk_id() const {
        return _meta->chunk_id;
    }

    uint32_t chunk_packet_index() const {
        return _chunk_packet_index;
    }

    bool is_first_chunk_packet() const {
        return _chunk_packet_index == 0;
    }

    bool is_ecn_feedback() const {
        return _ecn_feedback;
    }

    uint32_t round() const {
        return _meta->round;
    }

    uint32_t app_flow_id() const {
        return _meta->app_flow_id;
    }

    mem_b chunk_size() const {
        return _meta->chunk_size;
    }

    bool count_delivery() const {
        return _meta->count_delivery;
    }

    uint32_t ttl() const {
        return _ttl;
    }

    bool dec_ttl() {
        if (_ttl == 0) {
            return false;
        }
        _ttl--;
        return true;
    }

    void record_visit(uint32_t node_id) {
        _path.push_back(node_id);
    }

    const vector<uint32_t>& path() const {
        return _path;
    }

    void set_path(const vector<uint32_t>& path) {
        _path = path;
    }

    uint32_t feedback_hop_index() const {
        return _feedback_hop_index;
    }

    void set_feedback_hop_index(uint32_t index) {
        _feedback_hop_index = index;
    }

    uint32_t ecn_upstream() const {
        return _ecn_upstream;
    }

    uint32_t ecn_downstream() const {
        return _ecn_downstream;
    }

    SmartChunkMeta* meta() const {
        return _meta;
    }

private:
    static vector<SmartPacket*>& packet_pool() {
        static vector<SmartPacket*> pool;
        return pool;
    }

    uint32_t _ttl;
    uint32_t _chunk_packet_index;
    SmartChunkMeta* _meta;
    bool _ecn_feedback;
    uint32_t _feedback_hop_index;
    uint32_t _ecn_upstream;
    uint32_t _ecn_downstream;
    vector<uint32_t> _path;
};

#endif
