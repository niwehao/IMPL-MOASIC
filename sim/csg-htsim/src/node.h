// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef NODE_H
#define NODE_H

#include "config.h"
#include "ecn.h"
#include "eventlist.h"
#include "network.h"
#include "queue.h"
#include "route.h"
#include "wait_queue.h"
#include "packet.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <list>

class SmartNode;
class SmartLinkSender;
class BasicScheduler;
class StaticEcmpScheduler;
class SmartTecclScheduleScheduler;
class SmartSycclScheduleScheduler;
class SmartTacclScheduleScheduler;
class SmartFastScheduleScheduler;
class SchedulerCommon;

class SmartNodeActivator {
public:
    virtual ~SmartNodeActivator() {
    }

    virtual void activate(SmartNode* node) = 0;
};

class SmartDeliveryObserver {
public:
    virtual ~SmartDeliveryObserver() {
    }

    virtual void packet_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                                  uint32_t dst_node, mem_b bytes) {
        (void)app_flow_id;
        (void)chunk_id;
        (void)dst_node;
        (void)bytes;
    }

    virtual void chunk_delivered(uint32_t app_flow_id, uint32_t chunk_id,
                                 uint32_t dst_node, mem_b bytes) = 0;
};

class SmartNode : public PacketSink, public EventSource {
public:
    typedef enum {LOCAL_CHUNK, TRANSIT_CHUNK} ChunkKind;

    struct ForwardingChoices {
        ForwardingChoices() : distance_to_dst(-1) {
        }

        int distance_to_dst;
        vector<uint32_t> minus_one_hops;
        vector<uint32_t> zero_hops;
        vector<uint32_t> plus_one_hops;
    };

    SmartNode(uint32_t id, uint32_t node_count, EventList& eventlist,
              linkspeed_bps pacing_rate)
        : EventSource(eventlist, "node_sender"), _id(id),
          _node_count(node_count), _name("node" + ntoa(id)),
          _pacing_rate(pacing_rate) {
    }

    static void set_verbose(bool verbose) {
        verbose_ref() = verbose;
    }

    static void reset_stats() {
        delivered_chunks_ref() = 0;
        delivered_bytes_ref() = 0;
        sent_chunks_ref() = 0;
        pooled_transit_chunks_ref() = 0;
        split_minus_packets_ref() = 0;
        split_zero_packets_ref() = 0;
        split_plus_packets_ref() = 0;
        ecn_feedback_generated_ref() = 0;
        ecn_feedback_forwarded_ref() = 0;
        ecn_feedback_arrived_ref() = 0;
        ecn_rate_cuts_ref() = 0;
        app_completion_times_ref().clear();
    }

    static uint64_t delivered_chunks() {
        return delivered_chunks_ref();
    }

    static uint64_t delivered_bytes() {
        return delivered_bytes_ref();
    }

    static void record_application_delivery(mem_b bytes) {
        delivered_chunks_ref()++;
        delivered_bytes_ref() += bytes;
        app_completion_times_ref().push_back(EventList::now());
    }

    static const vector<simtime_picosec>& app_completion_times() {
        return app_completion_times_ref();
    }

    static uint64_t sent_chunks() {
        return sent_chunks_ref();
    }

    static uint64_t pooled_transit_chunks() {
        return pooled_transit_chunks_ref();
    }

    static uint64_t split_minus_packets() {
        return split_minus_packets_ref();
    }

    static uint64_t split_zero_packets() {
        return split_zero_packets_ref();
    }

    static uint64_t split_plus_packets() {
        return split_plus_packets_ref();
    }

    static uint64_t ecn_feedback_generated() {
        return ecn_feedback_generated_ref();
    }

    static uint64_t ecn_feedback_forwarded() {
        return ecn_feedback_forwarded_ref();
    }

    static uint64_t ecn_feedback_arrived() {
        return ecn_feedback_arrived_ref();
    }

    static uint64_t ecn_rate_cuts() {
        return ecn_rate_cuts_ref();
    }

    static void set_activator(SmartNodeActivator* activator) {
        activator_ref() = activator;
    }

    static void set_delivery_observer(SmartDeliveryObserver* observer) {
        delivery_observer_ref() = observer;
    }

    static void set_split_weights(uint32_t minus_weight,
                                  uint32_t zero_weight,
                                  uint32_t plus_weight) {
        if (minus_weight == 0 && zero_weight == 0 && plus_weight == 0) {
            minus_weight = 1;
        }
        split_minus_weight_ref() = minus_weight;
        split_zero_weight_ref() = zero_weight;
        split_plus_weight_ref() = plus_weight;
    }

    static void set_scheduler_algorithm(const string& algorithm) {
        scheduler_algorithm_ref() = algorithm.empty() ? "basic" : algorithm;
    }

    static const string& scheduler_algorithm() {
        return scheduler_algorithm_ref();
    }

    static void set_store_forward(bool enabled) {
        store_forward_ref() = enabled;
    }

    static bool store_forward_enabled() {
        return store_forward_ref();
    }

    bool scheduler_is_teccl_schedule() const;
    bool scheduler_is_syccl_schedule() const;
    bool scheduler_is_taccl_schedule() const;
    bool scheduler_is_fast_schedule() const;
    bool scheduler_is_static_ecmp() const;

    static void activate_node(SmartNode* node) {
        if (activator_ref() && node) {
            activator_ref()->activate(node);
        }
    }

    // Every physical egress port runs its own pacing loop so a node can
    // drive all of its ports concurrently; this is scheduler-independent on
    // purpose. On point-to-point topologies each neighbor is its own port.
    // On fabric topologies (switch-as-wire) several logical neighbors share
    // one physical port and its rate.
    void add_route(uint32_t neighbor, Route* route) {
        add_route(neighbor, route, neighbor, 0);
    }

    void add_route(uint32_t neighbor, Route* route, uint32_t port,
                   linkspeed_bps port_rate) {
        _routes[neighbor].push_back(route);
        _neighbor_port[neighbor] = port;
        if (port_rate > 0) {
            _port_rates[port] = port_rate;
        }
        ensure_link_sender(port);
    }

    uint32_t port_for_neighbor(uint32_t neighbor) const {
        map<uint32_t, uint32_t>::const_iterator it =
            _neighbor_port.find(neighbor);
        if (it == _neighbor_port.end()) {
            return neighbor;
        }
        return it->second;
    }

    linkspeed_bps port_rate(uint32_t port) const {
        map<uint32_t, linkspeed_bps>::const_iterator it =
            _port_rates.find(port);
        if (it != _port_rates.end()) {
            return it->second;
        }
        it = _neighbor_rates.find(port);
        if (it != _neighbor_rates.end()) {
            return it->second;
        }
        return _pacing_rate;
    }

    void link_send_event(uint32_t port) {
        _current_send_link = port;
        doNextEvent();
        _current_send_link = UINT32_MAX;
    }

    void schedule_link_event(uint32_t link, simtime_picosec delay);

    void set_neighbor_rate(uint32_t neighbor, linkspeed_bps rate) {
        if (rate > 0) {
            _neighbor_rates[neighbor] = rate;
        }
    }

    void set_forwarding_choices(uint32_t dst, int distance_to_dst,
                                const vector<uint32_t>& minus_one_hops,
                                const vector<uint32_t>& zero_hops,
                                const vector<uint32_t>& plus_one_hops) {
        ForwardingChoices choices;
        choices.distance_to_dst = distance_to_dst;
        choices.minus_one_hops = minus_one_hops;
        choices.zero_hops = zero_hops;
        choices.plus_one_hops = plus_one_hops;
        _forwarding_table[dst] = choices;
    }

    const ForwardingChoices* forwarding_choices_to(uint32_t dst) const {
        map<uint32_t, ForwardingChoices>::const_iterator it =
            _forwarding_table.find(dst);
        if (it == _forwarding_table.end()) {
            return NULL;
        }
        return &it->second;
    }

    void set_static_ecmp_path(uint32_t dst, const vector<uint32_t>& path) {
        _static_ecmp_paths[dst] = path;
    }

    const vector<uint32_t>* static_ecmp_path_to(uint32_t dst) const {
        map<uint32_t, vector<uint32_t> >::const_iterator it =
            _static_ecmp_paths.find(dst);
        if (it == _static_ecmp_paths.end()) {
            return NULL;
        }
        return &it->second;
    }

    void add_local_chunk(PacketFlow& flow, uint32_t dst, uint32_t round,
                         uint32_t app_flow_id, uint32_t chunk_id,
                         mem_b chunk_size, uint32_t ttl,
                         packetid_t& next_packet_id) {
        vector<uint32_t> path;
        add_scheduled_chunk(flow, _id, dst, round, app_flow_id, chunk_id,
                            chunk_size, ttl, next_packet_id, path);
    }

    void add_scheduled_chunk(PacketFlow& flow, uint32_t chunk_source,
                             uint32_t dst, uint32_t round,
                             uint32_t app_flow_id, uint32_t chunk_id,
                             mem_b chunk_size, uint32_t ttl,
                             packetid_t& next_packet_id,
                             const vector<uint32_t>& explicit_path,
                             bool count_delivery = true) {
        PendingChunk chunk;
        chunk.meta = new SmartChunkMeta();
        chunk.meta->flow = &flow;
        chunk.meta->src = chunk_source;
        chunk.meta->dst = dst;
        chunk.meta->round = round;
        chunk.meta->app_flow_id = app_flow_id;
        chunk.meta->chunk_id = chunk_id;
        chunk.meta->chunk_size = chunk_size;
        chunk.meta->path = explicit_path;
        chunk.meta->count_delivery = count_delivery;
        chunk.ttl = ttl;
        chunk.kind = LOCAL_CHUNK;
        chunk.next_packet_id = &next_packet_id;
        chunk.bytes_sent = 0;
        chunk.bytes_total = chunk_size;
        chunk.packet_index = 0;
        chunk.split_cursor = 0;
        chunk.selected_split_class = 0;
        chunk.selected_next_hop = _id;
        chunk.path.push_back(_id);
        pool_insert(chunk);
        activate_self();
    }

    bool has_pending_chunks() const {
        return !_chunk_pool.empty();
    }

    bool try_send_one_chunk(int packet_size);
    bool try_send_one_packet(int packet_size);

    void doNextEvent() override;

    void receivePacket(Packet& pkt) override {
        SmartPacket* sp = dynamic_cast<SmartPacket*>(&pkt);
        if (!sp) {
            abort();
        }

        if (sp->is_ecn_feedback()) {
            bool consumed = receive_ecn_feedback(*sp);
            if (consumed) {
                pkt.free();
            }
            return;
        }

        if (sp->dst() == _id) {
            receive_destination_packet(*sp);
            pkt.free();
            return;
        }

        if (!sp->dec_ttl()) {
            if (verbose_ref()) {
                cout << "drop packet " << sp->id()
                     << " at node " << _id
                     << " reason ttl" << endl;
            }
            pkt.free();
            return;
        }

        receive_transit_packet(*sp);
        pkt.free();
    }

    const string& nodename() override {
        return _name;
    }

    uint32_t id() const {
        return _id;
    }

private:
    struct PendingChunk {
        SmartChunkMeta* meta;
        uint32_t ttl;
        // Position ticket in the per-epoch FIFO bucket index (see
        // _epoch_buckets): monotone insertion sequence; a requeue gets a
        // fresh ticket, which IS "move to the back among equal epochs".
        uint64_t bucket_seq;
        ChunkKind kind;
        packetid_t* next_packet_id;
        mem_b bytes_sent;
        mem_b bytes_total;
        uint32_t packet_index;
        uint32_t split_cursor;
        uint32_t selected_split_class;
        uint32_t selected_next_hop;
        vector<uint32_t> path;
    };

    typedef list<PendingChunk> ChunkPool;
    typedef ChunkPool::iterator ChunkIter;

    struct ChunkAccumulator {
        ChunkAccumulator()
            : meta(NULL), ttl(0), received_bytes(0) {
        }

        SmartChunkMeta* meta;
        uint32_t ttl;
        mem_b received_bytes;
        vector<uint32_t> path;
    };

    Route* sendable_route_for_chunk(PendingChunk& chunk);
    Route* route_for_split_chunk(PendingChunk& chunk) const;
    Route* route_for_next_hop(uint32_t next) const;
    void record_split_packet(uint32_t split_class) const;
    void send_chunk(const PendingChunk& chunk, Route* route, int packet_size);
    bool send_packet_from_chunk(ChunkPool::iterator chunk,
                                Route* route, int packet_size);

    vector<uint32_t> path_with_current_node(const SmartPacket& pkt) const {
        vector<uint32_t> path = pkt.path();
        if (path.empty() || path.back() != _id) {
            path.push_back(_id);
        }
        return path;
    }

    void send_ecn_feedback_if_marked(const SmartPacket& pkt,
                                     const vector<uint32_t>& path) {
        if (!(pkt.flags() & ECN_CE) || path.size() < 2) {
            return;
        }

        SmartPacket* feedback = SmartPacket::new_ecn_feedback(
            pkt.meta(), _ecn_next_packet_id++, path,
            path.at(path.size() - 2), path.back());
        ecn_feedback_generated_ref()++;
        bool consumed = receive_ecn_feedback(*feedback);
        if (consumed) {
            feedback->free();
        }
    }

    bool receive_ecn_feedback(SmartPacket& pkt) {
        const vector<uint32_t>& path = pkt.path();
        if (path.empty()) {
            return true;
        }

        uint32_t index = pkt.feedback_hop_index();
        if (pkt.ecn_upstream() == _id) {
            reduce_rate_to_neighbor(pkt.ecn_downstream());
        }

        if (index == 0) {
            if (path.front() == _id) {
                ecn_feedback_arrived_ref()++;
                if (verbose_ref()) {
                    cout << "ecn feedback arrived flow " << pkt.app_flow_id()
                         << " chunk " << pkt.chunk_id()
                         << " at source " << _id << endl;
                }
            }
            return true;
        }

        if (index >= path.size()) {
            return true;
        }

        uint32_t previous = path.at(index - 1);
        Route* route = route_for_next_hop(previous);
        if (!route) {
            return true;
        }

        pkt.set_feedback_hop_index(index - 1);
        pkt.set_dst(previous);
        pkt.set_route(*route);
        ecn_feedback_forwarded_ref()++;
        pkt.sendOn();
        return false;
    }

    void receive_transit_packet(const SmartPacket& pkt) {
        vector<uint32_t> packet_path = path_with_current_node(pkt);
        send_ecn_feedback_if_marked(pkt, packet_path);

        // basic sprays packets of one chunk over multiple next hops, so a
        // per-node chunk accumulator can never complete; store-forward only
        // applies to schedulers with a fixed per-chunk path.
        if (store_forward_enabled()
            && (scheduler_is_static_ecmp() || scheduler_is_teccl_schedule()
                || scheduler_is_syccl_schedule()
                || scheduler_is_taccl_schedule()
                || scheduler_is_fast_schedule())) {
            store_forward_transit_packet(pkt, packet_path);
            return;
        }

        PendingChunk chunk;
        chunk.meta = pkt.meta();
        chunk.ttl = pkt.ttl();
        chunk.kind = TRANSIT_CHUNK;
        chunk.next_packet_id = &_transit_next_packet_id;
        chunk.bytes_sent = 0;
        chunk.bytes_total = pkt.size();
        chunk.packet_index = pkt.chunk_packet_index();
        chunk.split_cursor = 0;
        chunk.selected_split_class = 0;
        chunk.selected_next_hop = _id;
        chunk.path = packet_path;
        pool_insert(chunk);
        activate_self();

        pooled_transit_chunks_ref()++;
        if (verbose_ref()) {
            cout << "pooled transit flow " << chunk.meta->app_flow_id
                 << " round " << chunk.meta->round
                 << " chunk " << chunk.meta->chunk_id
                 << " packet " << pkt.chunk_packet_index()
                 << " at node " << _id << endl;
        }
    }

    void store_forward_transit_packet(const SmartPacket& pkt,
                                      const vector<uint32_t>& packet_path) {
        uint64_t chunk_key = chunk_key_for(pkt.app_flow_id(), pkt.chunk_id());
        ChunkAccumulator& accumulator = _incoming_chunks[chunk_key];
        if (!accumulator.meta) {
            accumulator.meta = pkt.meta();
            accumulator.ttl = pkt.ttl();
            accumulator.received_bytes = 0;
            accumulator.path = packet_path;
        } else if (accumulator.meta != pkt.meta()) {
            throw runtime_error("store-forward transit accumulator metadata changed");
        }

        accumulator.ttl = pkt.ttl();
        accumulator.path = packet_path;
        accumulator.received_bytes += pkt.size();
        if (accumulator.received_bytes > pkt.chunk_size()) {
            throw runtime_error("store-forward transit received too many bytes");
        }

        if (accumulator.received_bytes < pkt.chunk_size()) {
            return;
        }

        PendingChunk chunk;
        chunk.meta = accumulator.meta;
        chunk.ttl = accumulator.ttl;
        chunk.kind = TRANSIT_CHUNK;
        chunk.next_packet_id = &_transit_next_packet_id;
        chunk.bytes_sent = 0;
        chunk.bytes_total = pkt.chunk_size();
        chunk.packet_index = 0;
        chunk.split_cursor = 0;
        chunk.selected_split_class = 0;
        chunk.selected_next_hop = _id;
        chunk.path = accumulator.path;
        _incoming_chunks.erase(chunk_key);

        pool_insert(chunk);
        activate_self();

        pooled_transit_chunks_ref()++;
        if (verbose_ref()) {
            cout << "pooled store-forward transit flow "
                 << chunk.meta->app_flow_id
                 << " round " << chunk.meta->round
                 << " chunk " << chunk.meta->chunk_id
                 << " at node " << _id
                 << " bytes " << chunk.bytes_total << endl;
        }
    }

    void receive_destination_packet(const SmartPacket& pkt) {
        uint64_t chunk_key = chunk_key_for(pkt.app_flow_id(), pkt.chunk_id());
        vector<uint32_t> packet_path = path_with_current_node(pkt);
        send_ecn_feedback_if_marked(pkt, packet_path);

        mem_b previously_delivered = _delivered_chunk_bytes[chunk_key];
        mem_b delivered = previously_delivered + pkt.size();
        _delivered_chunk_bytes[chunk_key] = delivered;

        if (delivery_observer_ref()) {
            delivery_observer_ref()->packet_delivered(
                pkt.app_flow_id(), pkt.chunk_id(), _id, pkt.size());
        }

        if (previously_delivered < pkt.chunk_size()
            && delivered >= pkt.chunk_size()) {
            if (pkt.count_delivery()) {
                record_application_delivery(delivered);
            }
            if (delivery_observer_ref()) {
                delivery_observer_ref()->chunk_delivered(
                    pkt.app_flow_id(), pkt.chunk_id(), _id, delivered);
            }
            if (verbose_ref()) {
                cout << "delivered flow " << pkt.app_flow_id()
                     << " round " << pkt.round()
                     << " chunk " << pkt.chunk_id()
                     << " src " << pkt.src()
                     << " dst " << pkt.dst()
                     << " bytes " << delivered
                     << " path";
                for (uint32_t node : packet_path) {
                    cout << " " << node;
                }
                cout << endl;
            }
        }
    }

    const char* chunk_kind_name(ChunkKind kind) const;
    ChunkPool::iterator select_chunk_to_send(Route*& selected_route);
    Route* primary_route_for_chunk(const PendingChunk& chunk) const;
    BaseQueue* output_queue(Route* route) const;
    uint32_t select_next_hop_for_chunk(const PendingChunk& chunk) const;
    uint32_t choose_next_hop_to(uint32_t dst) const;
    uint32_t split_class_for_bucket(uint32_t bucket) const;
    uint32_t split_weight_total() const;
    simtime_picosec pacing_interval(int packet_bytes, uint32_t next_hop) const;
    void schedule_send_event(simtime_picosec delay);
    linkspeed_bps rate_to_neighbor(uint32_t neighbor) const;
    void reduce_rate_to_neighbor(uint32_t neighbor);

    static bool& verbose_ref() {
        static bool verbose = true;
        return verbose;
    }

    static uint64_t& delivered_chunks_ref() {
        static uint64_t delivered_chunks = 0;
        return delivered_chunks;
    }

    static uint64_t& delivered_bytes_ref() {
        static uint64_t delivered_bytes = 0;
        return delivered_bytes;
    }

    static uint64_t& sent_chunks_ref() {
        static uint64_t sent_chunks = 0;
        return sent_chunks;
    }

    static uint64_t& pooled_transit_chunks_ref() {
        static uint64_t pooled_transit_chunks = 0;
        return pooled_transit_chunks;
    }

    static uint64_t& split_minus_packets_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& split_zero_packets_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& split_plus_packets_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& ecn_feedback_generated_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& ecn_feedback_forwarded_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& ecn_feedback_arrived_ref() {
        static uint64_t packets = 0;
        return packets;
    }

    static uint64_t& ecn_rate_cuts_ref() {
        static uint64_t cuts = 0;
        return cuts;
    }

    static vector<simtime_picosec>& app_completion_times_ref() {
        static vector<simtime_picosec> times;
        return times;
    }

    static SmartNodeActivator*& activator_ref() {
        static SmartNodeActivator* activator = NULL;
        return activator;
    }

    static SmartDeliveryObserver*& delivery_observer_ref() {
        static SmartDeliveryObserver* observer = NULL;
        return observer;
    }

    static uint32_t& split_minus_weight_ref() {
        static uint32_t weight = 1;
        return weight;
    }

    static uint32_t& split_zero_weight_ref() {
        static uint32_t weight = 0;
        return weight;
    }

    static uint32_t& split_plus_weight_ref() {
        static uint32_t weight = 0;
        return weight;
    }

    static string& scheduler_algorithm_ref() {
        static string algorithm = "basic";
        return algorithm;
    }

    static bool& store_forward_ref() {
        static bool store_forward = false;
        return store_forward;
    }

    static uint64_t chunk_key_for(uint32_t app_flow_id, uint32_t chunk_id) {
        return (((uint64_t)app_flow_id) << 32) | chunk_id;
    }

    void activate_self();

    void ensure_link_sender(uint32_t neighbor);

    uint32_t _id;
    uint32_t _node_count;
    string _name;
    map<uint32_t, vector<Route*> > _routes;
    map<uint32_t, ForwardingChoices> _forwarding_table;
    map<uint32_t, vector<uint32_t> > _static_ecmp_paths;
    // The pool is a list (O(1) erase/requeue; a 1024-node run keeps
    // ~20k live chunks per node and the old vector paid an O(pool)
    // memmove per PACKET). _epoch_buckets is a pure INDEX over the same
    // chunks: epoch -> (insertion ticket -> iterator), giving min-epoch
    // selection O(log) access while preserving the historical semantics
    // exactly -- min epoch first, FIFO among equals, requeue-to-back
    // among equals (fresh ticket). Maintained by pool_insert/pool_erase/
    // pool_requeue; schedulers that linearly scan _chunk_pool see the
    // same order as before (list keeps insertion order).
    ChunkPool _chunk_pool;
    map<uint32_t, map<uint64_t, ChunkIter> > _epoch_buckets;
    uint64_t _bucket_seq_next = 0;

    void pool_insert(const PendingChunk& chunk) {
        _chunk_pool.push_back(chunk);
        ChunkIter it = _chunk_pool.end();
        --it;
        it->bucket_seq = _bucket_seq_next++;
        _epoch_buckets[it->meta->round][it->bucket_seq] = it;
    }

    void pool_erase(ChunkIter it) {
        map<uint32_t, map<uint64_t, ChunkIter> >::iterator b =
            _epoch_buckets.find(it->meta->round);
        if (b != _epoch_buckets.end()) {
            b->second.erase(it->bucket_seq);
            if (b->second.empty()) {
                _epoch_buckets.erase(b);
            }
        }
        _chunk_pool.erase(it);
    }

    // Move to the back of the pool AND to the back of its epoch bucket
    // (exactly the old erase+push_back, minus the O(pool) memmove).
    void pool_requeue(ChunkIter it) {
        _chunk_pool.splice(_chunk_pool.end(), _chunk_pool, it);
        map<uint32_t, map<uint64_t, ChunkIter> >::iterator b =
            _epoch_buckets.find(it->meta->round);
        if (b != _epoch_buckets.end()) {
            b->second.erase(it->bucket_seq);
        }
        it->bucket_seq = _bucket_seq_next++;
        _epoch_buckets[it->meta->round][it->bucket_seq] = it;
    }
    unordered_map<uint64_t, ChunkAccumulator> _incoming_chunks;
    unordered_map<uint64_t, mem_b> _delivered_chunk_bytes;
    packetid_t _transit_next_packet_id = 1;
    packetid_t _ecn_next_packet_id = 1;
    linkspeed_bps _pacing_rate;
    map<uint32_t, linkspeed_bps> _neighbor_rates;
    map<uint32_t, uint32_t> _neighbor_port;
    map<uint32_t, linkspeed_bps> _port_rates;
    map<uint32_t, SmartLinkSender*> _link_senders;
    uint32_t _current_send_link = UINT32_MAX;

    friend class BasicScheduler;
    friend class StaticEcmpScheduler;
    friend class SmartTecclScheduleScheduler;
    friend class SmartSycclScheduleScheduler;
    friend class SmartTacclScheduleScheduler;
    friend class SmartFastScheduleScheduler;
    friend class SchedulerCommon;
};

class SmartLinkSender : public EventSource {
public:
    SmartLinkSender(EventList& eventlist, SmartNode* node, uint32_t neighbor)
        : EventSource(eventlist,
                      "link_sender" + ntoa(node->id()) + "_" + ntoa(neighbor)),
          _node(node), _neighbor(neighbor), _pending(false) {
    }

    void doNextEvent() override {
        _pending = false;
        _node->link_send_event(_neighbor);
    }

    SmartNode* _node;
    uint32_t _neighbor;
    bool _pending;
};

inline void SmartNode::ensure_link_sender(uint32_t neighbor) {
    if (_link_senders.find(neighbor) == _link_senders.end()) {
        _link_senders[neighbor] = new SmartLinkSender(eventlist(), this,
                                                      neighbor);
    }
}

inline void SmartNode::schedule_link_event(uint32_t link,
                                           simtime_picosec delay) {
    map<uint32_t, SmartLinkSender*>::iterator it = _link_senders.find(link);
    if (it == _link_senders.end()) {
        return;
    }
    SmartLinkSender* sender = it->second;
    if (sender->_pending || _chunk_pool.empty()) {
        return;
    }
    sender->_pending = true;
    eventlist().sourceIsPendingRel(*sender, delay);
}

inline void SmartNode::activate_self() {
    for (map<uint32_t, SmartLinkSender*>::iterator it = _link_senders.begin();
         it != _link_senders.end(); ++it) {
        schedule_link_event(it->first, 0);
    }
}

#include "scheduler.h"

#endif
