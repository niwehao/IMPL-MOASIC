// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef OPTICAL_FABRIC_H
#define OPTICAL_FABRIC_H

// Chunk-level discrete-event model of a reconfigurable circuit-switched
// (photonic) scale-up domain, deliberately SEPARATE from the packet-level
// EPS simulator (different physics: a circuit is dedicated, an EPS link is
// statistically multiplexed).
//
// Model (follows the common abstraction of Reconfigure-or-Not
// arXiv:2602.10468 / Harvest arXiv:2602.09188):
//   - n nodes; at any instant the fabric realizes a set of directed
//     circuits (u -> v), each of full bandwidth `gbps` and propagation
//     delay `delay_ps`. Degree constraints are the CALLER's contract
//     (each node has k transceivers), asserted here.
//   - store-and-forward at chunk granularity: a chunk fully arrives at a
//     node before its next hop starts. Per-circuit FIFO service.
//   - reconfigure(edges): tears down all circuits, pays `reconf_ps`
//     (fabric silent), then the new circuits come up. When to call it
//     (barrier semantics etc.) is the scheduler's decision.
//   - transfers are injected at runtime with an explicit hop path; the
//     delivery callback lets an online scheduler react (wave feedback).
//
// Chunk-level is exact for this physics: with dedicated circuits and
// store-and-forward there is no sub-chunk statistical multiplexing to
// capture, so a packet loop would only add simulation cost.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>

namespace optical {

typedef uint64_t psec_t;
typedef int64_t bytes_t;

struct Chunk {
    uint32_t xfer_id;
    uint32_t chunk_id;
    bytes_t bytes;
    uint32_t next_hop;           // index into Transfer::path edges
};

struct Transfer {
    uint32_t id;
    uint32_t src;
    uint32_t dst;
    bytes_t bytes;
    std::vector<uint32_t> path;  // node sequence, path[0]==src, back()==dst
    uint64_t tag;                // scheduler cookie (e.g. wave / corridor)
    bytes_t delivered = 0;
    bool done = false;
};

class Fabric {
public:
    Fabric(uint32_t nodes, uint32_t degree, double gbps, psec_t delay_ps,
           psec_t reconf_ps, bytes_t chunk_bytes)
        : _n(nodes), _degree(degree), _gbps(gbps), _delay_ps(delay_ps),
          _reconf_ps(reconf_ps), _chunk_bytes(chunk_bytes) {
    }

    psec_t now() const { return _now; }

    // Paper bookkeeping (Reconfigure-or-Not Thm.1): a strategy with d
    // topologies pays d*R INCLUDING the initial configuration. Default
    // off (initial free); comparisons must use one convention for all
    // schemes.
    void set_pay_initial(bool on) { _pay_initial = on; }
    // Control-plane round trip charged on every circuit activation
    // (request + grant messages). 0 = legacy accounting (35/39/40).
    void set_ctl_ps(psec_t p) { _ctl_ps = p; }

    // Advance wall clock (per-phase alpha / software launch overhead).
    void advance(psec_t dt) { _now += dt; }
    psec_t reconf_delay() const { return _reconf_ps; }
    uint32_t reconf_count() const { return _reconfigs; }
    bytes_t hop_bytes() const { return _hop_bytes; }
    bytes_t app_bytes() const { return _app_bytes; }

    // Serialization time of `b` bytes on one circuit.
    psec_t ser_ps(bytes_t b) const {
        return (psec_t)((double)b * 8000.0 / _gbps + 0.5);
    }

    // ---- topology ----------------------------------------------------

    // Replace the circuit set. Pays the reconfiguration delay only if the
    // fabric was already configured (the initial setup is free, matching
    // the papers' accounting) AND the edge set actually changes.
    // In-flight chunks must be none: the caller enforces its barrier.
    void reconfigure(const std::vector<std::pair<uint32_t, uint32_t> >& edges) {
        assert(_inflight == 0 && "reconfigure with chunks in flight");
        if (_configured && edges == _edges) {
            return;                       // no-op, no delay
        }
        if (_configured || _pay_initial) {
            _now += _reconf_ps;
            _reconfigs++;
        }
        _edges = edges;
        _configured = true;
        _link_free.clear();
        _link_queue.clear();
        std::vector<uint32_t> outdeg(_n, 0), indeg(_n, 0);
        for (size_t i = 0; i < edges.size(); i++) {
            uint32_t u = edges[i].first, v = edges[i].second;
            assert(u < _n && v < _n && u != v);
            outdeg[u]++;
            indeg[v]++;
            _link_free[key(u, v)] = _now;
        }
        for (uint32_t v = 0; v < _n; v++) {
            assert(outdeg[v] <= _degree && indeg[v] <= _degree
                   && "degree constraint violated");
        }
    }

    bool has_edge(uint32_t u, uint32_t v) const {
        return _link_free.count(key(u, v)) > 0;
    }

    const std::vector<std::pair<uint32_t, uint32_t> >& edges() const {
        return _edges;
    }

    // BFS shortest path on the current circuit set; empty if unreachable.
    std::vector<uint32_t> shortest_path(uint32_t s, uint32_t d) const {
        std::vector<int32_t> prev(_n, -1);
        std::vector<bool> seen(_n, false);
        std::queue<uint32_t> q;
        q.push(s);
        seen[s] = true;
        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();
            if (u == d) {
                break;
            }
            for (size_t i = 0; i < _edges.size(); i++) {
                if (_edges[i].first != u) {
                    continue;
                }
                uint32_t v = _edges[i].second;
                if (!seen[v]) {
                    seen[v] = true;
                    prev[v] = (int32_t)u;
                    q.push(v);
                }
            }
        }
        std::vector<uint32_t> path;
        if (!seen[d]) {
            return path;
        }
        for (int32_t v = (int32_t)d; v != -1; v = (v == (int32_t)s) ? -1 : prev[v]) {
            path.push_back((uint32_t)v);
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // ---- transfers ---------------------------------------------------

    typedef std::function<void(const Transfer&)> DeliveryCb;

    void set_delivery_cb(DeliveryCb cb) { _on_delivery = cb; }

    // Inject a transfer along an explicit node path (every consecutive
    // pair must be a live circuit at injection time; later reconfigs must
    // keep the still-needed hops alive -- schedulers using mid-transfer
    // reconfiguration must drain first, which the barrier assert enforces).
    uint32_t inject(uint32_t src, uint32_t dst, bytes_t bytes,
                    const std::vector<uint32_t>& path, uint64_t tag = 0) {
        assert(path.size() >= 2 && path.front() == src && path.back() == dst);
        Transfer t;
        t.id = (uint32_t)_transfers.size();
        t.src = src;
        t.dst = dst;
        t.bytes = bytes;
        t.path = path;
        t.tag = tag;
        _transfers.push_back(t);
        bytes_t left = bytes;
        uint32_t cid = 0;
        while (left > 0) {
            bytes_t b = std::min(left, _chunk_bytes);
            left -= b;
            Chunk c;
            c.xfer_id = t.id;
            c.chunk_id = cid++;
            c.bytes = b;
            c.next_hop = 0;
            enqueue_on_link(path[0], path[1], c, _now);
        }
        return t.id;
    }

    const Transfer& transfer(uint32_t id) const { return _transfers[id]; }
    size_t transfer_count() const { return _transfers.size(); }

    bool idle() const { return _inflight == 0 && _events.empty(); }
    bool has_events() const { return !_events.empty(); }

    // ---- partial (per-port) reconfiguration extension ---------------
    //
    // Default OFF; none of the legacy entry points change behavior.
    // Semantics: the fabric is a set of circuits evolving per PORT
    // rather than per phase. close_circuit() tears a circuit down once
    // its queue drains; open_circuit() establishes one R after BOTH its
    // endpoints' previous circuits drained -- only those two ports are
    // blocked for R, the rest of the fabric keeps serving (this is the
    // standard per-port capability of MEMS/robotic OCS hardware: each
    // mirror retargets independently). In-flight chunks on unchanged
    // circuits are never touched, which is exactly what the legacy
    // whole-fabric barrier assert forbids.
    //
    // set_link_quota() fires a callback once `bytes` have been SERVED
    // on a circuit -- the executor's hook to advance a port to its next
    // session (event-driven receding horizon).

    void enable_partial() { _partial = true; }
    bool partial_mode() const { return _partial; }
    uint32_t port_reconfigs() const { return _port_reconfigs; }

    // Establish circuit u->v. `initial` marks the very first
    // configuration (free unless pay_initial, matching the papers'
    // accounting used by the legacy reconfigure()).
    void open_circuit(uint32_t u, uint32_t v, bool initial = false) {
        assert(_partial);
        assert(u < _n && v < _n && u != v);
        uint64_t k = key(u, v);
        assert(!has_edge(u, v) && !_pending.count(k));
        _pending[k] = initial ? 0 : 1;
        try_activate(k);
    }

    // Tear circuit u->v down: immediately if no chunk is mid-
    // serialization, else right after the current chunk completes.
    // QUEUED chunks are NOT drained -- they stay in the link queue and
    // are served if/when the same circuit is opened again (they belong
    // to a later session of this port pair). A pending (not yet live)
    // circuit is simply canceled.
    void close_circuit(uint32_t u, uint32_t v) {
        assert(_partial);
        uint64_t k = key(u, v);
        if (_pending.count(k)) {
            assert(_link_queue[k].empty());
            _pending.erase(k);
            return;
        }
        assert(has_edge(u, v));
        if (!_link_busy.count(k)) {
            teardown(k, std::max(_now, _link_free[k]));
        } else {
            _closing.insert(k);
        }
    }

    void set_link_quota(uint32_t u, uint32_t v, bytes_t bytes,
                        std::function<void()> cb) {
        assert(_partial);
        uint64_t k = key(u, v);
        _quota[k] = std::make_pair(_served[k] + bytes, cb);
    }

    bytes_t link_served(uint32_t u, uint32_t v) {
        return _served[key(u, v)];
    }

    // diagnostics: bytes currently queued toward circuit u->v
    bytes_t link_queued(uint32_t u, uint32_t v) {
        std::vector<Chunk>& q = _link_queue[key(u, v)];
        bytes_t b = 0;
        for (size_t i = 0; i < q.size(); i++) b += q[i].bytes;
        return b;
    }

    // Run until no pending events (all injected traffic delivered) or
    // until `until_ps` (0 = no limit). Returns current time.
    psec_t run(psec_t until_ps = 0) {
        while (!_events.empty()) {
            Ev e = _events.top();
            if (until_ps > 0 && e.t > until_ps) {
                break;
            }
            _events.pop();
            _now = std::max(_now, e.t);
            if (e.kind == EV_ARRIVE) {
                on_arrival(e);
            } else {
                on_link_free(e);
            }
        }
        return _now;
    }

private:
    enum EvKind { EV_ARRIVE, EV_FREE };
    struct Ev {
        psec_t t;
        uint64_t seq;            // deterministic tiebreak: creation order
        EvKind kind;
        uint64_t link;           // for EV_FREE
        Chunk chunk;             // for EV_ARRIVE
        bool operator>(const Ev& o) const {
            if (t != o.t) return t > o.t;
            return seq > o.seq;
        }
    };

    uint64_t key(uint32_t u, uint32_t v) const {
        return ((uint64_t)u << 32) | v;
    }

    // ---- partial-mode internals (no-ops unless enable_partial()) ----

    uint32_t live_outdeg(uint32_t u) const {
        uint32_t c = 0;
        for (size_t i = 0; i < _edges.size(); i++)
            if (_edges[i].first == u) c++;
        return c;
    }
    uint32_t live_indeg(uint32_t v) const {
        uint32_t c = 0;
        for (size_t i = 0; i < _edges.size(); i++)
            if (_edges[i].second == v) c++;
        return c;
    }

    // A pending circuit goes live once both its ports have spare degree;
    // it pays R on those two ports' own timelines (their free times),
    // not on the fabric clock.
    void try_activate(uint64_t k) {
        std::map<uint64_t, uint8_t>::iterator it = _pending.find(k);
        if (it == _pending.end()) return;
        uint32_t u = (uint32_t)(k >> 32), v = (uint32_t)(k & 0xffffffffu);
        if (live_outdeg(u) >= _degree || live_indeg(v) >= _degree) return;
        if (_tx_free_ps.size() < _n) {
            _tx_free_ps.assign(_n, 0);
            _rx_free_ps.assign(_n, 0);
        }
        bool initial = (it->second == 0);
        psec_t rr = (initial && !_pay_initial) ? 0 : _reconf_ps;
        psec_t t_live = std::max(std::max(_now, _tx_free_ps[u]),
                                 _rx_free_ps[v]) + rr + _ctl_ps;
        if (rr) _port_reconfigs++;
        _pending.erase(it);
        _edges.push_back(std::make_pair(u, v));
        _link_free[k] = t_live;
        _configured = true;
        if (!_link_queue[k].empty()) {
            schedule_service(k, t_live);
        }
    }

    void teardown(uint64_t k, psec_t t) {
        uint32_t u = (uint32_t)(k >> 32), v = (uint32_t)(k & 0xffffffffu);
        for (size_t i = 0; i < _edges.size(); i++) {
            if (_edges[i].first == u && _edges[i].second == v) {
                _edges.erase(_edges.begin() + i);
                break;
            }
        }
        _link_free.erase(k);
        _closing.erase(k);
        if (_tx_free_ps.size() < _n) {
            _tx_free_ps.assign(_n, 0);
            _rx_free_ps.assign(_n, 0);
        }
        _tx_free_ps[u] = std::max(_tx_free_ps[u], t);
        _rx_free_ps[v] = std::max(_rx_free_ps[v], t);
        // Freed degree may unblock pending circuits on these ports.
        std::vector<uint64_t> cand;
        for (std::map<uint64_t, uint8_t>::iterator it = _pending.begin();
             it != _pending.end(); ++it) {
            uint32_t pu = (uint32_t)(it->first >> 32);
            uint32_t pv = (uint32_t)(it->first & 0xffffffffu);
            if (pu == u || pv == v) cand.push_back(it->first);
        }
        for (size_t i = 0; i < cand.size(); i++) try_activate(cand[i]);
    }

    // Book the just-finished serialization and fire the quota callback.
    void partial_served(uint64_t k) {
        std::map<uint64_t, bytes_t>::iterator sv = _serving.find(k);
        if (sv == _serving.end() || sv->second == 0) return;
        _served[k] += sv->second;
        sv->second = 0;
        std::map<uint64_t,
                 std::pair<bytes_t, std::function<void()> > >::iterator
            q = _quota.find(k);
        if (q != _quota.end() && _served[k] >= q->second.first) {
            std::function<void()> cb = q->second.second;
            _quota.erase(q);
            if (cb) cb();
        }
    }

    void partial_check_close(uint64_t k, psec_t t) {
        if (_closing.count(k) && !_link_busy.count(k)) {
            teardown(k, t);
        }
    }

    void enqueue_on_link(uint32_t u, uint32_t v, const Chunk& c, psec_t t) {
        // Partial mode: a chunk may be queued toward a circuit that a
        // LATER session will open (validation catches true stranding).
        assert((has_edge(u, v) || _pending.count(key(u, v)) || _partial)
               && "chunk routed over a dead circuit");
        uint64_t k = key(u, v);
        _link_queue[k].push_back(c);
        _inflight++;
        schedule_service(k, t);
    }

    // If the circuit is free and has queued chunks, start the head chunk.
    void schedule_service(uint64_t k, psec_t t) {
        if (_partial
            && (_pending.count(k) || _closing.count(k)
                || !_link_free.count(k))) {
            return;   // circuit not live / closing: chunks wait (partial)
        }
        std::vector<Chunk>& q = _link_queue[k];
        if (q.empty() || _link_busy.count(k)) {
            return;
        }
        psec_t start = std::max(std::max(t, _now), _link_free[k]);
        // Lowest chunk index first (ties by transfer id): concurrent
        // transfers on one circuit interleave fairly instead of the
        // first-injected transfer hogging the whole circuit.
        size_t pick = 0;
        for (size_t i = 1; i < q.size(); i++) {
            if (q[i].chunk_id < q[pick].chunk_id
                || (q[i].chunk_id == q[pick].chunk_id
                    && q[i].xfer_id < q[pick].xfer_id)) {
                pick = i;
            }
        }
        Chunk c = q[pick];
        q.erase(q.begin() + pick);
        psec_t ser = ser_ps(c.bytes);
        _link_busy.insert(k);
        _hop_bytes += c.bytes;
        if (_partial) {
            _serving[k] = c.bytes;
        }

        Ev freed;
        freed.t = start + ser;
        freed.seq = _seq++;
        freed.kind = EV_FREE;
        freed.link = k;
        _events.push(freed);

        Ev arrive;
        arrive.t = start + ser + _delay_ps;   // store-and-forward arrival
        arrive.seq = _seq++;
        arrive.kind = EV_ARRIVE;
        arrive.link = k;
        arrive.chunk = c;
        _events.push(arrive);
    }

    void on_link_free(const Ev& e) {
        _link_busy.erase(e.link);
        _link_free[e.link] = e.t;
        if (_partial) {
            partial_served(e.link);   // may fire a quota callback
        }
        schedule_service(e.link, e.t);
        if (_partial) {
            partial_check_close(e.link, e.t);
        }
    }

    void on_arrival(Ev& e) {
        Chunk c = e.chunk;
        _inflight--;
        Transfer& t = _transfers[c.xfer_id];
        c.next_hop++;
        if (c.next_hop + 1 < t.path.size()) {
            enqueue_on_link(t.path[c.next_hop], t.path[c.next_hop + 1],
                            c, e.t);
            return;
        }
        // Delivered at the application endpoint.
        t.delivered += c.bytes;
        _app_bytes += c.bytes;
        if (t.delivered >= t.bytes && !t.done) {
            t.done = true;
            if (_on_delivery) {
                _on_delivery(t);
            }
        }
    }

    uint32_t _n;
    uint32_t _degree;
    double _gbps;
    psec_t _delay_ps;
    psec_t _reconf_ps;
    bytes_t _chunk_bytes;

    psec_t _now = 0;
    uint64_t _seq = 0;
    bool _configured = false;
    bool _pay_initial = false;
    psec_t _ctl_ps = 0;
    uint32_t _reconfigs = 0;
    uint64_t _inflight = 0;
    bytes_t _hop_bytes = 0;
    bytes_t _app_bytes = 0;

    // partial-mode state (all empty/unused in legacy mode)
    bool _partial = false;
    uint32_t _port_reconfigs = 0;
    std::vector<psec_t> _tx_free_ps, _rx_free_ps;
    std::map<uint64_t, uint8_t> _pending;    // 0 = initial, 1 = paid
    std::set<uint64_t> _closing;
    std::map<uint64_t, bytes_t> _served, _serving;
    std::map<uint64_t,
             std::pair<bytes_t, std::function<void()> > > _quota;

    std::vector<std::pair<uint32_t, uint32_t> > _edges;
    std::map<uint64_t, psec_t> _link_free;
    std::map<uint64_t, std::vector<Chunk> > _link_queue;
    std::set<uint64_t> _link_busy;
    std::vector<Transfer> _transfers;
    std::priority_queue<Ev, std::vector<Ev>, std::greater<Ev> > _events;
    DeliveryCb _on_delivery;
};

}  // namespace optical

#endif  // OPTICAL_FABRIC_H
