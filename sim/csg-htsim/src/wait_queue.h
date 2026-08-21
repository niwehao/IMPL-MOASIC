// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef WAIT_QUEUE_H
#define WAIT_QUEUE_H

#include "ecn.h"
#include "queue.h"

#include <functional>

class SmartNode;

class SmartQueue : public Queue {
public:
    typedef std::function<void(SmartNode*)> Activator;

    SmartQueue(linkspeed_bps bitrate, mem_b maxsize, EventList& eventlist,
               QueueLogger* logger)
        : Queue(bitrate, maxsize, eventlist, logger), _owner(NULL),
          _waiting(false), _threshold(0), _ecn_threshold(maxsize * 8 / 10),
          _pause_threshold(maxsize * 9 / 10),
          _resume_threshold(maxsize * 8 / 10), _paused(false),
          _ecn_marks(0), _pause_events(0) {
    }

    void set_owner(SmartNode* owner) {
        _owner = owner;
    }

    void set_activator(const Activator& activator) {
        _activator = activator;
    }

    void set_congestion_thresholds(mem_b ecn_threshold,
                                   mem_b pause_threshold) {
        if (ecn_threshold > 0) {
            _ecn_threshold = ecn_threshold;
        }
        if (pause_threshold > 0) {
            _pause_threshold = pause_threshold;
        }
        if (_ecn_threshold > _maxsize) {
            _ecn_threshold = _maxsize;
        }
        if (_pause_threshold > _maxsize) {
            _pause_threshold = _maxsize;
        }
        if (_pause_threshold < _ecn_threshold) {
            _pause_threshold = _ecn_threshold;
        }
        _resume_threshold = _ecn_threshold;
    }

    bool is_paused() const {
        return _paused;
    }

    uint64_t ecn_marks() const {
        return _ecn_marks;
    }

    uint64_t pause_events() const {
        return _pause_events;
    }

    void wait_for_space(SmartNode* node, mem_b threshold) {
        if (!_owner) {
            _owner = node;
        }

        if (!_waiting || threshold < _threshold) {
            _threshold = threshold;
        }
        _waiting = true;

        if (available_bytes() >= _threshold) {
            notify_waiter();
        }
    }

protected:
    void receivePacket(Packet& pkt) override {
        mem_b projected_size = _queuesize + pkt.size();
        if (projected_size > _maxsize) {
            if (_logger) {
                _logger->logQueue(*this, QueueLogger::PKT_DROP, pkt);
            }
            pkt.flow().logTraffic(pkt, *this, TrafficLogger::PKT_DROP);
            pkt.free();
            _num_drops++;
            return;
        }

        if (projected_size > _ecn_threshold) {
            pkt.set_flags(pkt.flags() | ECN_CE);
            _ecn_marks++;
        }

        if (projected_size > _pause_threshold && !_paused) {
            _paused = true;
            _pause_events++;
        }

        pkt.flow().logTraffic(pkt, *this, TrafficLogger::PKT_ARRIVE);

        bool queueWasEmpty = _enqueued.empty();
        Packet* pkt_p = &pkt;
        _enqueued.push(pkt_p);
        _queuesize += pkt.size();
        if (_logger) {
            _logger->logQueue(*this, QueueLogger::PKT_ENQUEUE, pkt);
        }

        if (queueWasEmpty) {
            assert(_enqueued.size() == 1);
            beginService();
        }
    }

    void completeService() override {
        mem_b before = available_bytes();
        Queue::completeService();
        mem_b after = available_bytes();

        if (_paused && _queuesize <= _resume_threshold) {
            _paused = false;
            notify_waiter();
        }

        if (_waiting && before < _threshold && after >= _threshold) {
            notify_waiter();
        }
    }

private:
    mem_b available_bytes() const {
        mem_b used = queuesize();
        mem_b limit = maxsize();
        return used < limit ? limit - used : 0;
    }

    void notify_waiter() {
        if (!_waiting) {
            return;
        }

        _waiting = false;
        if (_owner && _activator) {
            _activator(_owner);
        }
    }

    SmartNode* _owner;
    bool _waiting;
    mem_b _threshold;
    Activator _activator;
    mem_b _ecn_threshold;
    mem_b _pause_threshold;
    mem_b _resume_threshold;
    bool _paused;
    uint64_t _ecn_marks;
    uint64_t _pause_events;
};

#endif
