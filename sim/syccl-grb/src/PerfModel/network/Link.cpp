/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PerfModel/Link.h"
#include "PerfModel/Event.h"
#include "PerfModel/NetworkFunction.h"
#include "PerfModel/Chunk.h"
#include "PerfModel/Route.h"
#include "PerfModel/Device.h"
#include "PerfModel/Type.h"
#include "Sketch/common.h"
#include <cassert>

using namespace Simulation;

void Link::link_become_free(void* const link_ptr) noexcept {
    assert(link_ptr != nullptr);

    // cast to Link*
    auto* const link = static_cast<Link*>(link_ptr);

    // set link free
    link->set_free();

    link->schedule_process_pending_transmission();
}

void Link::link_process_pending_transmission(void *link_ptr) noexcept {
    assert(link_ptr != nullptr);
    // cast to Link*
    auto* const link = static_cast<Link*>(link_ptr);
    link->process_pending_transmission();
}

void Link::set_event_queue(std::shared_ptr<EventQueue> event_queue_ptr) noexcept {
    assert(event_queue_ptr != nullptr);

    // set the event queue
    event_queue = event_queue_ptr;
}

void Link::set_simres(std::shared_ptr<Solver::SimResult> sol) noexcept {
    assert(sol != nullptr);
    this->sol = sol;
}

Link::Link(const Bandwidth bandwidth, const Latency latency, DeviceId src, DeviceId dst) noexcept
    : bandwidth(bandwidth), latency(latency), pending_chunks(), busy(false), src(src), dst(dst) {
    assert(bandwidth > 0);
    assert(latency >= 0);

    // convert bandwidth from GB/s to B/ns
    bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);
}

void Link::add_pending_chunk(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    pending_chunks.push(std::move(chunk));
}

void Link::process_pending_transmission() noexcept {
    if (!pending_chunk_exists() || busy) {
        return;
    }

    // get chunk to process according to prioity
    auto chunk = std::move(const_cast<std::unique_ptr<Chunk>&>(pending_chunks.top()));
    pending_chunks.pop();

    // service this chunk
    schedule_chunk_transmission(std::move(chunk));
}

void Link::schedule_process_pending_transmission() noexcept {
    // schedule process pending transmission event
    const auto current_time = Link::event_queue->get_current_time();
    auto* const link_ptr = static_cast<void*>(this);
    Link::event_queue->schedule_event(current_time, link_process_pending_transmission, link_ptr, EVENT_PRIO_LINK_PROCESS_TRANSMIT_QUEUE);
}

bool Link::pending_chunk_exists() const noexcept {
    // check pending chunks is not empty
    return !pending_chunks.empty();
}

void Link::set_busy() noexcept {
    // set busy to true
    busy = true;
}

void Link::set_free() noexcept {
    // set busy to false
    busy = false;
}

EventTime Link::serialization_delay(const ChunkSize chunk_size) const noexcept {
    assert(chunk_size > 0);

    // calculate serialization delay
    const auto delay = static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns;

    // return serialization delay in EventTime type
    return static_cast<EventTime>(delay);
}

EventTime Link::communication_delay(const ChunkSize chunk_size, send_type_e send_type) const noexcept {
    assert(chunk_size > 0);

    // calculate communication delay
    // We only calculate Beta cost for the switch->GPU link as the flit size in the network should be way smaller than the chunk size
    // So Multiple switch hops should only count for one Beta cost
    // TODO: We didn't consider non-uniform bandwith for multi-layered switches here and assumes a 1:1 Clos, so the switch->GPU link represents the accurate cost
    const auto delay = sendCountBeta(send_type) ? 
        latency + (static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns) :
        latency;

    // return communication delay in EventTime type
    return static_cast<EventTime>(delay);
}

ChunkSize Link::getSliceSize(const ChunkSize chunk_size) noexcept {
    return std::min(chunk_size, SLICE_SIZE_B);
}

void Link::schedule_chunk_transmission(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);
    // link should be free
    assert(!busy);
    // set link busy
    set_busy();

    // get metadata
    const auto chunk_size = chunk->get_size();
    const auto current_time = event_queue->get_current_time();
    const auto send_type = chunk->send_type;

    LOOOG(PERFMODEL, TRACE, "Schedule transmission of %s at %.2lf us",
            chunk->toStr(false).c_str(), (double)current_time / 1000.0);

    if (chunk->currentDevice()->type == NODE_GPU && chunk->slice_id == 0) {
        LOOOG(PERFMODEL, TRACE, "GPU %d scheduled transfer data %s ==> GPU %d at %lld ns",
              chunk->currentDevice()->local_id, chunk->datatoStr().c_str(), chunk->getNextGpu()->local_id, current_time);
        auto next_device = chunk->nextRouteNode()->device;
        int layer = -1;
        if (next_device->type == NODE_SWITCH_NV) {
            layer = 1;
        } else if (next_device->type == NODE_NIC) {
            layer = 3;
        } else if (next_device->type == NODE_SWITCH_NET) {
            layer = 3;
        }
        // The time that the first slice is actually sent should be the time that the algorithm sends the chunk
        assert(chunk->currentDevice()->local_id == chunk->src_gpu);
        assert(chunk->getNextGpu()->local_id == chunk->dst_gpu);
        sol->schedule->addMILPEvent(chunk->data_rep, chunk->src_gpu, chunk->dst_gpu, current_time);
        sol->schedule->events[chunk->data_rep].back().set_layer(layer);
    }

    if (chunk->currentDevice()->type == NODE_GPU) {
        sol->addSliceSentTime(chunk->data_rep, chunk->src_gpu, chunk->dst_gpu, current_time);
    }

    // schedule chunk arrival event
    const auto communication_time = communication_delay(chunk_size, send_type);

    const auto chunk_full_arrival_time = current_time + communication_time;

    chunk->arrival_time = chunk_full_arrival_time;
    if (chunk->nextRouteNode()->device->type == NODE_GPU) {
        sol->addSliceReceivedTime(chunk->data_rep, chunk->src_gpu, chunk->dst_gpu, chunk_full_arrival_time);
    }

    auto* const chunk_ptr = static_cast<void*>(chunk.release());
    Link::event_queue->schedule_event(chunk_full_arrival_time, Chunk::chunk_arrived_next_device, chunk_ptr, EVENT_PRIO_CHUNK_ARRIVAL);

    // schedule link free time, which doesn't account for the latency part in our analytical modeling(aligned with TE-CCL modeling)
    const auto serialization_time = serialization_delay(chunk_size);
    const auto link_free_time = current_time + serialization_time;
    auto* const link_ptr = static_cast<void*>(this);
    Link::event_queue->schedule_event(link_free_time, link_become_free, link_ptr, EVENT_PRIO_LINK_FREE);
}
