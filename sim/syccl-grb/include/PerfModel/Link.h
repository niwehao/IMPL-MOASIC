/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Chunk.h"
#include "PerfModel/EventQueue.h"
#include "PerfModel/Type.h"
#include "Solver/common.h"
#include <memory>
#include <queue>

namespace Simulation {
    /**
 * Link models physical links between two devices.
 */
    class Link {
    public:
        /**
   * Callback to be called when a link becomes free.
   *  - If the link has pending chunks, process the first one.
   *  - If the link has no pending chunks, set the link as free.
   *
   * @param link_ptr pointer to the link that becomes free
   */
        static void link_become_free(void* link_ptr) noexcept;

        /**
   * Set the event queue to be used by the link.
   *
   * @param event_queue_ptr pointer to the event queue
   */
        void set_event_queue(std::shared_ptr<EventQueue> event_queue_ptr) noexcept;
        void set_simres(std::shared_ptr<Solver::SimResult> schedule) noexcept;

        static void link_process_pending_transmission(void* link_ptr) noexcept;
        /**
   * Constructor.
   *
   * @param bandwidth bandwidth of the link
   * @param latency latency of the link
   */
        Link(Bandwidth bandwidth, Latency latency, DeviceId src, DeviceId dst) noexcept;

        void add_pending_chunk(std::unique_ptr<Chunk> chunk) noexcept;

        /**
   * Dequeue and try to send the first pending chunk
   * in the pending chunks list.
   */
        void process_pending_transmission() noexcept;

        void schedule_process_pending_transmission() noexcept;

        [[nodiscard]] bool pending_chunk_exists() const noexcept;

        void set_busy() noexcept;

        void set_free() noexcept;

        Bandwidth getBandwidthBpns() const noexcept { return bandwidth_Bpns; }
        InvBandwidth getInvBandwidthnspB() const noexcept { return 1.0 / bandwidth_Bpns; }
        Latency getLatencyns() const noexcept { return latency; }
        Latency getLatency() const noexcept { return latency; }
        DeviceId getSrc() const noexcept { return src; }
        DeviceId getDst() const noexcept { return dst; }
        static ChunkSize getSliceSize(const ChunkSize chunk_size) noexcept;
        EventTime getCurrentTime() const noexcept { return event_queue->get_current_time(); }

    private:
        /// event queue Link uses to schedule events
        std::shared_ptr<EventQueue> event_queue;
        std::shared_ptr<Solver::SimResult> sol;

        /// bandwidth of the link in GB/s
        Bandwidth bandwidth;

        /// bandwidth of the link in B/ns, used in actual computation
        Bandwidth bandwidth_Bpns;

        /// latency of the link in ns
        Latency latency;

        /// queue of pending chunks
        std::priority_queue<std::unique_ptr<Chunk>, std::vector<std::unique_ptr<Chunk>>, ChunkPriority> pending_chunks;

        /// flag to indicate if the link is busy
        bool busy;

        DeviceId src, dst;

        /**
   * Compute the serialization delay of a chunk on the link.
   * i.e., serialization delay = (chunk size) / (link bandwidth)
   *
   * @param chunk_size size of the target chunk
   * @return serialization delay of the chunk
   */
        [[nodiscard]] EventTime serialization_delay(ChunkSize chunk_size) const noexcept;

        /**
   * Compute the communication delay of a chunk.
   * i.e., communication delay = (link latency) + (serialization delay)
   *
   * @param chunk_size size of the target chunk
   * @return communication delay of the chunk
   */
        [[nodiscard]] EventTime communication_delay(ChunkSize chunk_size, send_type_e send_type) const noexcept;

        /**
   * Schedule the transmission of a chunk.
   * - Set the link as busy.
   * - Link becomes free after the serialization delay.
   * - Chunk arrives next node after the communication delay.
   *
   * @param chunk chunk to be transmitted
   */
        void schedule_chunk_transmission(std::unique_ptr<Chunk> chunk) noexcept;
    };

}  // namespace Simulation
