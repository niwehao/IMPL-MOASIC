/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Type.h"
#include "Solver/common.h"
#include <memory>
#include <string>

namespace Simulation {

    /**
 * Chunk class represents a chunk.
 * Chunk is a basic unit of transmission.
 */
    class Chunk {
    public:
        /**
   * Callback to be invoked when a chunk arrives at the next device.
   *   - if the chunk arrived at its destination, the final callback is invoked
   *   - if not, the chunk is sent to the next device as designated by the route
   *
   * @param chunk_ptr: pointer to the chunk that's arrived at the next device
   */
     static void chunk_arrived_next_device(void* chunk_ptr) noexcept;

     Chunk(ChunkSize chunk_size, std::shared_ptr<Route> route, std::shared_ptr<RouteNode> current_node, std::set<Solver::SrcChunk> &data, Solver::SrcChunk data_rep, int slice_id, int nslices, int id, EventTime arrival_time, int src_gpu, int dst_gpu) noexcept;

     [[nodiscard]] std::shared_ptr<RouteNode> currentRouteNode() const noexcept;
     void scheduleNextRouteNode(std::shared_ptr<RouteNode> nxt, int epoch, send_type_e type) noexcept;
     [[nodiscard]] std::shared_ptr<RouteNode> nextRouteNode() const noexcept;
     [[nodiscard]] std::shared_ptr<Device> currentDevice() const noexcept;
     void mark_arrived_next_device() noexcept;
     [[nodiscard]] bool arrived_dest() const noexcept;
     [[nodiscard]] ChunkSize get_size() const noexcept;
     [[nodiscard]] std::unique_ptr<Chunk> copyChunk(int new_id) noexcept;

     [[nodiscard]] std::string toStr(bool verbose = true) const;
     std::string datatoStr() const;
     [[nodiscard]] double remainingLongestPath();
     [[nodiscard]] std::shared_ptr<Device> getNextGpu() const noexcept;

     bool isLastSlice() const noexcept;
     bool isFirstSlice() const noexcept;

     // id of the chunk, slices from the same chunk have the same id
     int id, parent_id;
     std::set<Solver::SrcChunk> data;
     // To mark chunks belonging to the same tree so that events are sorted properly
     Solver::SrcChunk data_rep;
     // priority/epoch of send from current_node to next_node
     int send_epoch = -1;
     send_type_e send_type;
     // only the last slice signals receival
     int slice_id, nslices;
     EventTime arrival_time;
     // The src and dst gpu for the chunk's current send(on the way), not to be confused with the global routing information in the RouteNodes
     // This is for generating the final schedule that doesn't contain the intermediate nodes
     int src_gpu, dst_gpu;

    private:
        /// size of the chunk
        ChunkSize chunk_size;
        
        std::shared_ptr<Route> route;
        /// Transient state from the sendinstruction
        // current and (scheduled) next RouteNode of the chunk
        std::shared_ptr<RouteNode> current_node, next_node;
    };

    const double PRIO_EPS = 1e-8;
    struct ChunkPriority {
        bool operator()(const std::unique_ptr<Chunk>& lhs, const std::unique_ptr<Chunk>& rhs) const;
    };

}  // namespace Simulation
