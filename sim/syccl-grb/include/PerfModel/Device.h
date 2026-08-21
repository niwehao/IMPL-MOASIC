/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "PerfModel/Type.h"
#include <map>
#include <memory>
#include <string>
#include "Solver/common.h"
#include "Sketch/Topology.h"

namespace Simulation {

    /**
 * Device class represents a single device in the network.
 * Device is usually an NPU or a switch.
 */
    class Device {
    public:
        explicit Device(DeviceId id, DeviceId local_id) noexcept;

        [[nodiscard]] DeviceId get_id() const noexcept;
        [[nodiscard]] DeviceId get_local_id() const noexcept { return local_id; }

        void schedule_send(std::unique_ptr<Chunk> chunk) noexcept;

        std::shared_ptr<Link> connect(DeviceId id, Bandwidth bandwidth, Latency latency) noexcept;

        node_type_e type;

        std::string toStr() const;

        DeviceId device_id, local_id;

        std::map<DeviceId, std::shared_ptr<Link>> links;

        [[nodiscard]] bool connected(DeviceId dest) const noexcept;

        std::vector<std::set<Solver::SrcChunk>> received_data;
        void clear_received_data() noexcept;
        void clear_chunk_id_map() noexcept;
        void add_received_data(const std::set<Solver::SrcChunk> &data) noexcept;
        // Record info of Chunks that are partially sent
        // (chunk_id, dest) -> copied chunk_ids
        std::map<std::pair<int, std::shared_ptr<RouteNode>>, int> outstanding_chunk_id_map;

        int *num_chunks = nullptr;
    };

}  // namespace Simulation