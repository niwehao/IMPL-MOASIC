/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "Sketch/Topology.h"
#include "PerfModel/EventQueue.h"
#include "PerfModel/Chunk.h"
#include "PerfModel/Device.h"
#include "PerfModel/Type.h"
#include "PerfModel/TopologyLayer.h"
#include "Solver/common.h"
#include <memory>
#include <vector>
namespace Simulation {

     class NPUInfo {
     public:
          std::shared_ptr<Device> device;
          std::shared_ptr<Device> NIC, NVswitch;
          std::vector<std::shared_ptr<Device>> layer_switches;
          int machine_id;
          NPUInfo(std::shared_ptr<Device> device): device(device), NIC(nullptr), NVswitch(nullptr), layer_switches(), machine_id(-1) {}
          std::string toStr();
     };

    /**
 * Topology abstracts a network topology.
 */
    class Topology {
    public:
        void set_event_queue(std::shared_ptr<EventQueue> event_queue) noexcept;
        void set_simres(std::shared_ptr<Solver::SimResult> res) noexcept;
        void set_num_chunks(int *num_chunks) noexcept;

        Topology() noexcept;

        void send(std::unique_ptr<Chunk> chunk) noexcept;

        [[nodiscard]] int get_npus_count() const noexcept;
        [[nodiscard]] int get_devices_count() const noexcept;

        std::shared_ptr<Device> getNpuDevice(int npu_id) noexcept;

     // find the full routing path between two NPUs including the source and destination NPUs
        std::vector<std::shared_ptr<Device>> getNpuRoute(int src_npu_id, int dst_npu_id, int layer, bool PXN) noexcept;

        std::vector<NPUInfo> npus;
        // devices include npus
        std::vector<std::shared_ptr<Device>> devices;
        // (deviceId, GPUid) -> deviceId for next hop
        // std::vector<std::vector<std::shared_ptr<Device>>> npu_next_hop;
        std::map<node_type_e, int> dev_count;
        topo_info_t topo_info;

        void addDevice(node_type_e type) noexcept;

        void resetDevices();

        /**
   * Connect src -> dest with the given bandwidth and latency.
   * (i.e., a `Link` gets constructed between the two npus)
   *
   * if bidirectional=true, dest -> src connection is also established.
   *
   * @param src src device id
   * @param dest dest device id
   * @param bandwidth bandwidth of link
   * @param latency latency of link
   * @param bidirectional true if connection is bidirectional, false otherwise
   */
        void connect(DeviceId src, DeviceId dest, Bandwidth bandwidth, Latency latency,
                     std::map<std::shared_ptr<Device>, std::map<std::shared_ptr<Device>, std::shared_ptr<Link>>>& links,
                     bool bidirectional = true) noexcept;

        void generateLayerInfo();

        void print();

        std::shared_ptr<Topology> clone();
    };
} // namespace Simulation
