/**
 * @file dpdk_init.cc
 * @brief Initialization code for a DPDK port. This is a separate file because
 * it's used by both the eRPC library and the DPDK QP management daemon.
 */

#ifdef ERPC_DPDK

#include <rte_flow.h>
#include "dpdk_externs.h"
#include "dpdk_transport.h"

namespace erpc {

constexpr uint8_t DpdkTransport::kDefaultRssKey[];

void DpdkTransport::setup_phy_port(uint16_t phy_port, size_t numa_node,
                                   DpdkProcType proc_type) {
  _unused(proc_type);
  uint16_t num_ports = rte_eth_dev_count_avail();
  if (phy_port >= num_ports) {
    fprintf(stderr,
            "Error: Port %u (0-based) requested, but only %u DPDK ports "
            "available. Please ensure:\n",
            phy_port, num_ports);
    fprintf(stderr,
            "1. If you have a DPDK-capable port, ensure that (a) the NIC's "
            "NUMA node has huge pages, and (b) this process is not pinned "
            "(e.g., via numactl) to a different NUMA node than the NIC's.\n");

    const char *ld_library_path = getenv("LD_LIBRARY_PATH");
    const char *library_path = getenv("LIBRARY_PATH");

    fprintf(stderr,
            "2. Your LD_LIBRARY_PATH (= %s) and/or LIBRARY_PATH (= %s) "
            "contains the NIC's userspace libraries (e.g., libmlx5.so).\n",
            ld_library_path == nullptr ? "not set" : ld_library_path,
            library_path == nullptr ? "not set" : library_path);
    rt_assert(false);
  }

  rte_eth_dev_info dev_info;
  rte_eth_dev_info_get(phy_port, &dev_info);
  rt_assert(dev_info.rx_desc_lim.nb_max >= kNumRxRingEntries,
            "Device RX ring too small");
  rt_assert(dev_info.tx_desc_lim.nb_max >= kNumTxRingDesc,
            "Device TX ring too small");
  ERPC_INFO("Initializing port %u with driver %s\n", phy_port,
            dev_info.driver_name);

  // Create per-thread RX and TX queues
  rte_eth_conf eth_conf;
  memset(&eth_conf, 0, sizeof(eth_conf));

  // Raven patch: no catch-all RSS. On a bifurcated driver (mlx5), configuring
  // RSS makes the PMD install a catch-all steering rule at dev_start, which
  // hijacks *all* ingress traffic on the port -- including SSH, ICMP and eRPC's
  // own session-management UDP packets (SM uses a normal kernel socket, not
  // DPDK). eRPC then cannot complete its own handshake. Instead we leave the
  // port in MQ_RX_NONE and install one explicit flow rule per queue below,
  // matching only that queue's datapath UDP port; everything else stays with
  // the kernel. See install_udp_flow_rule().
  eth_conf.rxmode.mq_mode = ETH_MQ_RX_NONE;

  eth_conf.txmode.mq_mode = ETH_MQ_TX_NONE;
  eth_conf.txmode.offloads = kOffloads;

  int ret = rte_eth_dev_configure(phy_port, kMaxQueuesPerPort,
                                  kMaxQueuesPerPort, &eth_conf);
  rt_assert(ret == 0, "Ethdev configuration error: ", strerror(-1 * ret));

  // Raven patch: flow isolated mode. Without it, mlx5 installs its own default
  // steering rules at dev_start (unicast-MAC -> DPDK RX queues), which hijack
  // every packet addressed to this port -- SSH, ICMP, and eRPC's own
  // session-management UDP among them. In isolated mode the port receives only
  // what the application's own flow rules match, and all other traffic stays
  // with the kernel driver. This is exactly the bifurcated-driver coexistence
  // mode, and it must be requested after dev_configure but before dev_start.
  {
    rte_flow_error iso_err;
    memset(&iso_err, 0, sizeof(iso_err));
    ret = rte_flow_isolate(phy_port, 1, &iso_err);
    rt_assert(ret == 0,
              std::string("Failed to enable flow isolated mode: ") +
                  (iso_err.message == nullptr ? "?" : iso_err.message));
  }

  // Set up all RX and TX queues and start the device. This can't be done later
  // on a per-thread basis since we must start the device to use any queue.
  // Once the device is started, more queues cannot be added without stopping
  // and reconfiguring the device.
  for (size_t i = 0; i < kMaxQueuesPerPort; i++) {
    const std::string pname = get_mempool_name(phy_port, i);
    rte_mempool *mempool =
        rte_pktmbuf_pool_create(pname.c_str(), kNumMbufs, 0 /* cache */,
                                0 /* priv size */, kMbufSize, numa_node);
    rt_assert(mempool != nullptr, "Mempool create failed: " + dpdk_strerror());

    rte_eth_rxconf eth_rx_conf;
    memset(&eth_rx_conf, 0, sizeof(eth_rx_conf));
    eth_rx_conf.rx_thresh.pthresh = 8;

    int ret = rte_eth_rx_queue_setup(phy_port, i, kNumRxRingEntries, numa_node,
                                     &eth_rx_conf, mempool);
    rt_assert(ret == 0, "Failed to setup RX queue: " + std::to_string(i) +
                            ". Error " + strerror(-1 * ret));

    rte_eth_txconf eth_tx_conf;
    memset(&eth_tx_conf, 0, sizeof(eth_tx_conf));
    eth_tx_conf.tx_thresh.pthresh = 32;
    eth_tx_conf.offloads = eth_conf.txmode.offloads;

    ret = rte_eth_tx_queue_setup(phy_port, i, kNumTxRingDesc, numa_node,
                                 &eth_tx_conf);
    rt_assert(ret == 0, "Failed to setup TX queue: " + std::to_string(i));
  }

  rte_eth_dev_start(phy_port);

  // Raven patch: steer only our own datapath UDP ports into DPDK queues.
  // Must be after dev_start (mlx5 rejects flow creation on a stopped port).
  for (size_t i = 0; i < kMaxQueuesPerPort; i++) {
    install_udp_flow_rule(phy_port, i,
                          static_cast<uint16_t>(kBaseEthUDPPort + i));
  }
}

// Raven patch: modern rte_flow replacement for the (dead, DPDK<=20.11-only)
// helpers in dpdk_flow_rules.cc. Matches ETH/IPv4/UDP with an exact destination
// port and directs those packets to \p qp_id. Unmatched traffic is left to the
// kernel, which is what makes DPDK coexist with the kernel networking stack on
// the same bifurcated (mlx5) port.
void DpdkTransport::install_udp_flow_rule(uint16_t phy_port, size_t qp_id,
                                          uint16_t udp_dst_port) {
  rte_flow_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.ingress = 1;

  rte_flow_item_udp udp_spec, udp_mask;
  memset(&udp_spec, 0, sizeof(udp_spec));
  memset(&udp_mask, 0, sizeof(udp_mask));
  udp_spec.hdr.dst_port = rte_cpu_to_be_16(udp_dst_port);
  udp_mask.hdr.dst_port = 0xffff;

  rte_flow_item pattern[4];
  memset(pattern, 0, sizeof(pattern));
  pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
  pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
  pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
  pattern[2].spec = &udp_spec;
  pattern[2].mask = &udp_mask;
  pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

  rte_flow_action_queue queue_action;
  memset(&queue_action, 0, sizeof(queue_action));
  queue_action.index = static_cast<uint16_t>(qp_id);

  rte_flow_action actions[2];
  memset(actions, 0, sizeof(actions));
  actions[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
  actions[0].conf = &queue_action;
  actions[1].type = RTE_FLOW_ACTION_TYPE_END;

  rte_flow_error error;
  memset(&error, 0, sizeof(error));
  if (rte_flow_validate(phy_port, &attr, pattern, actions, &error) != 0) {
    rt_assert(false, std::string("Flow rule validation failed for UDP port ") +
                         std::to_string(udp_dst_port) + ": " +
                         (error.message == nullptr ? "?" : error.message));
  }
  rte_flow *flow = rte_flow_create(phy_port, &attr, pattern, actions, &error);
  rt_assert(flow != nullptr,
            std::string("Flow rule creation failed for UDP port ") +
                std::to_string(udp_dst_port) + ": " +
                (error.message == nullptr ? "?" : error.message));
}

}  // namespace erpc

#endif
