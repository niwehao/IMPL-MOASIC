#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time


FLOW_RE = re.compile(
    r"Chunk\s+(?P<chunk>\d+)\s+from\s+(?P<source>\d+)\s+traveled\s+over\s+"
    r"(?P<src>\d+)->(?P<dst>\d+)"
    r"(?:\s+with\s+volume\s+(?P<volume>[0-9.]+))?"
    r"(?:\s+at|\s+in)\s+epoch\s+(?P<epoch>\d+)"
)

DEMAND_RE = re.compile(
    r"Demand\s+at\s+(?P<dst>\d+)\s+for\s+chunk\s+(?P<chunk>\d+)\s+"
    r"from\s+(?P<source>\d+)\s+met\s+by\s+epoch\s+(?P<epoch>\d+)"
)

PATH_HOP_RE = re.compile(
    r"(?P<src>\d+)->(?P<dst>\d+)\s+with\s+volume\s+"
    r"(?P<volume>[0-9.eE+-]+)"
    r"\s+in\s+epoch\s+(?P<epoch>\d+)"
)

NDV2_LIKE_RAW_CODES = [
    [0, 23, 46, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [23, 0, 46, 23, 0, 46, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0],
    [46, 46, 0, 23, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [46, 23, 23, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0],
    [23, 0, 0, 0, 0, 23, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 46, 0, 0, 23, 0, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 23, 0, 46, 46, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 46, 46, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 46, 46, 23, 0, 0, 0],
    [107, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 23, 0, 46, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 23, 0, 0, 23, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 46, 23, 23, 0, 0, 0, 0, 46],
    [0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 23, 46, 46],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 23, 0, 46, 23],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 46, 0, 23],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 23, 23, 0],
]

SYCCL_A100_GPU_COUNT = 32
SYCCL_A100_HOST_COUNT = 4
SYCCL_A100_GPUS_PER_HOST = 8
SYCCL_A100_NICS_PER_HOST = 4
SYCCL_A100_LEAF_COUNT = 2
SYCCL_A100_SPINE_COUNT = 1
MINI_A100_GPU_COUNT = 16
MINI_A100_HOST_COUNT = 4
MINI_A100_GPUS_PER_HOST = 4
MINI_A100_NICS_PER_HOST = 2
MINI_A100_LEAF_COUNT = 2
MINI_A100_SPINE_COUNT = 1


def ndv2_like_code_to_gbps(code):
    if code == 23:
        return 400.0
    if code == 46:
        return 200.0
    if code == 107:
        return 100.0
    return 0.0


def ndv2_like_code_to_alpha(code):
    if code == 107:
        return 1.3e-6
    if code:
        return 0.7e-6
    return -1.0


def ndv2_like_capacity_codes():
    node_count = 16
    matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
    for i, row in enumerate(NDV2_LIKE_RAW_CODES):
        for j, value in enumerate(row):
            matrix[j][i] = value
    return matrix


def syccl_a100_gpu_id(host, local_gpu):
    return host * SYCCL_A100_GPUS_PER_HOST + local_gpu


def syccl_a100_nic_id(host, local_nic):
    return SYCCL_A100_GPU_COUNT + host * SYCCL_A100_NICS_PER_HOST + local_nic


def syccl_a100_nvswitch_id(host):
    return (SYCCL_A100_GPU_COUNT
            + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
            + host)


def syccl_a100_leaf_id(leaf):
    return (SYCCL_A100_GPU_COUNT
            + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
            + SYCCL_A100_HOST_COUNT
            + leaf)


def syccl_a100_spine_id(spine):
    return (SYCCL_A100_GPU_COUNT
            + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
            + SYCCL_A100_HOST_COUNT
            + SYCCL_A100_LEAF_COUNT
            + spine)


def syccl_a100_total_nodes():
    return (SYCCL_A100_GPU_COUNT
            + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
            + SYCCL_A100_HOST_COUNT
            + SYCCL_A100_LEAF_COUNT
            + SYCCL_A100_SPINE_COUNT)


def add_directed_edge(edges, node_tx_gbps, left, right, gbps, alpha):
    edges.append([left, right, gbps, alpha])
    edges.append([right, left, gbps, alpha])
    node_tx_gbps[left] = max(node_tx_gbps[left], gbps)
    node_tx_gbps[right] = max(node_tx_gbps[right], gbps)


def syccl_a100_graph():
    node_count = syccl_a100_total_nodes()
    edges = []
    node_tx_gbps = [0.0 for _ in range(node_count)]

    for host in range(SYCCL_A100_HOST_COUNT):
        nvswitch = syccl_a100_nvswitch_id(host)
        for local_gpu in range(SYCCL_A100_GPUS_PER_HOST):
            add_directed_edge(edges, node_tx_gbps,
                              syccl_a100_gpu_id(host, local_gpu),
                              nvswitch, 2400.0, 9e-6)

        gpus_per_nic = SYCCL_A100_GPUS_PER_HOST // SYCCL_A100_NICS_PER_HOST
        for local_nic in range(SYCCL_A100_NICS_PER_HOST):
            nic = syccl_a100_nic_id(host, local_nic)
            for offset in range(gpus_per_nic):
                local_gpu = local_nic * gpus_per_nic + offset
                add_directed_edge(edges, node_tx_gbps,
                                  syccl_a100_gpu_id(host, local_gpu),
                                  nic, 180.0, 0.0)

    hosts_per_leaf = SYCCL_A100_HOST_COUNT // SYCCL_A100_LEAF_COUNT
    for leaf in range(SYCCL_A100_LEAF_COUNT):
        leaf_node = syccl_a100_leaf_id(leaf)
        for host in range(leaf * hosts_per_leaf,
                          (leaf + 1) * hosts_per_leaf):
            for local_nic in range(SYCCL_A100_NICS_PER_HOST):
                add_directed_edge(edges, node_tx_gbps,
                                  syccl_a100_nic_id(host, local_nic),
                                  leaf_node, 180.0, 25e-6)

    for leaf in range(SYCCL_A100_LEAF_COUNT):
        for spine in range(SYCCL_A100_SPINE_COUNT):
            add_directed_edge(edges, node_tx_gbps,
                              syccl_a100_leaf_id(leaf),
                              syccl_a100_spine_id(spine),
                              2880.0, 25e-6)

    switch_indices = list(range(SYCCL_A100_GPU_COUNT, node_count))
    return {
        "format": "htsim-syccl-a100-nic-graph-v1",
        "topology_model": "nic_graph",
        "node_count": node_count,
        "endpoint_indices": list(range(SYCCL_A100_GPU_COUNT)),
        "switch_indices": switch_indices,
        "directed": True,
        "rate_unit": "gbps",
        "alpha_unit": "seconds",
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def syccl_a100_1host_graph():
    node_count = 9
    edges = []
    node_tx_gbps = [0.0 for _ in range(node_count)]
    for gpu in range(8):
        add_directed_edge(edges, node_tx_gbps, gpu, 8, 2400.0, 9e-6)
    return {
        "format": "htsim-syccl-a100-1host-nic-graph-v1",
        "topology_model": "nic_graph",
        "node_count": node_count,
        "endpoint_indices": list(range(8)),
        "switch_indices": [8],
        "directed": True,
        "rate_unit": "gbps",
        "alpha_unit": "seconds",
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def mini_a100_gpu_id(host, local_gpu):
    return host * MINI_A100_GPUS_PER_HOST + local_gpu


def mini_a100_nic_id(host, local_nic):
    return MINI_A100_GPU_COUNT + host * MINI_A100_NICS_PER_HOST + local_nic


def mini_a100_nvswitch_id(host):
    return (MINI_A100_GPU_COUNT
            + MINI_A100_HOST_COUNT * MINI_A100_NICS_PER_HOST
            + host)


def mini_a100_leaf_id(leaf):
    return (MINI_A100_GPU_COUNT
            + MINI_A100_HOST_COUNT * MINI_A100_NICS_PER_HOST
            + MINI_A100_HOST_COUNT
            + leaf)


def mini_a100_spine_id(spine):
    return (MINI_A100_GPU_COUNT
            + MINI_A100_HOST_COUNT * MINI_A100_NICS_PER_HOST
            + MINI_A100_HOST_COUNT
            + MINI_A100_LEAF_COUNT
            + spine)


def mini_a100_total_nodes():
    return (MINI_A100_GPU_COUNT
            + MINI_A100_HOST_COUNT * MINI_A100_NICS_PER_HOST
            + MINI_A100_HOST_COUNT
            + MINI_A100_LEAF_COUNT
            + MINI_A100_SPINE_COUNT)


def mini_a100_16g_graph():
    node_count = mini_a100_total_nodes()
    edges = []
    node_tx_gbps = [0.0 for _ in range(node_count)]
    for host in range(MINI_A100_HOST_COUNT):
        nvswitch = mini_a100_nvswitch_id(host)
        for local_gpu in range(MINI_A100_GPUS_PER_HOST):
            add_directed_edge(edges, node_tx_gbps,
                              mini_a100_gpu_id(host, local_gpu),
                              nvswitch, 2400.0, 9e-6)

        gpus_per_nic = MINI_A100_GPUS_PER_HOST // MINI_A100_NICS_PER_HOST
        for local_nic in range(MINI_A100_NICS_PER_HOST):
            nic = mini_a100_nic_id(host, local_nic)
            for offset in range(gpus_per_nic):
                local_gpu = local_nic * gpus_per_nic + offset
                add_directed_edge(edges, node_tx_gbps,
                                  mini_a100_gpu_id(host, local_gpu),
                                  nic, 180.0, 0.0)

    hosts_per_leaf = MINI_A100_HOST_COUNT // MINI_A100_LEAF_COUNT
    for leaf in range(MINI_A100_LEAF_COUNT):
        leaf_node = mini_a100_leaf_id(leaf)
        for host in range(leaf * hosts_per_leaf,
                          (leaf + 1) * hosts_per_leaf):
            for local_nic in range(MINI_A100_NICS_PER_HOST):
                add_directed_edge(edges, node_tx_gbps,
                                  mini_a100_nic_id(host, local_nic),
                                  leaf_node, 180.0, 25e-6)

    for leaf in range(MINI_A100_LEAF_COUNT):
        for spine in range(MINI_A100_SPINE_COUNT):
            add_directed_edge(edges, node_tx_gbps,
                              mini_a100_leaf_id(leaf),
                              mini_a100_spine_id(spine),
                              2880.0, 25e-6)

    switch_indices = list(range(MINI_A100_GPU_COUNT, node_count))
    return {
        "format": "htsim-mini-a100-16g-nic-graph-v1",
        "topology_model": "nic_graph",
        "node_count": node_count,
        "endpoint_indices": list(range(MINI_A100_GPU_COUNT)),
        "switch_indices": switch_indices,
        "directed": True,
        "rate_unit": "gbps",
        "alpha_unit": "seconds",
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def mini_a100_16g_endpoint_graph(alpha_override=None):
    node_count = MINI_A100_GPU_COUNT
    edges = []
    node_tx_gbps = [0.0 for _ in range(node_count)]
    for src in range(node_count):
        src_host = src // MINI_A100_GPUS_PER_HOST
        src_leaf = src_host // (MINI_A100_HOST_COUNT // MINI_A100_LEAF_COUNT)
        for dst in range(node_count):
            if src == dst:
                continue
            dst_host = dst // MINI_A100_GPUS_PER_HOST
            dst_leaf = dst_host // (MINI_A100_HOST_COUNT // MINI_A100_LEAF_COUNT)
            if src_host == dst_host:
                gbps = 2400.0
                alpha = 9e-6
            elif src_leaf == dst_leaf:
                gbps = 180.0
                alpha = 25e-6
            else:
                gbps = 180.0
                alpha = 50e-6
            if alpha_override is not None:
                alpha = float(alpha_override)
            edges.append([src, dst, gbps, alpha])
            node_tx_gbps[src] = max(node_tx_gbps[src], gbps)
    return {
        "format": "htsim-mini-a100-16g-endpoint-projection-v1",
        "topology_model": "nic_graph",
        "node_count": node_count,
        "endpoint_indices": list(range(node_count)),
        "switch_indices": [],
        "directed": True,
        "rate_unit": "gbps",
        "alpha_unit": "seconds",
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def clos_graph(node_count, leaf_count, spine_count, link_rate_gbps, alpha,
               uplink_gbps=0.0):
    if node_count <= 0 or leaf_count <= 0 or spine_count <= 0:
        raise ValueError("clos node_count, leaf_count, and spine_count must be positive")
    first_leaf = node_count
    first_spine = first_leaf + leaf_count
    total_nodes = node_count + leaf_count + spine_count
    hosts_per_leaf = (node_count + leaf_count - 1) // leaf_count
    if uplink_gbps <= 0.0:
        # Match htsim build_two_layer_clos_graph default: 2:1 leaf
        # oversubscription split across spine links.
        uplink_gbps = link_rate_gbps * hosts_per_leaf / (2.0 * spine_count)
    edges = []
    node_tx_gbps = [0.0 for _ in range(total_nodes)]

    for host in range(node_count):
        leaf = first_leaf + min(host // hosts_per_leaf, leaf_count - 1)
        add_directed_edge(edges, node_tx_gbps, host, leaf, link_rate_gbps, alpha)

    for leaf in range(leaf_count):
        for spine in range(spine_count):
            add_directed_edge(
                edges,
                node_tx_gbps,
                first_leaf + leaf,
                first_spine + spine,
                uplink_gbps,
                alpha,
            )

    return {
        "format": "htsim-two-layer-clos-v1",
        "topology_model": "nic_graph",
        "node_count": total_nodes,
        "htsim_node_count": node_count,
        "endpoint_indices": list(range(node_count)),
        "switch_indices": list(range(first_leaf, total_nodes)),
        "directed": True,
        "rate_unit": "gbps",
        "alpha_unit": "seconds",
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def te_trap_same_cluster(left, right):
    return (left < 8 and right < 8) or (left >= 8 and right >= 8)


def te_trap_paired_cross_edge(left, right):
    if te_trap_same_cluster(left, right):
        return False
    low = min(left, right)
    high = max(left, right)
    return low < 8 and high == low + 8


def te_trap_edge_gbps(left, right):
    if te_trap_same_cluster(left, right):
        return 400.0
    if te_trap_paired_cross_edge(left, right):
        return 400.0
    return 5.0


def regular_edges(node_count, degree):
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if degree >= node_count:
        raise ValueError("degree must be smaller than node_count")
    if (node_count * degree) % 2 != 0:
        raise ValueError("node_count * degree must be even")
    if degree % 2 == 1 and node_count % 2 == 1:
        raise ValueError("odd degree requires even node_count")

    offsets = []
    seen = set()
    max_offset = (node_count - 1) // 2
    for i in range(degree // 2):
        exponent = i / (degree // 2) if degree // 2 else 0
        offset = int((node_count ** exponent) + 0.5)
        offset = max(1, min(offset, max_offset))
        seen.add(offset)
    offset = 1
    while len(seen) < degree // 2 and offset <= max_offset:
        seen.add(offset)
        offset += 1
    offsets = sorted(seen)

    edges = set()
    for offset in offsets:
        for node in range(node_count):
            peer = (node + offset) % node_count
            edges.add((min(node, peer), max(node, peer)))
    if degree % 2:
        offset = node_count // 2
        for node in range(node_count):
            peer = (node + offset) % node_count
            edges.add((min(node, peer), max(node, peer)))
    return sorted(edges)


def is_regular_ring_edge(node_count, left, right):
    if node_count <= 1:
        return False
    diff = abs(int(left) - int(right))
    return diff == 1 or diff == node_count - 1


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def file_digest(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_graph(topo):
    name = topo.get("name", "regular")
    if name in ("te_trap", "TETrap", "te-trap"):
        node_count = int(topo.get("node_count", 16))
        if node_count != 16:
            raise ValueError("te_trap topology requires node_count=16")
        alpha = float(topo.get("alpha", 0.7e-6))
        edges = []
        node_tx_gbps = [0.0 for _ in range(node_count)]
        for left in range(node_count):
            for right in range(left + 1, node_count):
                gbps = te_trap_edge_gbps(left, right)
                edges.append([left, right, gbps, alpha])
                edges.append([right, left, gbps, alpha])
                node_tx_gbps[left] = max(node_tx_gbps[left], gbps)
                node_tx_gbps[right] = max(node_tx_gbps[right], gbps)
        return {
            "format": "htsim-te-trap-nic-graph-v1",
            "topology_model": "nic_graph",
            "node_count": node_count,
            "endpoint_indices": list(range(node_count)),
            "switch_indices": [],
            "directed": True,
            "rate_unit": "gbps",
            "alpha_unit": "seconds",
            "node_tx_gbps_policy": "max_outgoing_link_rate",
            "node_tx_gbps": node_tx_gbps,
            "edges": edges,
        }

    if name in ("ndv2", "NDv2", "ndv2_like", "NDv2Like"):
        node_count = int(topo.get("node_count", 16))
        if node_count != 16:
            raise ValueError("ndv2 topology requires node_count=16")
        capacity = ndv2_like_capacity_codes()
        edges = []
        node_tx_gbps = [0.0 for _ in range(node_count)]
        for left in range(node_count):
            for right in range(left + 1, node_count):
                left_to_right = capacity[left][right]
                right_to_left = capacity[right][left]
                if left_to_right == 0 and right_to_left == 0:
                    continue
                if left_to_right and right_to_left:
                    code = min(left_to_right, right_to_left)
                else:
                    code = left_to_right or right_to_left
                gbps = ndv2_like_code_to_gbps(code)
                alpha = ndv2_like_code_to_alpha(code)
                edges.append([left, right, gbps, alpha])
                edges.append([right, left, gbps, alpha])
                node_tx_gbps[left] = max(node_tx_gbps[left], gbps)
                node_tx_gbps[right] = max(node_tx_gbps[right], gbps)
        return {
            "format": "htsim-ndv2-nic-graph-v1",
            "topology_model": "nic_graph",
            "node_count": node_count,
            "endpoint_indices": list(range(node_count)),
            "switch_indices": [],
            "directed": True,
            "rate_unit": "gbps",
            "alpha_unit": "seconds",
            "node_tx_gbps_policy": "max_outgoing_link_rate",
            "node_tx_gbps": node_tx_gbps,
            "edges": edges,
        }

    if name in ("dgx_rail", "DgxRail", "dgx-rail"):
        # Solver view of csg-htsim/topo/dgx_rail_graph.h: GPUs 0-31,
        # NVSwitch 32-35 (one per host), leaf 36-43 (one per rail),
        # spine 44. Switches are TE-CCL switch nodes (buffer == 0), so the
        # emitted chunk paths collapse them and every hop is GPU->GPU,
        # matching the htsim switch-as-wire fabric. Latency alignment:
        # intra-host 9us, same-rail cross-host 50us, cross-rail 100us.
        hosts = int(topo.get("hosts", 4))
        gpus_per_host = int(topo.get("gpus_per_host", 8))
        gpu_count = hosts * gpus_per_host
        if int(topo.get("node_count", gpu_count)) != gpu_count:
            raise ValueError("dgx_rail topology requires node_count = "
                             "hosts * gpus_per_host")
        taper_gbps = float(topo.get("taper_gbps", 180.0))
        # flex_rail generalization: same wiring, tunable scale-up/NIC rates.
        # nv_gbps=2400 / nic_gbps=180 reproduces dgx_rail bit-for-bit, so
        # cached dgx_rail schedules stay valid.
        nv_gbps = float(topo.get("nv_gbps", 2400.0))
        nic_gbps = float(topo.get("nic_gbps", 180.0))
        first_nvswitch = gpu_count
        first_leaf = first_nvswitch + hosts
        spine = first_leaf + gpus_per_host
        total_nodes = spine + 1

        edges = []
        node_tx_gbps = [0.0 for _ in range(total_nodes)]
        for gpu in range(gpu_count):
            nvswitch = first_nvswitch + gpu // gpus_per_host
            leaf = first_leaf + gpu % gpus_per_host
            edges.append([gpu, nvswitch, nv_gbps, 9e-6])
            edges.append([nvswitch, gpu, nv_gbps, 0.0])
            edges.append([gpu, leaf, nic_gbps, 25e-6])
            edges.append([leaf, gpu, nic_gbps, 25e-6])
            node_tx_gbps[gpu] = max(nv_gbps, nic_gbps)
        for rail in range(gpus_per_host):
            leaf = first_leaf + rail
            edges.append([leaf, spine, taper_gbps, 25e-6])
            edges.append([spine, leaf, taper_gbps, 25e-6])
        # Switches keep node_tx_gbps == 0 so TE-CCL skips the per-node TX
        # cap for them: a leaf forwards up to 4x its fastest single link.
        return {
            "format": "htsim-dgx-rail-nic-graph-v1",
            "topology_model": "nic_graph",
            "node_count": total_nodes,
            "htsim_node_count": gpu_count,
            "endpoint_indices": list(range(gpu_count)),
            "switch_indices": list(range(first_nvswitch, total_nodes)),
            "directed": True,
            "rate_unit": "gbps",
            "alpha_unit": "seconds",
            "node_tx_gbps_policy": "max_outgoing_link_rate_endpoints_only",
            "node_tx_gbps": node_tx_gbps,
            "edges": edges,
        }

    if name in ("flex_cube", "FlexCube", "flex-cube"):
        # Solver view of csg-htsim/topo/flex_cube_graph.h: GPUs 0-31 with
        # PER-EDGE heterogeneous intra-host cube-mesh links (NDv2
        # adjacency, tier spec identical to the htsim knob), leaf 32-39
        # (one per rail), spine 40. Leaf/spine are TE-CCL switch nodes so
        # emitted chunk paths collapse them to GPU->GPU hops. Latency
        # alignment: cube link link_delay_us (default 0.7us), NIC/leaf/
        # spine segments 25us each, as in the htsim builder.
        hosts = int(topo.get("hosts", 4))
        gpus_per_host = int(topo.get("gpus_per_host", 8))
        gpu_count = hosts * gpus_per_host
        if int(topo.get("node_count", gpu_count)) != gpu_count:
            raise ValueError("flex_cube topology requires node_count = "
                             "hosts * gpus_per_host")
        nic_gbps = float(topo.get("nic_gbps", 180.0))
        taper_gbps = float(topo.get("taper_gbps", 180.0))
        link_alpha = float(topo.get("link_delay_us", 0.7)) * 1e-6
        speeds_spec = topo.get("speeds", "400x6,200x6,100x4")
        rot = int(topo.get("rot", 0))
        full_gbps = float(topo.get("full_gbps", 0.0))
        cube_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6), (4, 7),
            (5, 6), (5, 7),
            (6, 7),
        ]
        local = [[0.0] * gpus_per_host for _ in range(gpus_per_host)]
        if full_gbps > 0.0:
            for a in range(gpus_per_host):
                for b in range(gpus_per_host):
                    if a != b:
                        local[a][b] = full_gbps
        else:
            if gpus_per_host != 8:
                raise ValueError("flex_cube cube-mesh mode needs "
                                 "gpus_per_host = 8")
            tiers = []
            for item in speeds_spec.split(","):
                gbps, count = item.split("x")
                tiers.extend([float(gbps)] * int(count))
            if len(tiers) != len(cube_edges):
                raise ValueError("flex_cube tier counts must sum to 16")
            for e, (a, b) in enumerate(cube_edges):
                s = tiers[(e + rot) % len(tiers)]
                local[a][b] = s
                local[b][a] = s
        first_leaf = gpu_count
        spine = first_leaf + gpus_per_host
        total_nodes = spine + 1
        edges = []
        node_tx_gbps = [0.0 for _ in range(total_nodes)]
        for host in range(hosts):
            base = host * gpus_per_host
            for a in range(gpus_per_host):
                for b in range(gpus_per_host):
                    if local[a][b] > 0.0:
                        edges.append([base + a, base + b, local[a][b],
                                      link_alpha])
        for gpu in range(gpu_count):
            leaf = first_leaf + gpu % gpus_per_host
            edges.append([gpu, leaf, nic_gbps, 25e-6])
            edges.append([leaf, gpu, nic_gbps, 25e-6])
            # htsim paces every port independently: TX cap = all cube
            # links + the NIC concurrently.
            node_tx_gbps[gpu] = sum(local[gpu % gpus_per_host]) + nic_gbps
        for rail in range(gpus_per_host):
            leaf = first_leaf + rail
            edges.append([leaf, spine, taper_gbps, 25e-6])
            edges.append([spine, leaf, taper_gbps, 25e-6])
        return {
            "format": "htsim-flex-cube-nic-graph-v1",
            "topology_model": "nic_graph",
            "node_count": total_nodes,
            "htsim_node_count": gpu_count,
            "endpoint_indices": list(range(gpu_count)),
            "switch_indices": list(range(first_leaf, total_nodes)),
            "directed": True,
            "rate_unit": "gbps",
            "alpha_unit": "seconds",
            "node_tx_gbps_policy": "max_outgoing_link_rate_endpoints_only",
            "node_tx_gbps": node_tx_gbps,
            "edges": edges,
        }

    if name in ("ndv2_32", "Ndv232", "ndv2-32"):
        # Solver view of csg-htsim/topo/ndv2_graph.h build_ndv2_32_graph:
        # GPUs 0-31 (4 chassis x 8, NDv2 hybrid cube-mesh intra at a uniform
        # 200 Gbps / 0.7us), NICs 32-39 (2x 100G IB per chassis, GPU 0-3 ->
        # NIC0, 4-7 -> NIC1), one non-blocking IB switch 40. NIC and switch
        # are TE-CCL switch nodes (buffer == 0) so emitted chunk paths
        # collapse them and every hop is GPU->GPU. Latency alignment with
        # htsim: intra 0.7us, cross-chassis 2.6us (1.3us up + 1.3us down).
        chassis_count = 4
        gpus_per_chassis = 8
        nics_per_chassis = 2
        gpu_count = chassis_count * gpus_per_chassis
        if int(topo.get("node_count", gpu_count)) != gpu_count:
            raise ValueError("ndv2_32 topology requires node_count=32")
        first_nic = gpu_count
        ib_switch = first_nic + chassis_count * nics_per_chassis
        total_nodes = ib_switch + 1
        gpus_per_nic = gpus_per_chassis // nics_per_chassis

        cube_edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6), (4, 7),
            (5, 6), (5, 7),
            (6, 7),
        ]

        edges = []
        node_tx_gbps = [0.0 for _ in range(total_nodes)]
        for chassis in range(chassis_count):
            base = chassis * gpus_per_chassis
            for left, right in cube_edges:
                edges.append([base + left, base + right, 200.0, 0.7e-6])
                edges.append([base + right, base + left, 200.0, 0.7e-6])
        for gpu in range(gpu_count):
            local = gpu % gpus_per_chassis
            nic = (first_nic + (gpu // gpus_per_chassis) * nics_per_chassis
                   + local // gpus_per_nic)
            edges.append([gpu, nic, 100.0, 0.0])
            edges.append([nic, gpu, 100.0, 0.0])
            node_tx_gbps[gpu] = 200.0
        for nic in range(first_nic, ib_switch):
            edges.append([nic, ib_switch, 100.0, 1.3e-6])
            edges.append([ib_switch, nic, 100.0, 1.3e-6])
        # Switches keep node_tx_gbps == 0 so TE-CCL skips the per-node TX
        # cap for them (the IB switch forwards up to 8x 100G).
        return {
            "format": "htsim-ndv2-32-nic-graph-v1",
            "topology_model": "nic_graph",
            "node_count": total_nodes,
            "htsim_node_count": gpu_count,
            "endpoint_indices": list(range(gpu_count)),
            "switch_indices": list(range(first_nic, total_nodes)),
            "directed": True,
            "rate_unit": "gbps",
            "alpha_unit": "seconds",
            "node_tx_gbps_policy": "max_outgoing_link_rate_endpoints_only",
            "node_tx_gbps": node_tx_gbps,
            "edges": edges,
        }

    if name in ("syccl_a100", "SyCCLA100", "a100_clos"):
        return syccl_a100_graph()

    if name in ("syccl_a100_1host", "SyCCLA1001Host", "a100_1host"):
        return syccl_a100_1host_graph()

    if name in ("mini_a100_16g", "MiniA10016G", "mini-a100-16g"):
        return mini_a100_16g_graph()

    if name in ("mini_a100_16g_endpoint", "MiniA10016GEndpoint"):
        return mini_a100_16g_endpoint_graph(topo.get("alpha_override"))

    if name in ("clos", "Clos", "clos_16_4_1"):
        return clos_graph(
            int(topo.get("node_count", 16)),
            int(topo.get("leaf_count", topo.get("clos_leafs", 4))),
            int(topo.get("spine_count", topo.get("clos_spines", 1))),
            float(topo.get("link_rate_gbps", 100.0)),
            float(topo.get("alpha", 5e-6)),
            float(topo.get("uplink_gbps", 0.0)),
        )

    if name not in ("regular", "RegularGraph"):
        raise ValueError(f"unsupported topo name for htsim regular integration: {name}")
    node_count = int(topo.get("node_count", 16))
    degree = int(topo.get("degree", 4))
    link_rate = float(topo.get("link_rate_gbps", 100.0))
    # Default aligned with htsim main_regular -link_delay_us default (5us)
    alpha = float(topo.get("alpha", 5e-6))
    hetero_ring = bool(topo.get("regular_hetero_ring", False))
    hetero_multiplier = float(topo.get("regular_hetero_multiplier", 2.0))
    if hetero_ring and hetero_multiplier <= 0:
        raise ValueError("regular_hetero_multiplier must be positive")

    edges = []
    node_tx_gbps = [0.0 for _ in range(node_count)]
    for left, right in regular_edges(node_count, degree):
        edge_rate = link_rate
        if hetero_ring and is_regular_ring_edge(node_count, left, right):
            edge_rate = link_rate * hetero_multiplier
        edges.append([left, right, edge_rate, alpha])
        node_tx_gbps[left] = max(node_tx_gbps[left], edge_rate)
        node_tx_gbps[right] = max(node_tx_gbps[right], edge_rate)

    return {
        "format": "htsim-regular-graph-v1",
        "topology_model": "regular_graph",
        "node_count": node_count,
        "endpoint_indices": list(range(node_count)),
        "switch_indices": [],
        "directed": False,
        "degree": degree,
        "link_rate_gbps": link_rate,
        "regular_hetero_ring": hetero_ring,
        "regular_hetero_multiplier": hetero_multiplier,
        "alpha": alpha,
        "node_tx_gbps_policy": "max_outgoing_link_rate",
        "node_tx_gbps": node_tx_gbps,
        "edges": edges,
    }


def teccl_input(topo, workflow, graph_file, raw_schedule_file):
    collective = workflow.get("collective", "allgather").lower()
    if collective == "allgather":
        collective_id = 1
    elif collective == "alltoall":
        collective_id = 2
    else:
        raise ValueError("TE-CCL wrapper currently supports allgather/alltoall")

    chunk_size_bytes = int(workflow.get("chunk_size_bytes", 1048576))
    chunk_size_gb = float(workflow.get("chunk_size_gb", chunk_size_bytes / 1_000_000_000.0))
    topo_params = {
        "name": "RegularGraph",
        "chassis": 1,
        "chunk_size": chunk_size_gb,
        "alpha": [float(topo.get("alpha", 5e-6)), 0.0],
        "graph_file": str(graph_file),
        "link_rate_gbps": float(topo.get("link_rate_gbps", 100.0)),
    }
    gurobi_params = {
        "time_limit": float(workflow.get("time_limit_hours", 0.05)),
        "output_flag": int(workflow.get("gurobi_output_flag", 0)),
        "log_to_console": int(workflow.get("gurobi_log_to_console", 0)),
    }
    instance_params = {
        "collective": collective_id,
        "num_chunks": int(workflow.get("num_chunks", 1)),
        "epoch_type": int(workflow.get("epoch_type", 1)),
        "epoch_duration": float(workflow.get("epoch_duration", -1)),
        "epoch_multiplier": int(workflow.get("epoch_multiplier", 1)),
        "num_epochs": int(workflow.get("num_epochs", 16)),
        "alpha_threshold": float(workflow.get("alpha_threshold", 0.1)),
        "switch_copy": bool(workflow.get("switch_copy", False)),
        "switch_to_gpu_link_on": bool(workflow.get("switch_to_gpu_link_on", True)),
        "debug": bool(workflow.get("debug", False)),
        "debug_output_file": str(workflow.get("debug_output_file", "")),
        "objective_type": int(workflow.get("objective_type", 3)),
        "solution_method": int(workflow.get("solution_method", 2)),
        "schedule_output_file": str(raw_schedule_file),
    }
    return {
        "TopologyParams": topo_params,
        "GurobiParams": gurobi_params,
        "InstanceParams": instance_params,
    }


def normalize_workflow_for_graph(workflow, graph):
    normalized = dict(workflow)
    # Switch semantics are part of the solver contract; materialize them here
    # so they enter the cache key and meta instead of being silently
    # defaulted at input-generation time. The defaults deliberately differ
    # from TE-CCL's own (switch_copy=True, switch_to_gpu_link_on=False):
    # htsim's runtime has no switch multicast, and in switch-as-wire graphs
    # the switch->GPU segment is a real capacity-bearing port, so treating
    # it as instantaneous would let the LP oversubscribe the fabric.
    normalized.setdefault("switch_copy", False)
    normalized.setdefault("switch_to_gpu_link_on", True)
    if normalized["switch_copy"]:
        raise ValueError(
            "switch_copy=True cannot be executed by the htsim runtime "
            "(no switch multicast); keep switch_copy=False")
    collective = normalized.get("collective", "allgather").lower()
    output_size = int(normalized.get("output_buffer_size_bytes", 0))
    if output_size > 0:
        if collective != "alltoall":
            raise ValueError("output_buffer_size_bytes currently requires alltoall")
        node_count = int(graph["node_count"])
        if node_count <= 1:
            raise ValueError("output_buffer_size_bytes requires at least 2 nodes")
        bytes_per_pair = max(1, int(round(output_size / float(node_count - 1))))
        normalized["chunk_size_bytes"] = bytes_per_pair
        normalized["message_size_bytes"] = bytes_per_pair
        normalized["effective_output_buffer_size_bytes"] = bytes_per_pair * (node_count - 1)
        normalized["bytes_per_pair"] = bytes_per_pair
    return normalized


def parse_chunk_paths(data, chunk_size_bytes):
    chunk_paths = data.get("8-Chunk paths", {})
    if not chunk_paths:
        return []

    steps = []
    demand_id = 0
    fragment_id = 0
    for demand, paths in chunk_paths.items():
        match = DEMAND_RE.search(demand)
        if not match:
            continue

        demand_dst = int(match.group("dst"))
        chunk = int(match.group("chunk"))
        source = int(match.group("source"))
        parsed_paths = []
        for path_index, path_hops in enumerate(paths):
            if not path_hops:
                continue

            nodes = []
            hops = []
            hop_epochs = []
            hop_volumes = []
            for hop in path_hops:
                hop_text = hop[1] if isinstance(hop, list) and len(hop) > 1 else str(hop)
                hop_match = PATH_HOP_RE.search(hop_text)
                if not hop_match:
                    raise ValueError(f"bad TE-CCL chunk path hop: {hop_text}")
                hop_src = int(hop_match.group("src"))
                hop_dst = int(hop_match.group("dst"))
                if not nodes:
                    nodes.append(hop_src)
                elif nodes[-1] != hop_src:
                    raise ValueError(f"non-contiguous TE-CCL chunk path in demand: {demand}")
                nodes.append(hop_dst)
                hops.append({
                    "epoch": int(hop_match.group("epoch")),
                    "src": hop_src,
                    "dst": hop_dst,
                })
                hop_epochs.append(int(hop_match.group("epoch")))
                hop_volumes.append(float(hop_match.group("volume")))

            if not nodes:
                continue
            if nodes[0] != source or nodes[-1] != demand_dst:
                raise ValueError(f"TE-CCL chunk path endpoints do not match demand: {demand}")
            volume = hop_volumes[0]
            for hop_volume in hop_volumes[1:]:
                if abs(hop_volume - volume) > 1e-6:
                    raise ValueError(f"TE-CCL chunk path changes volume within one path: {demand}")
            parsed_paths.append({
                "epoch": min(hop_epochs),
                "chunk_source": source,
                "src": source,
                "dst": demand_dst,
                "chunk": chunk,
                "path": nodes,
                "hops": hops,
                "volume": volume,
                "path_index": path_index,
                "raw": demand,
            })

        if not parsed_paths:
            continue

        total_volume = sum(item["volume"] for item in parsed_paths)
        if total_volume <= 0:
            raise ValueError(f"TE-CCL chunk path demand has zero volume: {demand}")
        allocated = [
            max(1, int(round(item["volume"] / total_volume * chunk_size_bytes)))
            for item in parsed_paths
        ]
        diff = int(chunk_size_bytes) - sum(allocated)
        if diff != 0:
            largest = max(range(len(allocated)), key=lambda i: allocated[i])
            allocated[largest] += diff
        if any(size <= 0 for size in allocated):
            raise ValueError(f"TE-CCL chunk path produced non-positive bytes: {demand}")

        for item, bytes_for_path in zip(parsed_paths, allocated):
            path_fragment_id = fragment_id
            fragment_id += 1
            for hop_index, hop in enumerate(item["hops"]):
                steps.append({
                    "mode": "hop_level",
                    "epoch": hop["epoch"],
                    "demand_id": demand_id,
                    "fragment_id": path_fragment_id,
                    "hop_index": hop_index,
                    "chunk_source": source,
                    "chunk": chunk,
                    "final_dst": demand_dst,
                    "src": hop["src"],
                    "dst": hop["dst"],
                    "bytes": bytes_for_path,
                    "fragment_bytes": bytes_for_path,
                    "demand_bytes": int(chunk_size_bytes),
                    "raw": demand,
                })

        demand_id += 1

    return steps


def parse_flows(data, chunk_size_bytes):
    steps = []
    for line in data.get("7-Flows", []):
        match = FLOW_RE.search(line)
        if not match:
            continue
        source = int(match.group("source"))
        src = int(match.group("src"))
        dst = int(match.group("dst"))
        volume = float(match.group("volume") or 1.0)
        steps.append({
            "epoch": int(match.group("epoch")),
            "chunk_source": source,
            "src": src,
            "dst": dst,
            "chunk": int(match.group("chunk")),
            "bytes": max(1, int(round(volume * chunk_size_bytes))),
            "path": [src, dst],
            "raw": line,
        })
    return steps


def parse_teccl_schedule(path, chunk_size_bytes):
    data = load_json(path)
    steps = parse_chunk_paths(data, chunk_size_bytes)
    if not steps:
        steps = parse_flows(data, chunk_size_bytes)
    if not steps:
        raise ValueError(f"no parsable TE-CCL flows found in {path}")
    steps.sort(key=lambda item: (
        item["epoch"], item["chunk_source"], item["chunk"], item["src"],
        item["dst"], item.get("fragment_id", item.get("path_index", 0)),
        item.get("hop_index", 0)))
    return data, steps


def write_sched(path, node_count, chunk_size_bytes, epoch_duration, steps):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"node_count {node_count}\n")
        f.write(f"chunk_size_bytes {int(chunk_size_bytes)}\n")
        f.write(f"epoch_duration {epoch_duration}\n")
        if steps and steps[0].get("mode") == "hop_level":
            f.write("mode hop_level\n")
            for step in steps:
                f.write(
                    "hop {epoch} {demand_id} {fragment_id} {hop_index} "
                    "{chunk_source} {chunk} {final_dst} {src} {dst} "
                    "{bytes} {fragment_bytes} {demand_bytes}\n".format(**step)
                )
            return

        for step in steps:
            path_nodes = " ".join(str(node) for node in step["path"])
            f.write(
                "step {epoch} {chunk_source} {src} {dst} {chunk} {bytes} path {path_nodes}\n".format(
                    path_nodes=path_nodes, **step
                )
            )


def run_teccl(input_file, conda_env):
    if conda_env:
        conda = shutil.which("conda") or "/home/weihaoni/miniconda3/bin/conda"
        cmd = [conda, "run", "-n", conda_env, "teccl", "solve", "--input_args", str(input_file)]
    else:
        cmd = ["teccl", "solve", "--input_args", str(input_file)]
    subprocess.run(cmd, check=True)


def resolve_schedule_with_epoch_growth(topo, workflow, graph_file, raw_schedule_file,
                                       input_file, teccl_root, conda_env):
    base_epochs = int(workflow.get("num_epochs", 16))
    if base_epochs <= 0:
        raise ValueError("workflow num_epochs must be positive")
    max_epochs = int(workflow.get("max_auto_num_epochs", 1024))
    growth_factor = int(workflow.get("epoch_growth_factor", 2))
    if growth_factor < 2:
        raise ValueError("epoch_growth_factor must be at least 2")

    attempted_epochs = []
    current_epochs = base_epochs
    while True:
        attempted_epochs.append(current_epochs)
        trial_workflow = dict(workflow)
        trial_workflow["num_epochs"] = current_epochs
        input_data = teccl_input(topo, trial_workflow, graph_file, raw_schedule_file)
        write_json(input_file, input_data)
        if raw_schedule_file.exists():
            raw_schedule_file.unlink()

        old_cwd = os.getcwd()
        try:
            os.chdir(teccl_root)
            run_teccl(input_file.resolve(), conda_env)
        except subprocess.CalledProcessError:
            if raw_schedule_file.exists():
                break
            if current_epochs >= max_epochs:
                raise RuntimeError(
                    "TE-CCL did not produce a schedule after doubling num_epochs up to "
                    f"{current_epochs}. attempted_epochs={attempted_epochs}"
                )
            next_epochs = current_epochs * growth_factor
            if next_epochs <= current_epochs:
                next_epochs = current_epochs + 1
            print(f"TE-CCL schedule not found at num_epochs={current_epochs}; retrying with {next_epochs}")
            current_epochs = next_epochs
            continue
        finally:
            os.chdir(old_cwd)

        if raw_schedule_file.exists():
            break
        if current_epochs >= max_epochs:
            raise RuntimeError(
                "TE-CCL exited without a schedule after doubling num_epochs up to "
                f"{current_epochs}. attempted_epochs={attempted_epochs}"
            )
        next_epochs = current_epochs * growth_factor
        if next_epochs <= current_epochs:
            next_epochs = current_epochs + 1
        print(f"TE-CCL exited without a schedule at num_epochs={current_epochs}; retrying with {next_epochs}")
        current_epochs = next_epochs

    return input_data, current_epochs, attempted_epochs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topo", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--teccl-root", default="../TE-CCL")
    parser.add_argument("--cache-dir", default="cache/teccl_schedules")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conda-env", default="myenv")
    parser.add_argument("--force-resolve", action="store_true")
    args = parser.parse_args()

    topo = load_json(args.topo)
    workflow = load_json(args.workflow)
    graph = build_graph(topo)
    workflow = normalize_workflow_for_graph(workflow, graph)
    cache_basis = {
        "format": "htsim-teccl-cache-v1",
        "topo": topo,
        "workflow": workflow,
        "graph": graph,
    }
    cache_key = file_digest(cache_basis)
    cache_dir = pathlib.Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_schedule = cache_dir / f"{cache_key}.teccl.json"
    sched_cache = cache_dir / f"{cache_key}.sched"
    meta_cache = cache_dir / f"{cache_key}.meta.json"
    graph_file = cache_dir / f"{cache_key}.graph.json"
    input_file = cache_dir / f"{cache_key}.input.json"

    if sched_cache.exists() and raw_schedule.exists() and not args.force_resolve:
        shutil.copyfile(sched_cache, args.output)
        print(f"TE-CCL schedule cache hit: {sched_cache}")
        return 0

    write_json(graph_file, graph)
    teccl_root = pathlib.Path(args.teccl_root).resolve()
    input_data, resolved_epochs, attempted_epochs = resolve_schedule_with_epoch_growth(
        topo, workflow, graph_file, raw_schedule, input_file, teccl_root,
        args.conda_env)

    raw_data, steps = parse_teccl_schedule(raw_schedule, int(workflow.get("chunk_size_bytes", 1048576)))
    epoch_duration = raw_data.get("1-Epoch_Duration", input_data["InstanceParams"]["epoch_duration"])
    # Schedules execute on the htsim endpoint graph; switch-as-wire
    # topologies expose fewer htsim nodes than the solver graph.
    sched_node_count = graph.get("htsim_node_count", graph["node_count"])
    write_sched(sched_cache, sched_node_count, int(workflow.get("chunk_size_bytes", 1048576)), epoch_duration, steps)
    shutil.copyfile(sched_cache, args.output)
    gurobi_status = raw_data.get("Gurobi_Status")
    gurobi_status_code = raw_data.get("Gurobi_Status_Code")
    solution_count = raw_data.get("Gurobi_Solution_Count")
    mip_gap = raw_data.get("Gurobi_MIPGap")
    objective_value = raw_data.get("Gurobi_Objective_Value")
    objective_bound = raw_data.get("Gurobi_Objective_Bound")
    write_json(meta_cache, {
        "cache_key": cache_key,
        "created_at": int(time.time()),
        "topo": topo,
        "workflow": workflow,
        "topology_model": graph.get("topology_model", graph.get("format")),
        "endpoint_count": len(graph.get("endpoint_indices", [])),
        "switch_count": len(graph.get("switch_indices", [])),
        "node_tx_gbps_policy": graph.get("node_tx_gbps_policy", "none"),
        "graph_file": str(graph_file),
        "teccl_input": str(input_file),
        "teccl_raw_schedule": str(raw_schedule),
        "requested_num_epochs": int(workflow.get("num_epochs", 16)),
        "resolved_num_epochs": resolved_epochs,
        "attempted_num_epochs": attempted_epochs,
        "step_count": len(steps),
        "gurobi_status": gurobi_status,
        "gurobi_status_code": gurobi_status_code,
        "gurobi_solution_count": solution_count,
        "gurobi_mip_gap": mip_gap,
        "gurobi_objective_value": objective_value,
        "gurobi_objective_bound": objective_bound,
        "is_proven_optimal": gurobi_status == "OPTIMAL",
        "demand_count": len({
            (step.get("demand_id"), step.get("chunk_source"), step.get("chunk"), step.get("final_dst"))
            for step in steps
        }) if steps and steps[0].get("mode") == "hop_level" else len(steps),
    })
    print(f"TE-CCL schedule generated: {args.output}")
    print(f"TE-CCL schedule cache stored: {sched_cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
