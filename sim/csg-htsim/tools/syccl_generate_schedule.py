#!/usr/bin/env python3
import argparse
import json
import pathlib
import re
import sys


SRC_CHUNK_RE = re.compile(r"\((\d+),\s*(\d+)\)")

SYCCL_A100_GPU_COUNT = 32
SYCCL_A100_HOST_COUNT = 4
SYCCL_A100_GPUS_PER_HOST = 8
SYCCL_A100_NICS_PER_HOST = 4
SYCCL_A100_LEAF_COUNT = 2
MINI_A100_GPU_COUNT = 16
MINI_A100_HOST_COUNT = 4
MINI_A100_GPUS_PER_HOST = 4
MINI_A100_NICS_PER_HOST = 2
MINI_A100_LEAF_COUNT = 2


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_src_chunk(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, str):
        match = SRC_CHUNK_RE.search(value)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(f"cannot parse SyCCL src_chunk: {value!r}")


def iter_map_entries(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield parse_src_chunk(key), item
        return
    if isinstance(value, list):
        for entry in value:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"bad SyCCL map entry: {entry!r}")
            yield parse_src_chunk(entry[0]), entry[1]
        return
    raise ValueError(f"unsupported SyCCL map representation: {type(value)}")


def select_algorithm(data, algorithm_index):
    algorithms = data.get("algorithms")
    if not algorithms:
        raise ValueError("SyCCL result JSON has no algorithms")
    if algorithm_index < 0 or algorithm_index >= len(algorithms):
        raise ValueError(
            f"algorithm index {algorithm_index} outside 0..{len(algorithms) - 1}"
        )
    return algorithms[algorithm_index]


def syccl_a100_gpu_host(gpu):
    return gpu // SYCCL_A100_GPUS_PER_HOST


def syccl_a100_local_gpu(gpu):
    return gpu % SYCCL_A100_GPUS_PER_HOST


def syccl_a100_nic_id(host, local_nic):
    return SYCCL_A100_GPU_COUNT + host * SYCCL_A100_NICS_PER_HOST + local_nic


def syccl_a100_nvswitch_id(host):
    return (
        SYCCL_A100_GPU_COUNT
        + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
        + host
    )


def syccl_a100_leaf_id(leaf):
    return (
        SYCCL_A100_GPU_COUNT
        + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
        + SYCCL_A100_HOST_COUNT
        + leaf
    )


def syccl_a100_spine_id():
    return (
        SYCCL_A100_GPU_COUNT
        + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
        + SYCCL_A100_HOST_COUNT
        + SYCCL_A100_LEAF_COUNT
    )


def syccl_a100_gpu_nic(gpu):
    host = syccl_a100_gpu_host(gpu)
    local_gpu = syccl_a100_local_gpu(gpu)
    gpus_per_nic = SYCCL_A100_GPUS_PER_HOST // SYCCL_A100_NICS_PER_HOST
    return syccl_a100_nic_id(host, local_gpu // gpus_per_nic)


def syccl_a100_gpu_leaf(gpu):
    host = syccl_a100_gpu_host(gpu)
    hosts_per_leaf = SYCCL_A100_HOST_COUNT // SYCCL_A100_LEAF_COUNT
    return syccl_a100_leaf_id(host // hosts_per_leaf)


def syccl_a100_event_path(src, dst, layer_used):
    if not (0 <= src < SYCCL_A100_GPU_COUNT and 0 <= dst < SYCCL_A100_GPU_COUNT):
        raise ValueError("syccl_a100 explicit path requires GPU ids in 0..31")

    src_host = syccl_a100_gpu_host(src)
    dst_host = syccl_a100_gpu_host(dst)
    if src_host == dst_host and layer_used < 2:
        return [src, syccl_a100_nvswitch_id(src_host), dst]

    src_leaf = syccl_a100_gpu_leaf(src)
    dst_leaf = syccl_a100_gpu_leaf(dst)
    if src_leaf == dst_leaf:
        return [src, syccl_a100_gpu_nic(src), src_leaf,
                syccl_a100_gpu_nic(dst), dst]
    return [src, syccl_a100_gpu_nic(src), src_leaf, syccl_a100_spine_id(),
            dst_leaf, syccl_a100_gpu_nic(dst), dst]


def clos_16_4_1_path(src, dst):
    if not (0 <= src < 16 and 0 <= dst < 16):
        raise ValueError("clos_16_4_1 explicit path requires GPU ids in 0..15")
    src_leaf = 16 + min(src // 4, 3)
    dst_leaf = 16 + min(dst // 4, 3)
    if src_leaf == dst_leaf:
        return [src, src_leaf, dst]
    return [src, src_leaf, 20, dst_leaf, dst]


def clos_4_1_1_path(src, dst):
    if not (0 <= src < 4 and 0 <= dst < 4):
        raise ValueError("clos_4_1_1 explicit path requires GPU ids in 0..3")
    return [src, 4, dst]


def ring_4_path(src, dst):
    if not (0 <= src < 4 and 0 <= dst < 4):
        raise ValueError("ring_4 explicit path requires GPU ids in 0..3")
    clockwise = [(src + i) % 4 for i in range((dst - src) % 4 + 1)]
    counter = [(src - i) % 4 for i in range((src - dst) % 4 + 1)]
    return clockwise if len(clockwise) <= len(counter) else counter


def mini_a100_gpu_host(gpu):
    return gpu // MINI_A100_GPUS_PER_HOST


def mini_a100_local_gpu(gpu):
    return gpu % MINI_A100_GPUS_PER_HOST


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


def mini_a100_spine_id():
    return (MINI_A100_GPU_COUNT
            + MINI_A100_HOST_COUNT * MINI_A100_NICS_PER_HOST
            + MINI_A100_HOST_COUNT
            + MINI_A100_LEAF_COUNT)


def mini_a100_gpu_nic(gpu):
    host = mini_a100_gpu_host(gpu)
    local_gpu = mini_a100_local_gpu(gpu)
    gpus_per_nic = MINI_A100_GPUS_PER_HOST // MINI_A100_NICS_PER_HOST
    return mini_a100_nic_id(host, local_gpu // gpus_per_nic)


def mini_a100_gpu_leaf(gpu):
    hosts_per_leaf = MINI_A100_HOST_COUNT // MINI_A100_LEAF_COUNT
    return mini_a100_leaf_id(mini_a100_gpu_host(gpu) // hosts_per_leaf)


def mini_a100_16g_path(src, dst):
    if not (0 <= src < MINI_A100_GPU_COUNT and 0 <= dst < MINI_A100_GPU_COUNT):
        raise ValueError("mini_a100_16g explicit path requires GPU ids in 0..15")
    src_host = mini_a100_gpu_host(src)
    dst_host = mini_a100_gpu_host(dst)
    if src_host == dst_host:
        return [src, mini_a100_nvswitch_id(src_host), dst]
    src_leaf = mini_a100_gpu_leaf(src)
    dst_leaf = mini_a100_gpu_leaf(dst)
    if src_leaf == dst_leaf:
        return [src, mini_a100_gpu_nic(src), src_leaf,
                mini_a100_gpu_nic(dst), dst]
    return [src, mini_a100_gpu_nic(src), src_leaf, mini_a100_spine_id(),
            dst_leaf, mini_a100_gpu_nic(dst), dst]


DGX_RAIL_GPU_COUNT = 32
DGX_RAIL_HOSTS = 4
DGX_RAIL_GPUS_PER_HOST = 8


def dgx_rail_host(gpu):
    return gpu // DGX_RAIL_GPUS_PER_HOST


def dgx_rail_rail(gpu):
    return gpu % DGX_RAIL_GPUS_PER_HOST


def validate_dgx_rail_events(events):
    layer_counts = {}
    cross_rail_sends = 0
    same_host_nic_sends = 0
    for event in events:
        if abs(event["amount"] - 1.0) > 1e-9:
            raise ValueError(
                "dgx_rail expects reconstructed A2A sends with amount==1.0, "
                f"got {event['amount']} for event {event['id']}"
            )
        if event["reduce"]:
            raise ValueError(
                f"dgx_rail A2A schedule must not contain reduce sends "
                f"(event {event['id']})"
            )
        layer = event["layer_used"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        src = event["src"]
        dst = event["dst"]
        if dgx_rail_host(src) != dgx_rail_host(dst):
            if dgx_rail_rail(src) != dgx_rail_rail(dst):
                cross_rail_sends += 1
        elif layer >= 2:
            same_host_nic_sends += 1
    if same_host_nic_sends:
        print(
            f"WARNING: {same_host_nic_sends} same-host sends have "
            "layer_used >= 2 (solver meant NIC); htsim dgx_rail only has an "
            "NVLink logical edge for same-host pairs",
            file=sys.stderr,
        )
    print(
        "dgx_rail schedule stats: "
        f"layer_used counts {sorted(layer_counts.items())}, "
        f"cross_rail_sends {cross_rail_sends}",
        file=sys.stderr,
    )


def dgx_rail_path(src, dst):
    if not (0 <= src < DGX_RAIL_GPU_COUNT and 0 <= dst < DGX_RAIL_GPU_COUNT):
        raise ValueError("dgx_rail explicit path requires GPU ids in 0..31")
    # GPUs are logically fully connected in htsim dgx_rail; the fabric
    # (NVSwitch/NIC/leaf/spine) is wire, so every hop is a direct edge.
    return [src, dst]


NDV2_32_GPU_COUNT = 32
NDV2_32_GPUS_PER_CHASSIS = 8
# Hybrid cube-mesh adjacency of htsim ndv2_32 (topo/ndv2_graph.h), local ids.
NDV2_32_CUBE_ADJ = {
    0: (1, 2, 3, 4),
    1: (0, 2, 3, 5),
    2: (0, 1, 3, 6),
    3: (0, 1, 2, 7),
    4: (0, 5, 6, 7),
    5: (1, 4, 6, 7),
    6: (2, 4, 5, 7),
    7: (3, 4, 5, 6),
}


def ndv2_32_chassis(gpu):
    return gpu // NDV2_32_GPUS_PER_CHASSIS


def ndv2_32_path(src, dst):
    if not (0 <= src < NDV2_32_GPU_COUNT and 0 <= dst < NDV2_32_GPU_COUNT):
        raise ValueError("ndv2_32 explicit path requires GPU ids in 0..31")
    if ndv2_32_chassis(src) != ndv2_32_chassis(dst):
        # Cross-chassis: one logical IB edge (NIC -> switch -> NIC is wire).
        return [src, dst]
    base = ndv2_32_chassis(src) * NDV2_32_GPUS_PER_CHASSIS
    local_src = src - base
    local_dst = dst - base
    if local_dst in NDV2_32_CUBE_ADJ[local_src]:
        return [src, dst]
    # SyCCL was solved with an NVSwitch full-crossbar approximation of the
    # intra-chassis cube mesh (the solver's host_links=="nvlink" code path
    # crashes upstream). Expand non-adjacent intra-chassis sends into the
    # 2-hop cube route; pick a deterministic common neighbor, salted to
    # spread relay load.
    common = sorted(set(NDV2_32_CUBE_ADJ[local_src])
                    & set(NDV2_32_CUBE_ADJ[local_dst]))
    if not common:
        raise ValueError(
            f"ndv2_32 cube mesh has no common neighbor for {src}->{dst}")
    mid = common[(src * 31 + dst) % len(common)]
    return [src, base + mid, dst]


def validate_ndv2_32_events(events):
    layer_counts = {}
    expanded_intra_sends = 0
    cross_chassis_sends = 0
    for event in events:
        if abs(event["amount"] - 1.0) > 1e-9:
            raise ValueError(
                "ndv2_32 expects reconstructed A2A sends with amount==1.0, "
                f"got {event['amount']} for event {event['id']}"
            )
        if event["reduce"]:
            raise ValueError(
                f"ndv2_32 A2A schedule must not contain reduce sends "
                f"(event {event['id']})"
            )
        layer = event["layer_used"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        src = event["src"]
        dst = event["dst"]
        if ndv2_32_chassis(src) != ndv2_32_chassis(dst):
            cross_chassis_sends += 1
        elif (dst % NDV2_32_GPUS_PER_CHASSIS) not in \
                NDV2_32_CUBE_ADJ[src % NDV2_32_GPUS_PER_CHASSIS]:
            expanded_intra_sends += 1
    if expanded_intra_sends:
        print(
            f"WARNING: {expanded_intra_sends} intra-chassis sends are not "
            "cube-mesh adjacent (solver saw an NVSwitch crossbar); they were "
            "expanded into 2-hop cube routes and share NVLink bandwidth the "
            "solver did not account for",
            file=sys.stderr,
        )
    print(
        "ndv2_32 schedule stats: "
        f"layer_used counts {sorted(layer_counts.items())}, "
        f"cross_chassis_sends {cross_chassis_sends}, "
        f"expanded_intra_sends {expanded_intra_sends}",
        file=sys.stderr,
    )


CLOS_32_HOST_COUNT = 32


def validate_clos_32_events(events):
    for event in events:
        if abs(event["amount"] - 1.0) > 1e-9:
            raise ValueError(
                "clos_32 expects reconstructed A2A sends with amount==1.0, "
                f"got {event['amount']} for event {event['id']}"
            )
        if event["reduce"]:
            raise ValueError(
                f"clos_32 A2A schedule must not contain reduce sends "
                f"(event {event['id']})"
            )
        if not (0 <= event["src"] < CLOS_32_HOST_COUNT
                and 0 <= event["dst"] < CLOS_32_HOST_COUNT):
            raise ValueError("clos_32 explicit path requires host ids in 0..31")
    print(f"clos_32 schedule stats: sends {len(events)}")


def attach_explicit_paths(events, topology):
    if topology == "none":
        return
    if topology == "clos_32":
        validate_clos_32_events(events)
        for event in events:
            # Pure clos is a logical full mesh over hosts (switch-as-wire).
            event["path"] = [event["src"], event["dst"]]
        return
    if topology == "dgx_rail":
        validate_dgx_rail_events(events)
        for event in events:
            event["path"] = dgx_rail_path(event["src"], event["dst"])
        return
    if topology == "ndv2_32":
        validate_ndv2_32_events(events)
        for event in events:
            event["path"] = ndv2_32_path(event["src"], event["dst"])
        return
    if topology == "syccl_a100_1host":
        for event in events:
            event["path"] = [event["src"], 8, event["dst"]]
        return
    if topology == "clos_16_4_1":
        for event in events:
            event["path"] = clos_16_4_1_path(event["src"], event["dst"])
        return
    if topology == "clos_4_1_1":
        for event in events:
            event["path"] = clos_4_1_1_path(event["src"], event["dst"])
        return
    if topology == "ring_4":
        for event in events:
            event["path"] = ring_4_path(event["src"], event["dst"])
        return
    if topology == "mini_a100_16g":
        for event in events:
            event["path"] = mini_a100_16g_path(event["src"], event["dst"])
        return
    if topology != "syccl_a100":
        raise ValueError(f"unsupported explicit path topology: {topology}")
    for event in events:
        event["path"] = syccl_a100_event_path(
            event["src"], event["dst"], event["layer_used"]
        )


def schedule_from_syccl_result(data, algorithm_index, explicit_path_topology):
    ngpus = int(data["ngpus"])
    chunk_size_byte = int(data["chunk_size_byte"])
    coll_name = data.get("coll_name", "")
    if coll_name.lower() != "alltoall":
        raise ValueError(f"expected SyCCL alltoall result, got {coll_name!r}")

    algorithm = select_algorithm(data, algorithm_index)
    final = algorithm["final_schedule"]
    schedule = final["Schedule"]

    chunk_sizes = {
        src_chunk: int(size)
        for src_chunk, size in iter_map_entries(schedule["chunk_sizes"])
    }

    chunks = {}
    for src in range(ngpus):
        for dst in range(ngpus):
            if src == dst:
                continue
            chunks.setdefault((src, dst), chunk_size_byte)

    for chunk_key, size in chunk_sizes.items():
        chunks[chunk_key] = size

    events = []
    next_event_id = 0
    for event_group in schedule["Events"]:
        chunk_source, chunk = parse_src_chunk(event_group["src_chunk"])
        bytes_for_chunk = chunk_sizes.get((chunk_source, chunk), chunk_size_byte)
        chunks.setdefault((chunk_source, chunk), bytes_for_chunk)
        for send in event_group.get("sends", []):
            src = int(send["src_gpu"])
            dst = int(send["dst_gpu"])
            if src == dst:
                continue
            amount = float(send.get("amount", 1.0))
            events.append(
                {
                    "id": next_event_id,
                    "epoch": int(send["epoch"]),
                    "chunk_source": chunk_source,
                    "chunk": chunk,
                    "src": src,
                    "dst": dst,
                    "bytes": max(1, int(round(bytes_for_chunk * amount))),
                    "amount": amount,
                    "copy": 1 if bool(send.get("copy", False)) else 0,
                    "reduce": 1 if bool(send.get("reduce", False)) else 0,
                    "layer_used": int(send.get("layer_used", -1)),
                }
            )
            next_event_id += 1

    if not events:
        raise ValueError("SyCCL result contained no send steps")

    # One app-level demand per chunk. SyCCL may SPLIT a (src,dst) pair into
    # several chunks (addr = dst + k*ngpus, seen on weak-NV solves); the
    # runtime credits deliveries by (chunk_source, chunk, dst), so each
    # split chunk needs its own demand row carrying its own byte count.
    demands = []
    for (chunk_source, chunk_id), size in sorted(chunks.items()):
        dst = chunk_id % ngpus
        if dst == chunk_source:
            raise ValueError(
                "SyCCL chunk (%d,%d) maps to its own source; unexpected "
                "chunk numbering" % (chunk_source, chunk_id))
        demands.append(
            {
                "src": chunk_source,
                "dst": dst,
                "chunk": chunk_id,
                "bytes": size,
            }
        )

    events.sort(
        key=lambda s: (
            s["epoch"],
            s["chunk_source"],
            s["chunk"],
            s["src"],
            s["dst"],
        )
    )
    for new_id, event in enumerate(events):
        event["id"] = new_id
    attach_explicit_paths(events, explicit_path_topology)
    solver_predicted_us = float(
        final.get("Time", data.get("sketch_solve_best_time", -1.0))
    )
    return ngpus, chunk_size_byte, solver_predicted_us, chunks, demands, events


def write_htsim_schedule(path, node_count, chunk_size_bytes,
                         solver_predicted_us, chunks, demands, events,
                         runtime_mode):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# htsim SyCCL GPU-to-GPU event schedule\n")
        f.write(f"node_count {node_count}\n")
        f.write(f"chunk_size_bytes {chunk_size_bytes}\n")
        f.write("epoch_duration 1\n")
        f.write(f"mode {runtime_mode}\n")
        f.write(f"solver_predicted_us {solver_predicted_us}\n")
        for (chunk_source, chunk), bytes_for_chunk in sorted(chunks.items()):
            f.write(
                f"chunk {chunk_source} {chunk} {bytes_for_chunk} init {chunk_source}\n"
            )
        for demand in sorted(demands, key=lambda d: (d["src"], d["dst"], d["chunk"])):
            f.write("demand {src} {dst} {chunk} {bytes}\n".format(**demand))
        for event in events:
            path_suffix = ""
            if event.get("path"):
                path_suffix = " path " + " ".join(str(node) for node in event["path"])
            f.write(
                (
                    "event {id} {epoch} {chunk_source} {chunk} {src} {dst} "
                    "{bytes} {amount:.12g} {copy} {reduce} {layer_used}"
                    "{path_suffix}\n"
                ).format(
                    path_suffix=path_suffix, **event
                )
            )


def main():
    parser = argparse.ArgumentParser(
        description="Convert a SyCCL result JSON into htsim syccl_schedule format."
    )
    parser.add_argument("--syccl-result", required=True)
    parser.add_argument("--algorithm-index", type=int, default=0)
    parser.add_argument("--runtime-mode", default="syccl_event_dag",
                        choices=["syccl_event", "syccl_event_dag"])
    parser.add_argument("--explicit-path-topology", default="syccl_a100",
                        choices=["none", "syccl_a100", "syccl_a100_1host",
                                 "clos_16_4_1", "clos_4_1_1", "ring_4",
                                 "mini_a100_16g", "dgx_rail", "ndv2_32",
                                 "clos_32"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_json(args.syccl_result)
    (node_count, chunk_size_bytes, solver_predicted_us,
     chunks, demands, events) = schedule_from_syccl_result(
        data, args.algorithm_index, args.explicit_path_topology
    )
    write_htsim_schedule(args.output, node_count, chunk_size_bytes,
                         solver_predicted_us, chunks, demands, events,
                         args.runtime_mode)
    print(
        f"wrote {len(events)} SyCCL events and {len(demands)} demands "
        f"for {node_count} GPUs to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
