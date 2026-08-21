#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys


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
MINI_A100_GPU_COUNT = 16
MINI_A100_HOST_COUNT = 4
MINI_A100_GPUS_PER_HOST = 4
MINI_A100_NICS_PER_HOST = 2
MINI_A100_LEAF_COUNT = 2


def ndv2_edges():
    node_count = 16
    matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
    for i, row in enumerate(NDV2_LIKE_RAW_CODES):
        for j, value in enumerate(row):
            matrix[j][i] = value

    edges = set()
    for left in range(node_count):
        for right in range(left + 1, node_count):
            if matrix[left][right] or matrix[right][left]:
                edges.add((left, right))
                edges.add((right, left))
    return edges


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


def syccl_a100_spine_id():
    return (SYCCL_A100_GPU_COUNT
            + SYCCL_A100_HOST_COUNT * SYCCL_A100_NICS_PER_HOST
            + SYCCL_A100_HOST_COUNT
            + SYCCL_A100_LEAF_COUNT)


def syccl_a100_gpu_host(gpu):
    return gpu // SYCCL_A100_GPUS_PER_HOST


def syccl_a100_local_gpu(gpu):
    return gpu % SYCCL_A100_GPUS_PER_HOST


def syccl_a100_gpu_nic(gpu):
    gpus_per_nic = SYCCL_A100_GPUS_PER_HOST // SYCCL_A100_NICS_PER_HOST
    return syccl_a100_nic_id(
        syccl_a100_gpu_host(gpu),
        syccl_a100_local_gpu(gpu) // gpus_per_nic,
    )


def syccl_a100_gpu_leaf(gpu):
    hosts_per_leaf = SYCCL_A100_HOST_COUNT // SYCCL_A100_LEAF_COUNT
    return syccl_a100_leaf_id(syccl_a100_gpu_host(gpu) // hosts_per_leaf)


def syccl_a100_event_path(src, dst):
    src_host = syccl_a100_gpu_host(src)
    dst_host = syccl_a100_gpu_host(dst)
    if src_host == dst_host:
        return [src, syccl_a100_nvswitch_id(src_host), dst]

    src_leaf = syccl_a100_gpu_leaf(src)
    dst_leaf = syccl_a100_gpu_leaf(dst)
    if src_leaf == dst_leaf:
        return [src, syccl_a100_gpu_nic(src), src_leaf,
                syccl_a100_gpu_nic(dst), dst]
    return [src, syccl_a100_gpu_nic(src), src_leaf, syccl_a100_spine_id(),
            dst_leaf, syccl_a100_gpu_nic(dst), dst]


def syccl_a100_edges():
    edges = set()
    for host in range(SYCCL_A100_HOST_COUNT):
        nvswitch = syccl_a100_nvswitch_id(host)
        for local_gpu in range(SYCCL_A100_GPUS_PER_HOST):
            gpu = host * SYCCL_A100_GPUS_PER_HOST + local_gpu
            edges.add((gpu, nvswitch))
            edges.add((nvswitch, gpu))
            edges.add((gpu, syccl_a100_gpu_nic(gpu)))
            edges.add((syccl_a100_gpu_nic(gpu), gpu))
    for gpu in range(SYCCL_A100_GPU_COUNT):
        leaf = syccl_a100_gpu_leaf(gpu)
        nic = syccl_a100_gpu_nic(gpu)
        edges.add((nic, leaf))
        edges.add((leaf, nic))
    spine = syccl_a100_spine_id()
    for leaf in range(SYCCL_A100_LEAF_COUNT):
        leaf_node = syccl_a100_leaf_id(leaf)
        edges.add((leaf_node, spine))
        edges.add((spine, leaf_node))
    return edges


def clos_16_4_1_path(src, dst):
    if not (0 <= src < 16 and 0 <= dst < 16):
        raise ValueError("clos_16_4_1 path requires rank ids in 0..15")
    if src == dst:
        return [src]
    src_leaf = 16 + min(src // 4, 3)
    dst_leaf = 16 + min(dst // 4, 3)
    if src_leaf == dst_leaf:
        return [src, src_leaf, dst]
    return [src, src_leaf, 20, dst_leaf, dst]


def clos_16_4_1_edges():
    edges = set()
    for host in range(16):
        leaf = 16 + min(host // 4, 3)
        edges.add((host, leaf))
        edges.add((leaf, host))
    for leaf in range(16, 20):
        edges.add((leaf, 20))
        edges.add((20, leaf))
    return edges


def clos_4_1_1_path(src, dst):
    if not (0 <= src < 4 and 0 <= dst < 4):
        raise ValueError("clos_4_1_1 path requires rank ids in 0..3")
    return [src, 4, dst]


def clos_4_1_1_edges():
    return {(host, 4) for host in range(4)} | {(4, host) for host in range(4)}


def ring_4_path(src, dst):
    if not (0 <= src < 4 and 0 <= dst < 4):
        raise ValueError("ring_4 path requires rank ids in 0..3")
    if src == dst:
        return [src]
    clockwise = [(src + i) % 4 for i in range((dst - src) % 4 + 1)]
    counter = [(src - i) % 4 for i in range((src - dst) % 4 + 1)]
    return clockwise if len(clockwise) <= len(counter) else counter


def ring_4_edges():
    return {(i, (i + 1) % 4) for i in range(4)} | {((i + 1) % 4, i) for i in range(4)}


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
        raise ValueError("mini_a100_16g path requires rank ids in 0..15")
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


def mini_a100_16g_edges():
    edges = set()
    for host in range(MINI_A100_HOST_COUNT):
        nvswitch = mini_a100_nvswitch_id(host)
        for local_gpu in range(MINI_A100_GPUS_PER_HOST):
            gpu = host * MINI_A100_GPUS_PER_HOST + local_gpu
            edges.add((gpu, nvswitch))
            edges.add((nvswitch, gpu))
            nic = mini_a100_gpu_nic(gpu)
            edges.add((gpu, nic))
            edges.add((nic, gpu))
        for local_nic in range(MINI_A100_NICS_PER_HOST):
            nic = mini_a100_nic_id(host, local_nic)
            leaf = mini_a100_gpu_leaf(host * MINI_A100_GPUS_PER_HOST)
            edges.add((nic, leaf))
            edges.add((leaf, nic))
    spine = mini_a100_spine_id()
    for leaf in range(MINI_A100_LEAF_COUNT):
        leaf_node = mini_a100_leaf_id(leaf)
        edges.add((leaf_node, spine))
        edges.add((spine, leaf_node))
    return edges


DGX_RAIL_GPU_COUNT = 32
DGX_RAIL_GPUS_PER_HOST = 8


def dgx_rail_host(gpu):
    return gpu // DGX_RAIL_GPUS_PER_HOST


def dgx_rail_rail(gpu):
    return gpu % DGX_RAIL_GPUS_PER_HOST


def dgx_rail_edges():
    # htsim dgx_rail is a logical full mesh over GPUs; the fabric
    # (NVSwitch/NIC/leaf/spine) is switch-as-wire inside each route.
    return {(a, b) for a in range(DGX_RAIL_GPU_COUNT)
            for b in range(DGX_RAIL_GPU_COUNT) if a != b}


NDV2_32_GPU_COUNT = 32
NDV2_32_CHASSIS_COUNT = 4
NDV2_32_GPUS_PER_CHASSIS = 8
NDV2_32_CUBE_ADJ = {
    0: [1, 2, 3, 4], 1: [0, 2, 3, 5], 2: [0, 1, 3, 6], 3: [0, 1, 2, 7],
    4: [0, 5, 6, 7], 5: [1, 4, 6, 7], 6: [2, 4, 5, 7], 7: [3, 4, 5, 6],
}


def ndv2_32_chassis(gpu):
    return gpu // NDV2_32_GPUS_PER_CHASSIS


def ndv2_32_edges():
    # htsim ndv2_32: intra-chassis only the hybrid cube-mesh pairs are
    # physical edges; any cross-chassis pair is a logical edge through the
    # switch-as-wire IB fabric.
    edges = set()
    for chassis in range(NDV2_32_CHASSIS_COUNT):
        base = chassis * NDV2_32_GPUS_PER_CHASSIS
        for local, neighbors in NDV2_32_CUBE_ADJ.items():
            for neighbor in neighbors:
                edges.add((base + local, base + neighbor))
    for a in range(NDV2_32_GPU_COUNT):
        for b in range(NDV2_32_GPU_COUNT):
            if a != b and ndv2_32_chassis(a) != ndv2_32_chassis(b):
                edges.add((a, b))
    return edges


CLOS_32_HOST_COUNT = 32


def clos_32_edges():
    # htsim clos (switch-as-wire) is a logical full mesh over hosts; the
    # leaf/spine fabric lives inside each route.
    return {(a, b) for a in range(CLOS_32_HOST_COUNT)
            for b in range(CLOS_32_HOST_COUNT) if a != b}


REGULAR_32_NODE_COUNT = 32
REGULAR_32_OFFSETS = (1, 6)


def regular_32_edges():
    # 32-node degree-4 circulant graph (offsets 1 and 6), matching both
    # topo/regular_graph.h choose_low_diameter_offsets and the teccl
    # adapter's regular_edges() for nodes=32 degree=4.
    edges = set()
    for a in range(REGULAR_32_NODE_COUNT):
        for off in REGULAR_32_OFFSETS:
            b = (a + off) % REGULAR_32_NODE_COUNT
            edges.add((a, b))
            edges.add((b, a))
    return edges


def graph_edges(topology):
    if topology == "regular_32":
        return regular_32_edges()
    if topology == "clos_32":
        return clos_32_edges()
    if topology == "ndv2_32":
        return ndv2_32_edges()
    if topology == "dgx_rail":
        return dgx_rail_edges()
    if topology == "ndv2":
        return ndv2_edges()
    if topology == "syccl_a100":
        return syccl_a100_edges()
    if topology == "syccl_a100_1host":
        return {(gpu, 8) for gpu in range(8)} | {(8, gpu) for gpu in range(8)}
    if topology == "clos_16_4_1":
        return clos_16_4_1_edges()
    if topology == "clos_4_1_1":
        return clos_4_1_1_edges()
    if topology == "ring_4":
        return ring_4_edges()
    if topology == "mini_a100_16g":
        return mini_a100_16g_edges()
    raise ValueError("taccl adapter currently supports --topology ndv2, ndv2_32, clos_32, regular_32, syccl_a100, syccl_a100_1host, clos_16_4_1, clos_4_1_1, ring_4, mini_a100_16g, or dgx_rail")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def chunk_metadata(algo, chunk_size_bytes):
    chunks = {}
    for chunk in algo["collective"]["chunks"]:
        addr = int(chunk["addr"])
        pre = chunk.get("pre", [])
        post = chunk.get("post", [])
        if len(pre) != 1 or len(post) != 1:
            raise ValueError(f"expected one pre and one post rank for chunk addr={addr}")
        source = int(pre[0])
        final_dst = int(post[0])
        chunks[addr] = {
            "addr": addr,
            "source": source,
            "final_dst": final_dst,
            "bytes": int(chunk_size_bytes),
        }
    return chunks


def validate_send_edge(src, dst, allowed_edges, allow_nonedge_routing):
    if src == dst:
        return
    if (src, dst) in allowed_edges:
        return
    if allow_nonedge_routing:
        return
    raise ValueError(
        f"TACCL send {src}->{dst} is not a physical htsim edge. "
        "Regenerate TACCL with a matching topology/sketch, or pass "
        "--allow-nonedge-routing only for a non-strict smoke test."
    )


def convert(input_file, output_file, chunk_size_bytes, topology,
            allow_nonedge_routing):
    algo = load_json(input_file)
    collective = algo["collective"]
    node_count = int(collective["nodes"])
    if collective.get("name", "").lower().find("alltoall") < 0:
        raise ValueError("TACCL htsim adapter currently supports Alltoall only")
    if topology == "ndv2" and node_count != 16:
        raise ValueError("htsim ndv2 TACCL conversion requires 16 ranks")
    if topology == "syccl_a100" and node_count != SYCCL_A100_GPU_COUNT:
        raise ValueError("htsim syccl_a100 TACCL conversion requires 32 ranks")
    if topology == "syccl_a100_1host" and node_count != 8:
        raise ValueError("htsim syccl_a100_1host TACCL conversion requires 8 ranks")
    if topology == "clos_16_4_1" and node_count != 16:
        raise ValueError("htsim clos_16_4_1 TACCL conversion requires 16 ranks")
    if topology == "clos_4_1_1" and node_count != 4:
        raise ValueError("htsim clos_4_1_1 TACCL conversion requires 4 ranks")
    if topology == "ring_4" and node_count != 4:
        raise ValueError("htsim ring_4 TACCL conversion requires 4 ranks")
    if topology == "mini_a100_16g" and node_count != MINI_A100_GPU_COUNT:
        raise ValueError("htsim mini_a100_16g TACCL conversion requires 16 ranks")
    if topology == "dgx_rail" and node_count != DGX_RAIL_GPU_COUNT:
        raise ValueError("htsim dgx_rail TACCL conversion requires 32 ranks")
    if topology == "ndv2_32" and node_count != NDV2_32_GPU_COUNT:
        raise ValueError("htsim ndv2_32 TACCL conversion requires 32 ranks")
    if topology == "clos_32" and node_count != CLOS_32_HOST_COUNT:
        raise ValueError("htsim clos_32 TACCL conversion requires 32 ranks")
    if topology == "regular_32" and node_count != REGULAR_32_NODE_COUNT:
        raise ValueError("htsim regular_32 TACCL conversion requires 32 ranks")

    allowed_edges = graph_edges(topology)
    chunks = chunk_metadata(algo, chunk_size_bytes)
    steps = []
    step_id = 0
    cross_rail_sends = 0
    cross_chassis_sends = 0
    # epoch = cumulative step.rounds rather than the step list index:
    # TACCL's rounds encode each step's bandwidth-normalized duration, so
    # cumulative rounds preserve the solver's time structure in the .sched.
    # The htsim runtime only consumes epoch ordering (send priority), which
    # is unchanged because cumulative sums are monotonic in step index.
    cumulative_rounds = 0
    for step in algo["steps"]:
        epoch = cumulative_rounds
        cumulative_rounds += max(1, int(step.get("rounds", 1)))
        for send in step["sends"]:
            if len(send) < 3:
                raise ValueError(f"bad TACCL send entry: {send}")
            addr, src, dst = int(send[0]), int(send[1]), int(send[2])
            if addr not in chunks:
                raise ValueError(f"TACCL send references unknown addr={addr}")
            if src == dst:
                continue
            if topology == "syccl_a100":
                path = syccl_a100_event_path(src, dst)
            elif topology == "syccl_a100_1host":
                path = [src, 8, dst]
            elif topology == "clos_16_4_1":
                path = clos_16_4_1_path(src, dst)
            elif topology == "clos_4_1_1":
                path = clos_4_1_1_path(src, dst)
            elif topology == "ring_4":
                path = ring_4_path(src, dst)
            elif topology == "mini_a100_16g":
                path = mini_a100_16g_path(src, dst)
            elif topology == "dgx_rail":
                # Logical GPU->GPU hop maps 1:1 onto an htsim dgx_rail
                # route; the rail sketch should keep cross-host sends on
                # the same rail (cross-rail would traverse the tapered
                # spine).
                if (dgx_rail_host(src) != dgx_rail_host(dst)
                        and dgx_rail_rail(src) != dgx_rail_rail(dst)):
                    cross_rail_sends += 1
                path = [src, dst]
            elif topology == "clos_32":
                # Pure clos: every host pair is one logical edge through the
                # switch-as-wire fabric.
                path = [src, dst]
            elif topology == "ndv2_32":
                # Intra-chassis sends must ride an existing cube-mesh
                # NVLink; cross-chassis pairs are logical IB edges.
                validate_send_edge(src, dst, allowed_edges,
                                   allow_nonedge_routing)
                if ndv2_32_chassis(src) != ndv2_32_chassis(dst):
                    cross_chassis_sends += 1
                path = [src, dst]
            else:
                validate_send_edge(src, dst, allowed_edges,
                                   allow_nonedge_routing)
                path = [src, dst]
            steps.append({
                "step_id": step_id,
                "epoch": epoch,
                "addr": addr,
                "src": src,
                "dst": dst,
                "bytes": int(chunk_size_bytes),
                "path": path,
            })
            step_id += 1

    if not steps:
        raise ValueError("TACCL algorithm has no sends")

    demands = [
        item for item in chunks.values()
        if item["source"] != item["final_dst"]
    ]
    if not demands:
        raise ValueError("TACCL Alltoall conversion found no non-self demands")

    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"node_count {node_count}\n")
        f.write(f"chunk_size_bytes {int(chunk_size_bytes)}\n")
        f.write("collective alltoall\n")
        f.write("epoch_duration 1\n")
        f.write("mode taccl_step\n")
        for item in sorted(chunks.values(), key=lambda x: x["addr"]):
            f.write(
                "chunk {addr} {source} {final_dst} {bytes}\n".format(**item)
            )
        for item in sorted(demands, key=lambda x: x["addr"]):
            f.write(
                "demand {addr} {source} {final_dst} {bytes}\n".format(**item)
            )
        for step in steps:
            path_nodes = " ".join(str(node) for node in step["path"])
            f.write(
                "step {step_id} {epoch} {addr} {src} {dst} {bytes} path {path_nodes}\n"
                .format(path_nodes=path_nodes, **step)
            )

    stats = {
        "node_count": node_count,
        "chunks": len(chunks),
        "app_demands": len(demands),
        "transport_steps": len(steps),
        "app_bytes": sum(item["bytes"] for item in demands),
    }
    if topology == "dgx_rail":
        stats["cross_rail_sends"] = cross_rail_sends
    if topology == "ndv2_32":
        stats["cross_chassis_sends"] = cross_chassis_sends
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size-bytes", type=int, required=True)
    parser.add_argument("--topology", default="ndv2")
    parser.add_argument("--allow-nonedge-routing", action="store_true")
    args = parser.parse_args()
    try:
        stats = convert(args.input, args.output, args.chunk_size_bytes,
                        args.topology, args.allow_nonedge_routing)
    except Exception as exc:
        print(f"TACCL htsim conversion failed: {exc}", file=sys.stderr)
        return 1
    print(f"TACCL htsim schedule written: {args.output}")
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
