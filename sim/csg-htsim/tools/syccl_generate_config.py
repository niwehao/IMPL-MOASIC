#!/usr/bin/env python3
import argparse
import json
import pathlib
import sys


def mb_per_us_from_gbps(gbps):
    return gbps / 8000.0


def build_config(args):
    gpu_count = args.gpus if args.gpus is not None else args.hosts
    if args.hosts is not None:
        print(
            "warning: --hosts is deprecated for SyCCL config generation; "
            "it is treated as total GPU count. Use --gpus instead.",
            file=sys.stderr,
        )
    if gpu_count <= 0:
        raise ValueError("gpus must be positive")
    if args.gpus_per_host <= 0:
        raise ValueError("gpus_per_host must be positive")
    if gpu_count % args.gpus_per_host != 0:
        raise ValueError("gpus must be divisible by gpus_per_host")
    host_count = gpu_count // args.gpus_per_host

    if args.spines != 1:
        raise ValueError(
            "SyCCL upstream pod topology does not model full leaf-spine Clos. "
            "Use --spines 1 for an exact match with the currently generated "
            "SyCCL pod config, or extend SyCCL topology construction first."
        )
    if host_count % args.leafs != 0:
        raise ValueError("SyCCL host_count must be divisible by leafs for pod config")

    link_mbpus = mb_per_us_from_gbps(args.link_rate_gbps)
    nvlink = [
        [peer for peer in range(args.gpus_per_host) if peer != gpu]
        for gpu in range(args.gpus_per_host)
    ]
    return {
        "coll": {
            "name": "alltoall",
            "byte": args.message_size_bytes,
            "root_sender": -1,
            "root_receiver": -1,
        },
        "hosts": {
            "host_num": host_count,
            "host_gpu_num": args.gpus_per_host,
            "host_nic_num": 1,
            "host_links": "nvlink",
        },
        "topo": [
            {"layer_id": 0, "type": "local", "link_spec": "memcpy"},
            {"layer_id": 1, "type": "host", "link_spec": "nvlink"},
            {"layer_id": 2, "type": "nic", "link_spec": "link_nic"},
            {
                "layer_id": 3,
                "type": "switch",
                "switch_topo": "pod",
                "switch_num": args.leafs,
                "link_spec": "netlink_leaf",
            },
            {
                "layer_id": 4,
                "type": "switch",
                "switch_topo": "pod",
                "switch_num": args.spines,
                "link_spec": "netlink_spine",
            },
        ],
        "host_links": {"nvlink": nvlink},
        "link_spec": {
            "memcpy": {"bw_mbpus": link_mbpus, "lat_us": 0.01},
            "nvlink": {"bw_mbpus": link_mbpus, "lat_us": 0.01},
            "link_nic": {"bw_mbpus": link_mbpus, "lat_us": args.latency_us},
            "netlink_leaf": {"bw_mbpus": link_mbpus, "lat_us": args.latency_us},
            "netlink_spine": {"bw_mbpus": link_mbpus, "lat_us": args.latency_us},
        },
        "solver": {
            "split_chunks": 1,
            "chunk_size_B": 0,
            "sw_hyperedge": False,
            "nic_hyperedge": True,
            "alpha_threshold": 0.1,
            "trial_time_factor": 1.0,
            "epoch_time_strat": "BETA_MULT",
            "epoch_model": "ACCURATE",
            "model_extra_sends_big_beta": False,
            "model_extra_sends_small_beta": False,
            "balance_ratio": 1.0,
            "alpha_epoch_duration_ratio_max": 100,
            "mip_config": {
                "feasibility_tol": 0.0001,
                "intfeas_tol": 0.0001,
                "optimality_tol": 0.0001,
                "mip_gap": 0.001,
                "mip_focus": 0,
                "heuristics": 0.25,
                "time_limit_h": args.time_limit_hours,
            },
        },
        "sketch": {
            "customize_sketch": False,
            "use_sketch_input": False,
            "save_sketch": False,
            "sketch_path": "../config/sketch.json",
        },
        "prune": {
            "max_comb_layer_num": 3,
            "allow_unequal_source_per_group": False,
            "ignored_layers": [0, 2],
            "must_contain_layers": [1, 3],
            "layer_comm_ordering": {},
            "layer_comm_uplimit": {"3": 16},
            "all_groups_used": False,
            "limit_num_steps": 2,
            "allow_source_group_special": True,
        },
        "algo_solve": {
            "filter_balance_ratio": 4.0,
            "solve_balance_ratio": 1.0,
            "raw_solve_balance_ratio": 1,
            "do_raw_solve": False,
            "num_used_algos_min": 1,
            "num_used_algos_max": 4,
            "time_diff_threshold": 0.2,
            "solve_output": args.solve_output,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a small SyCCL alltoall config matching an htsim Clos run."
    )
    parser.add_argument("--gpus", type=int, default=None)
    parser.add_argument("--hosts", type=int, default=None)
    parser.add_argument("--gpus-per-host", type=int, default=2)
    parser.add_argument("--leafs", type=int, default=4)
    parser.add_argument("--spines", type=int, default=1)
    parser.add_argument("--message-size-bytes", type=int, default=1048576)
    parser.add_argument("--link-rate-gbps", type=float, default=100.0)
    parser.add_argument("--latency-us", type=float, default=5.0)
    parser.add_argument("--time-limit-hours", type=float, default=0.05)
    parser.add_argument("--solve-output", default="./result-htsim-clos-a2a.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        config = build_config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
