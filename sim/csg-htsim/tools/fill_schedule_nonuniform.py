#!/usr/bin/env python3
"""Fill a UNIFORM htsim schedule with a NON-UNIFORM demand matrix.

TE-CCL/TACCL/SyCCL only solve uniform alltoall, so for non-uniform A2A the
plan is solved at the power-of-two padded per-pair size (>= max real pair)
and then acts as a filler: each (src,dst) pair's planned capacity is truncated
to the real demand -- sources only send real bytes, destinations complete
once the real bytes arrive, and planned slack is simply never transmitted.

Mechanically: group the schedule's transfers into per-pair chunk/fragment
chains, assign the pair's real bytes to chains in deterministic order
(min(chain_capacity, remaining)), rewrite every hop/step/event of a chain to
its assigned size, and drop chains (or whole pairs) that get zero bytes.
Supports all four on-disk formats: TE-CCL hop_level, TE-CCL path steps,
TACCL taccl_step, SyCCL syccl_event.
"""

import argparse
from collections import OrderedDict


def load_demand(path):
    nodes = 0
    demand = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "node_count":
                nodes = int(parts[1])
            elif parts[0] == "demand":
                s, d, b = int(parts[1]), int(parts[2]), int(parts[3])
                demand[(s, d)] = demand.get((s, d), 0) + b
    if nodes == 0:
        raise SystemExit("demand file missing node_count")
    return nodes, demand


def parse_schedule(path):
    header = []   # (kind, tokens) lines before/besides transfers, kept in order
    lines = []    # every line as token list (or raw comment string)
    mode = "teccl_path"
    with open(path) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("#"):
                lines.append(raw)
                continue
            toks = raw.split()
            lines.append(toks)
            if toks[0] == "mode":
                if toks[1] == "hop_level":
                    mode = "teccl_hop"
                elif toks[1] == "taccl_step":
                    mode = "taccl"
                elif toks[1].startswith("syccl_event"):
                    mode = "syccl"
    return mode, lines


def assign_chains(chains, pair_demand, label):
    """chains: OrderedDict chain_key -> {'pair':(s,d),'capacity':int}.
    Returns chain_key -> assigned bytes. Chains are filled in insertion
    order per pair."""
    remaining = dict(pair_demand)
    assigned = {}
    for key, info in chains.items():
        pair = info["pair"]
        want = remaining.get(pair, 0)
        take = min(info["capacity"], want)
        assigned[key] = take
        remaining[pair] = want - take
    leftover = {p: r for p, r in remaining.items() if r > 0}
    if leftover:
        sample = list(leftover.items())[:5]
        raise SystemExit("%s: plan capacity smaller than demand for pairs %s"
                         % (label, sample))
    return assigned


def fill_teccl_hop(lines, demand):
    # hop epoch demand_id fragment_id hop_index chunk_source chunk final_dst
    #     src dst bytes fragment_bytes demand_bytes
    frag_info = OrderedDict()   # frag_id -> {'pair','capacity'}
    demand_pair = {}
    for toks in lines:
        if isinstance(toks, str) or toks[0] != "hop":
            continue
        frag_id = int(toks[3])
        pair = (int(toks[5]), int(toks[7]))
        demand_pair[int(toks[2])] = pair
        if frag_id not in frag_info:
            frag_info[frag_id] = {"pair": pair, "capacity": int(toks[11])}
    assigned = assign_chains(frag_info, demand, "teccl_hop")

    # Dropping zero-byte fragments leaves id gaps; the runtime requires
    # contiguous demand/fragment ids, so renumber in first-appearance order.
    new_demand_id = {}
    new_frag_id = {}
    out = []
    for toks in lines:
        if isinstance(toks, str):
            out.append(toks)
            continue
        if toks[0] != "hop":
            out.append(" ".join(toks))
            continue
        frag_id = int(toks[3])
        take = assigned[frag_id]
        if take == 0:
            continue
        pair = frag_info[frag_id]["pair"]
        demand_id = int(toks[2])
        toks = list(toks)
        toks[2] = str(new_demand_id.setdefault(demand_id,
                                               len(new_demand_id)))
        toks[3] = str(new_frag_id.setdefault(frag_id, len(new_frag_id)))
        toks[10] = str(take)              # bytes
        toks[11] = str(take)              # fragment_bytes
        toks[12] = str(demand[pair])      # demand_bytes
        out.append(" ".join(toks))
    return out


def fill_teccl_path(lines, demand):
    # step epoch chunk_source src dst chunk bytes path n1 n2 ...
    # A chunk's chain: steps sharing (chunk_source, chunk). Final dst = the
    # dst never used as a src within the chain.
    chain_steps = OrderedDict()
    for toks in lines:
        if isinstance(toks, str) or toks[0] != "step":
            continue
        key = (int(toks[2]), int(toks[5]))
        chain_steps.setdefault(key, []).append(toks)
    chains = OrderedDict()
    for key, steps in chain_steps.items():
        srcs = set(int(t[3]) for t in steps)
        dsts = [int(t[4]) for t in steps if int(t[4]) not in srcs]
        if len(set(dsts)) != 1:
            raise SystemExit("teccl_path: chunk %s has ambiguous final dst"
                             % (key,))
        chains[key] = {"pair": (key[0], dsts[0]),
                       "capacity": int(steps[0][6])}
    assigned = assign_chains(chains, demand, "teccl_path")

    out = []
    for toks in lines:
        if isinstance(toks, str):
            out.append(toks)
            continue
        if toks[0] != "step":
            out.append(" ".join(toks))
            continue
        key = (int(toks[2]), int(toks[5]))
        take = assigned[key]
        if take == 0:
            continue
        toks = list(toks)
        toks[6] = str(take)
        out.append(" ".join(toks))
    return out


def fill_taccl(lines, demand):
    # chunk addr source final_dst bytes / demand addr source final_dst bytes
    # step step_id epoch addr src dst bytes path ...
    chains = OrderedDict()
    for toks in lines:
        if isinstance(toks, str) or toks[0] != "chunk":
            continue
        addr = int(toks[1])
        chains[addr] = {"pair": (int(toks[2]), int(toks[3])),
                        "capacity": int(toks[4])}
    assigned = assign_chains(chains, demand, "taccl")

    out = []
    for toks in lines:
        if isinstance(toks, str):
            out.append(toks)
            continue
        kind = toks[0]
        if kind in ("chunk", "demand"):
            addr = int(toks[1])
            take = assigned.get(addr, 0)
            if take == 0:
                continue
            toks = list(toks)
            toks[4] = str(take)
            out.append(" ".join(toks))
        elif kind == "step":
            addr = int(toks[3])
            take = assigned.get(addr, 0)
            if take == 0:
                continue
            toks = list(toks)
            toks[6] = str(take)
            out.append(" ".join(toks))
        else:
            out.append(" ".join(toks))
    return out


def fill_syccl(lines, demand):
    # chunk chunk_source chunk bytes [init nodes...]
    # demand src dst chunk bytes
    # event id epoch chunk_source chunk src dst bytes amount copy reduce layer [path ...]
    chains = OrderedDict()
    pair_of_chunk = {}
    for toks in lines:
        if isinstance(toks, str) or toks[0] != "demand":
            continue
        key = (int(toks[1]), int(toks[3]))  # (chunk_source, chunk)
        pair_of_chunk[key] = (int(toks[1]), int(toks[2]))
    for toks in lines:
        if isinstance(toks, str) or toks[0] != "chunk":
            continue
        key = (int(toks[1]), int(toks[2]))
        if key not in pair_of_chunk:
            raise SystemExit("syccl: chunk %s has no demand" % (key,))
        chains[key] = {"pair": pair_of_chunk[key], "capacity": int(toks[3])}
    assigned = assign_chains(chains, demand, "syccl")

    out = []
    for toks in lines:
        if isinstance(toks, str):
            out.append(toks)
            continue
        kind = toks[0]
        if kind == "chunk":
            key = (int(toks[1]), int(toks[2]))
            take = assigned[key]
            if take == 0:
                continue
            toks = list(toks)
            toks[3] = str(take)
            out.append(" ".join(toks))
        elif kind == "demand":
            key = (int(toks[1]), int(toks[3]))
            take = assigned[key]
            if take == 0:
                continue
            toks = list(toks)
            toks[4] = str(take)
            out.append(" ".join(toks))
        elif kind == "event":
            key = (int(toks[3]), int(toks[4]))
            take = assigned[key]
            if take == 0:
                continue
            capacity = chains[key]["capacity"]
            old_bytes = int(toks[7])
            if old_bytes != capacity:
                raise SystemExit(
                    "syccl: partial-chunk event (bytes %d != chunk %d); "
                    "fill tool assumes amount==1 A2A events" %
                    (old_bytes, capacity))
            toks = list(toks)
            toks[7] = str(take)
            out.append(" ".join(toks))
        else:
            out.append(" ".join(toks))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", required=True,
                        help="uniform schedule solved at the padded pow2 size")
    parser.add_argument("--demand", required=True, help="demand matrix file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    nodes, demand = load_demand(args.demand)
    mode, lines = parse_schedule(args.schedule)

    if mode == "teccl_hop":
        out = fill_teccl_hop(lines, demand)
    elif mode == "teccl_path":
        out = fill_teccl_path(lines, demand)
    elif mode == "taccl":
        out = fill_taccl(lines, demand)
    else:
        out = fill_syccl(lines, demand)

    with open(args.output, "w") as f:
        f.write("# filled from %s with %s (fill semantics: plan is padded "
                "capacity, only real bytes transmitted)\n"
                % (args.schedule, args.demand))
        for line in out:
            f.write(line + "\n")

    total = sum(demand.values())
    print("mode %s pairs %d demand_total_bytes %d output %s"
          % (mode, len(demand), total, args.output))


if __name__ == "__main__":
    main()
