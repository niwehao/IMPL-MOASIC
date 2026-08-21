# mosaic-rt

MOSAIC is an alltoallv communication runtime for expert-parallel (MoE)
training. The idea is simple: a CPU protocol core decides *which bytes go
which way and when* (using a globally exchanged demand matrix and runtime
feedback), while the GPU only moves bytes that the CPU has already
scheduled. On the receive side a credit-based backpressure loop keeps
senders from overloading a hot node, so heavily skewed traffic (incast)
stays stable without giving up any throughput on uniform traffic.

It plugs into Megatron-Core as a drop-in replacement for
`all_to_all_single` — training code does not change.

## Directory layout

```
mosaic-rt/
├── src/            The runtime itself (single C++/CUDA library)
│   ├── api.cc          Public API: mosaicAlltoallv[Ex], completion contract
│   ├── engine.cc       Event loop, wave engine, online W adaptation,
│   │                   ledger, credit bookkeeping, delivery
│   ├── topo.cc         Topology, corridor families, resource accounting
│   ├── net.cc          Bootstrap, intra-node shared-memory rings, NIC-unit
│   │                   consensus
│   ├── verbs.cc        Inter-node RDMA data plane + control plane
│   │                   (host-bounce; InfiniBand and RoCE, auto-detected)
│   ├── erpc_ctrl.cc    Fallback control plane (eRPC/DPDK) for machines
│   │                   without an RDMA NIC
│   ├── consumer_kernel.cu  The single GPU copy kernel ("consumer")
│   └── internal.h      Shared contracts (wire format, Comm state)
├── aux/            Everything around the runtime
│   ├── build/          Build output (libmosaic.so, bench_a2a, tests)
│   ├── include/        Public header mosaic/mosaic.h
│   ├── python/         ctypes binding + mosaic_torch (Megatron drop-in)
│   ├── bench/          C++ benchmark (bench_a2a)
│   ├── tests/          Unit/replay tests and fixtures
│   ├── scripts/        Two-host launch helpers
│   ├── tables/         Static topology tables
│   ├── tools/          PTX regeneration for the consumer kernel
│   └── third_party/    Vendored eRPC patch + DPDK install tree
├── nccl/           NCCL baseline: bench_nccl.py (same wall-clock harness
│                   and end-to-end verification for both backends) + run2.sh
├── sim/            The MOSAIC simulator lineage (csg-htsim, FAST, syccl-grb,
│                   taccl). The protocol core is replay-checked against it.
└── note/           timeline.md (development diary), experiment.md
                    (current results and setup)
```

## Build

```bash
cd Impl/mosaic-rt
make -j8              # outputs land in aux/build/
```

Requirements: CUDA (>= 12.x), libibverbs, a C++17 compiler. If you edit
`consumer_kernel.cu`, run `bash aux/tools/gen_consumer_ptx.sh` first
(needs nvcc <= 12.6), then `make`.

## Using it in training (Megatron-Core)

```python
import mosaic_torch as mt        # add Impl/mosaic-rt/aux/python to sys.path

mt.init(backend="mosaic", gpus_per_host=4)   # after dist.init_process_group
mt.patch_megatron()              # swaps mcore's all_to_all primitive
```

Environment needed at launch:

- `MOSAIC_HOSTS` — comma-separated IPv4 list, one per node (IP addresses,
  not hostnames)
- `MOSAIC_G` — GPUs per node
- `MOSAIC_JOB` — any small integer unique per job (keeps shared-memory
  names from colliding)

Selecting `backend="nccl"` skips the patch entirely and runs native NCCL,
which is how the baseline is measured: same model, same data, same seeds —
the two backends produce bit-identical loss columns.

A complete end-to-end example is `Impl/moe-rt/train_moe_mcore.py`
(`MT_BACKEND=nccl|mosaic`).

## Benchmarks

Both backends run through the same Python harness with the same wall
clock and built-in end-to-end data verification:

```bash
cd Impl
bash mosaic-rt/nccl/run2.sh mosaic-rt/nccl/bench_nccl.py \
     --backend mosaic --mode hotdst --alpha 2 --msg 8388608
```

- `--mode uniform|hotdst|persrc|khot` — traffic pattern
- `--alpha` — skew strength for hotdst (0 = uniform, 2 = extreme incast)
- `--msg` — bytes per pair
- `--iters/--warmup` — samples (median reported)

The C++ benchmark `aux/build/bench_a2a` exercises the same engine
directly (see `aux/scripts/run2host.sh`).

Run the regression suite after any change:

```bash
cd Impl && bash regression.sh     # 20 checks: bit-exactness, replay vs
                                  # simulator, torch drop-in, loss parity
```

## Parameters worth knowing

Everything is an environment variable with a sensible default; none of
them change *how* data flows — only capacities and policies. The ones you
may actually want to touch:

| Variable | Default | What it does |
|---|---|---|
| `MOSAIC_WAVES` | 6 | Upper bound for the number of waves a collective may be split into. The runtime adapts the actual W online per traffic shape; leave this alone unless experimenting. |
| `MOSAIC_SLICE_BYTES` | 1 MiB | Slice size (transfer granularity). Larger = less per-slice overhead; smaller = finer pipelining. |
| `MOSAIC_CREDIT_HOT_W0` | 2 | Initial burst allowance (slices per sender) toward a hot destination before receiver credits take over. Raise for a fuller pipeline at the cost of getting closer to the incast cliff; lower for stronger protection. |
| `MOSAIC_RDMA_RX_SLOTS` | 32 MB worth | Posted receive window per peer. |
| `MOSAIC_KCONS` | 4,4,512 | Consumer-kernel shape (blocks, channels, threads). Calibrated per GPU generation; 4 blocks is the A100 sweet spot. |
| `MOSAIC_NODE_INFLIGHT_MB` | 24 | Per-destination-node in-flight byte cap on the sender. |
| `MOSAIC_MEM_BUDGET_MB` | unset | One-knob cap for GPU-side buffer pools when training is tight on memory. |
| `MOSAIC_CTRL` | auto | Control plane: `verbs` (RDMA SEND, default when a NIC is present) or `erpc` (DPDK fallback). |
| `MOSAIC_PHASE` / `MOSAIC_TRACE` | 0 | Observability: per-collective phase timing / binary event trace. No effect on behavior. |

Platform things (device selection, HCA choice, NIC sharing topology) are
auto-detected at runtime; there are no static host lists to maintain.

## Current results

See `note/experiment.md` for the current benchmark table (16× A100 over
4 nodes, one shared HDR200 NIC per node) and the exact measurement
methodology, and `note/timeline.md` for how the system got here.
