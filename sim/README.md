# XCCL

A research repository for **fair, packet-level comparison of collective
communication schedulers** on the htsim simulator. It puts
**TE-CCL / TACCL / SyCCL** (offline solvers), **FAST** (online Birkhoff
scheduler, ported from the official codebase) and our **MOSAIC**
(multi-corridor sub-schedule assembly + online waterfilling feedback) on
the *same* topologies with the *same* execution semantics, and compares
AllToAll completion time against a **static ECMP** baseline. Both uniform
and non-uniform (MoE-style skewed) A2A workloads are supported.

## Repository layout

| Directory | Contents |
|---|---|
| `csg-htsim/` | Simulator body (forked from Broadcom/csg-htsim, see THIRD_PARTY_REPOS.md) |
| `csg-htsim/src/` | Main (`main_regular.cpp`), SmartNode, workflow runtime |
| `csg-htsim/topo/` | **All topologies** (dgx_rail / ndv2 / ndv2_32 / clos / regular / flex_rail / flex_pod / flex_rail2p) |
| `csg-htsim/scheduler/` | **All schedulers** (static_ecmp / teccl / taccl / syccl execution runtimes; `fast_schedule.h` = FAST port; `mosaic_schedule.h` = MOSAIC) |
| `csg-htsim/script/` | One-shot experiment scripts per topology (`run_*.sh` + `common.sh`) |
| `csg-htsim/tools/` | Schedule generation / conversion / fill tools + cut lower bounds (Python) |
| `csg-htsim/cache/` | Demand matrices, solver artifacts, `.sched` files (content-addressed caching) |
| `csg-htsim/experiments/mosaic_x1/` | **Experiment archive**: 25 groups, each with results.csv + plot.png + README |
| `Note/` | Reports and docs: `report_mosaic_eps.md` (EPS main report, v6) + `report_mosaic_ocs.md` (OCS main report), `report_scale_1024.md`/`report_switch_centric.md` (headline results), `SOTA.md`, `fail.md` (negative results), `handoff.md`; exploration history in `explore/` |
| `Note/Design/` | Per-baseline deep dives (`baseline_{teccl,taccl,syccl,fast,ecmp_basic}.md`), mechanism comparison, FAST paper notes |
| `TE-CCL/` `taccl/` `syccl-grb/` | The three solvers (origins & pinned HEADs in THIRD_PARTY_REPOS.md; only thin adapter changes by us) |
| `FAST/` | Official FAST reference code (source of the line-by-line port) |
| `tccl/` | TCCL (NCCL fork, runtime-mechanism reference) |
| `Paper/` | Paper PDFs |

## Setup

### 1. Simulator (required, minimal dependencies)

- Linux, g++ ≥ 9 (C++11), make.
- Build:
  ```bash
  cd csg-htsim/sim && make            # htsim core
  cd tests && make htsim_smart_regular
  ```
- This is all you need to run **static_ecmp / basic / FAST / MOSAIC**
  (zero external dependencies, plans computed online). `cache/` already
  ships all pre-solved `.sched` files and demand matrices, so the solver
  comparisons can be **reproduced directly from cache** without installing
  the solver environments below.

### 2. Solver environment (only to re-solve schedules)

- **conda env** (Python 3.11):
  ```bash
  conda create -n myenv python=3.11
  conda activate myenv
  pip install gurobipy numpy
  pip install -e ./TE-CCL          # provides the teccl CLI
  pip install -e ./taccl           # provides the taccl CLI
  ```
- **Gurobi**: required by all three solvers (academic WLS / named-user
  license). TACCL solve limits via env vars:
  `TACCL_ROUTING_TIME_LIMIT` / `TACCL_SCHEDULER_TIME_LIMIT` (seconds,
  default 600 each).
- **SyCCL** (C++, needs the Gurobi C++ libs):
  ```bash
  cd syccl-grb && mkdir -p build && cd build && cmake .. && make
  ```
- Scripts default to conda at `/home/weihaoni/miniconda3`; override with
  `CONDA` / `CONDA_ENV` (see `csg-htsim/script/common.sh`).

### 3. Plotting (optional)

- matplotlib (already in the conda env):
  ```bash
  /path/to/conda/envs/myenv/bin/python3 csg-htsim/experiments/mosaic_x1/plot_all.py
  ```

## Quick start

```bash
cd csg-htsim

# Uniform A2A on dgx_rail, all five schedulers (schedules from cache):
bash script/run_dgx_rail.sh

# Non-uniform A2A (hot-dst zipf; solvers use padded+fill semantics):
NONUNIFORM=1 bash script/run_dgx_rail.sh

# Subset of schemes / force re-solving:
SCHEDULERS="static_ecmp fast" NONUNIFORM=1 bash script/run_ndv2_32.sh
FORCE_RESOLVE=1 bash script/run_clos_32.sh

# MOSAIC vs FAST crossover (flex_rail, sweep scale-up bandwidth).
# -mosaic_corridors 6 is the v5 recommended default; the code default (4)
# reproduces the v4 numbers bit-for-bit:
./sim/tests/htsim_smart_regular -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
  -packet_size 1024 -chunk_size 33554432 -scheduler mosaic_schedule \
  -mosaic_corridors 6 -workload A2A -all_to_all -store_forward -quiet \
  -demand_file cache/nonuniform/demand_n32_zipf1.0_seed42.txt
```

Every run prints one line `validation PASSED/FAILED ... completion_us ...`;
PASSED means delivered bytes match the demand matrix byte-for-byte. Only
PASSED numbers may enter comparison tables.

## Where the results are

- **MOSAIC pre-paper document** (motivation → design → math → 25
  experiment groups → limitations): `Note/report_mosaic_eps.md`, with the
  experiment archive in `csg-htsim/experiments/mosaic_x1/`.
- **Operations manual / pitfalls / reuse rules** (start here if you are
  picking up the project): `Note/handoff.md`, plus the hard rules in
  `CLAUDE.md`.
- **Per-baseline deep dives** (code pointers, strengths, limits, traps):
  `Note/Design/baseline_*.md`.
- **Four solvers × five topologies panorama** (uniform + non-uniform):
  `Note/explore/experiment_summary.md`.
- **Solver topology-expressiveness comparison**:
  `Note/explore/solver_topology_capabilities.md`.

## License note

`FAST/` is academic / non-commercial; `tccl/`, `csg-htsim/`, `TE-CCL/`,
`taccl/`, `syccl-grb/` keep their original LICENSE files. This repository
is for academic research only.
