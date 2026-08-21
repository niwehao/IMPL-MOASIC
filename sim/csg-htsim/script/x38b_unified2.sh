#!/bin/bash
# 38 组第二批:统一方案在 cube 其余画像 / khot / 64GPU / drift 的缺测。
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
U="-mosaic_pair_color_auto -mosaic_cold_pace 1"
COMMON="-packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule"
LP=cache/mosaic_lp

run() { local tag="$1"; shift
  echo "=== RUN $tag"
  timeout 1200 $BIN $COMMON "$@" 2>&1 \
    | grep -E "pair_color|cold_pace|validation" | sed "s/^/$tag /"
  echo "=== END $tag"
}

# cube 画像(PC 会被 lp_spread 门自动关;无 spread 的 full 系 PC 开)
run "cube_400x16" -nodes 32 -topology flex_cube -flex_cube_speeds 400x16 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 -mosaic_waves 12 \
    -mosaic_lp_spread $LP/jd_400_k6.txt $U
run "cube_100x16" -nodes 32 -topology flex_cube -flex_cube_speeds 100x16 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 -mosaic_waves 16 \
    -mosaic_lp_intra $LP/cube_100.txt -mosaic_lp_local \
    -mosaic_lp_spread $LP/jh_100.txt $U
run "cube_50x16" -nodes 32 -topology flex_cube -flex_cube_speeds 50x16 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 -mosaic_waves 16 \
    -mosaic_lp_intra $LP/cube_50.txt -mosaic_lp_local $U
run "cube_full300" -nodes 32 -topology flex_cube -flex_cube_full_gbps 300 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 -mosaic_waves 12 \
    -mosaic_lp_spread $LP/jh_full300.txt $U
run "cube_full50" -nodes 32 -topology flex_cube -flex_cube_full_gbps 50 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 \
    -mosaic_lp_spread $LP/jh_full50.txt $U

# khot 多热点(rail nv360)
for k in 2 4 8; do
  run "khot${k}" -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
      -chunk_size 33554432 -demand_file cache/nonuniform/demand_n32_khot${k}_seed42.txt \
      -mosaic_corridors 6 $U
done

# 64 GPU(8×8,rail nv360)
run "gpu64" -nodes 64 -topology flex_rail -flex_hosts 8 -flex_nv_gbps 360 \
    -chunk_size 33554432 -demand_file cache/nonuniform/demand_n64_zipf1.0_seed42.txt \
    -mosaic_corridors 6 $U

# 先验漂移(prior=seed43,跑 seed42;rail nv360)
run "drift_prior43" -nodes 32 -topology flex_rail -flex_nv_gbps 360 \
    -chunk_size 33554432 -demand_file $DZ \
    -mosaic_prior_demand_file cache/nonuniform/demand_n32_zipf1.0_seed43.txt \
    -mosaic_corridors 6 $U

touch cache/nonuniform/x38b.DONE
echo ALL_DONE
