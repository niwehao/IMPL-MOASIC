#!/bin/bash
# 38 组:统一方案 MOSAIC-U = C(wiring) + W(拓扑) + PC_auto(diag∧wiring 门)
# + PACE(三门) [+ LP(cube 可选)] 的缺测补齐。
# nohup script/x38_unified.sh > cache/nonuniform/x38_unified.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
BIN=./sim/tests/htsim_smart_regular
DZ=cache/nonuniform/demand_n32_zipf1.0_seed42.txt
U="-mosaic_pair_color_auto -mosaic_cold_pace 1"
COMMON="-nodes 32 -packet_size 1024 -workload A2A -all_to_all -store_forward -quiet -scheduler mosaic_schedule"

run() { local tag="$1"; shift
  echo "=== RUN $tag"
  timeout 1200 $BIN $COMMON "$@" 2>&1 \
    | grep -E "pair_color|cold_pace|validation|mosaic_plan" | sed "s/^/$tag /"
  echo "=== END $tag"
}

# rail 缺测 nv 点 + 360 回归
for nv in 90 180 420 2400 360; do
  run "rail_nv${nv}" -topology flex_rail -flex_nv_gbps $nv \
      -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 $U
done
# rail 360 uniform(门回归:PACE 应自动关)
run "rail_nv360_uniform" -topology flex_rail -flex_nv_gbps 360 \
    -chunk_size 2097152 -message_size 2097152 -mosaic_corridors 6 $U

# pod(S=4,nv360,上行总额扫)
for up in 1440 720 360; do
  run "pod_up${up}" -topology flex_pod -flex_spines 4 -flex_nv_gbps 360 \
      -flex_taper_gbps $up -chunk_size 33554432 -demand_file $DZ \
      -mosaic_corridors 5 $U
done

# rail2p(PC 应被 wiring 门关掉 → 有效 C6+PACE)
run "rail2p_2400_sym" -topology flex_rail2p -flex_nv_gbps 2400 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 $U
run "rail2p_360_b2" -topology flex_rail2p -flex_nv_gbps 360 \
    -flex_planeb_factor 2 -chunk_size 33554432 -demand_file $DZ \
    -mosaic_corridors 6 $U
run "rail2p_90_b4" -topology flex_rail2p -flex_nv_gbps 90 \
    -flex_planeb_factor 4 -chunk_size 33554432 -demand_file $DZ \
    -mosaic_corridors 6 $U

# cube mix 全栈 + 统一
run "cube_mix_fullstack" -topology flex_cube -chunk_size 33554432 \
    -demand_file $DZ -mosaic_corridors 6 -mosaic_waves 16 \
    -mosaic_lp_intra cache/mosaic_lp/cube_mix_rot0.txt -mosaic_lp_local \
    -mosaic_lp_spread cache/mosaic_lp/jh_mix_rot0.txt $U

# clos(37 组桥接:zipf@2:1 与 incast1 默认 buffer)
run "clos_zipf_up400" -topology clos -clos_leafs 4 -clos_spines 1 \
    -chunk_size 33554432 -demand_file $DZ -mosaic_corridors 6 $U
run "clos_incast1" -topology clos -clos_leafs 4 -clos_spines 1 \
    -chunk_size 67108864 -demand_file cache/nonuniform/demand_n32_incast1.txt \
    -mosaic_corridors 6 $U

touch cache/nonuniform/x38_unified.DONE
echo ALL_DONE
