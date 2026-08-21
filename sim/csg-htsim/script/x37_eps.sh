#!/bin/bash
# 37 组 EPS:switch-centric clos(n=32)incast/oversub/collision 批次。
# 单层后台运行:nohup script/x37_eps.sh > cache/nonuniform/x37_eps.log 2>&1 &
# 结束标记:cache/nonuniform/x37_eps.DONE
set -u
cd "$(dirname "$0")/.."
BIN=${BIN:-./sim/tests/htsim_smart_regular}
CACHE=cache/nonuniform
TIMEOUT=${TIMEOUT:-1200}   # 20min/run 政策
NODES=32
LEAFS=4

run() {  # tag scheduler-args chunk topo-extra...
    local tag="$1"; shift
    local sched="$1"; shift
    local chunk="$1"; shift
    echo "=== RUN $tag"
    timeout "$TIMEOUT" $BIN -nodes $NODES -topology clos -clos_leafs $LEAFS \
        -packet_size 1024 -chunk_size "$chunk" \
        -workload A2A -all_to_all -store_forward -quiet \
        $sched "$@" 2>&1 | grep -E "mosaic_plan|fast_plan|validation|drops" \
        | sed "s/^/$tag /"
    local rc=${PIPESTATUS[0]}
    [ "$rc" = 124 ] && echo "$tag TIMEOUT(${TIMEOUT}s policy)"
    echo "=== END $tag"
}

M="-scheduler mosaic_schedule -mosaic_corridors 6"
MP="$M -mosaic_pair_color_auto"
F="-scheduler fast_schedule -fast_alpha_beta_guard"
E="-scheduler static_ecmp"

# ---------- P1: incast m sweep(spines=1, 2:1 默认 uplink,默认 1GB buffer)
for m in 1 2 4 8; do
    D=$CACHE/demand_n32_incast${m}.txt
    CH=$(awk '/padded_pow2_bytes/{print $NF}' "$D")
    for cfg in "mosaic|$M" "mosaic_pc|$MP" "fast|$F" "ecmp|$E"; do
        run "p1_incast${m}_${cfg%%|*}" "${cfg#*|}" "$CH" -clos_spines 1 -demand_file "$D"
    done
done

# ---------- P1b: FAILED 者的 buffer 补偿(4GB/16GB;对手最有利)
for m in ${P1B_M:-1 2 4 8}; do
    D=$CACHE/demand_n32_incast${m}.txt
    CH=$(awk '/padded_pow2_bytes/{print $NF}' "$D")
    for mb in 4096 16384; do
        for cfg in "fast|$F" "ecmp|$E"; do
            run "p1b_incast${m}_${cfg%%|*}_buf${mb}" "${cfg#*|}" "$CH" \
                -clos_spines 1 -demand_file "$D" -flex_fabric_buffer_mb $mb
        done
    done
done

# ---------- P2: oversubscription 扫(spines=1;uniform 与 zipf1.0)
DZ=$CACHE/demand_n32_zipf1.0_seed42.txt
for up in 800 400 200 100; do
    for cfg in "mosaic|$M" "mosaic_pc|$MP" "fast|$F" "ecmp|$E"; do
        run "p2_uni_up${up}_${cfg%%|*}" "${cfg#*|}" 2097152 -clos_spines 1 \
            -clos_uplink_gbps $up -message_size 2097152
        run "p2_zipf_up${up}_${cfg%%|*}" "${cfg#*|}" 33554432 -clos_spines 1 \
            -clos_uplink_gbps $up -demand_file "$DZ"
    done
done

# ---------- P3: spine hash collision(spines=4,uplink auto=100G/link)
for perm in permcollide permbal; do
    D=$CACHE/demand_n32_${perm}_s4.txt
    CH=$(awk '/padded_pow2_bytes/{print $NF}' "$D")
    for cfg in "mosaic|$M" "mosaic_pc|$MP" "fast|$F" "ecmp|$E"; do
        run "p3_${perm}_${cfg%%|*}" "${cfg#*|}" "$CH" -clos_spines 4 -demand_file "$D"
    done
done

# ---------- P4: mixed incast+background(spines=1)
DM=$CACHE/demand_n32_mixed_incast.txt
CHM=$(awk '/padded_pow2_bytes/{print $NF}' "$DM")
for cfg in "mosaic|$M" "mosaic_pc|$MP" "fast|$F" "ecmp|$E"; do
    run "p4_mixed_${cfg%%|*}" "${cfg#*|}" "$CHM" -clos_spines 1 -demand_file "$DM"
done

touch $CACHE/x37_eps.DONE
echo ALL_DONE
