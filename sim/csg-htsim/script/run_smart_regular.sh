#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/sim/tests/htsim_smart_regular"
SRC_ENTRY="$ROOT_DIR/src/main_regular.cpp"

# Edit these defaults for normal runs. Environment variables with the
# same names still override them, and extra CLI args are passed through.
DEFAULT_NODES=128
DEFAULT_DEGREE=8
# Default workload contract: every P2P pair carries 2MB, packet size 1KB,
# store-forward forwarding. Keep these stable across topology changes.
DEFAULT_MESSAGE_SIZE_BYTES=2097152
DEFAULT_CHUNK_SIZE_BYTES=2097152
DEFAULT_ROUNDS=1
# Packet::_size is uint16_t in htsim, so keep packet size <= 65535 unless
# the core Packet size type is widened.
DEFAULT_PACKET_SIZE_BYTES=1024
DEFAULT_LINK_RATE_GBPS=100
DEFAULT_ECN_THRESHOLD_BYTES=0
DEFAULT_PAUSE_THRESHOLD_BYTES=0
DEFAULT_SPLIT_MINUS=1
DEFAULT_SPLIT_ZERO=0
DEFAULT_SPLIT_PLUS=0
DEFAULT_TOPOLOGY=regular
DEFAULT_CLOS_LEAFS=0
DEFAULT_CLOS_SPINES=0
DEFAULT_REGULAR_HETERO_RING=0
DEFAULT_REGULAR_HETERO_MULTIPLIER=2
DEFAULT_SCHEDULER=basic
DEFAULT_WORKLOAD=A2A
DEFAULT_ALL_TO_ALL=1
DEFAULT_SCHEDULE_FILE=
DEFAULT_STORE_FORWARD=1
# For non-all-to-all
DEFAULT_SRC=0
DEFAULT_DST=4
DEFAULT_QUIET=1

NODES="${NODES:-$DEFAULT_NODES}"
DEGREE="${DEGREE:-$DEFAULT_DEGREE}"
MESSAGE_SIZE_BYTES="${MESSAGE_SIZE_BYTES:-$DEFAULT_MESSAGE_SIZE_BYTES}"
CHUNK_SIZE_BYTES="${CHUNK_SIZE_BYTES:-$DEFAULT_CHUNK_SIZE_BYTES}"
ROUNDS="${ROUNDS:-$DEFAULT_ROUNDS}"
PACKET_SIZE_BYTES="${PACKET_SIZE_BYTES:-$DEFAULT_PACKET_SIZE_BYTES}"
LINK_RATE_GBPS="${LINK_RATE_GBPS:-$DEFAULT_LINK_RATE_GBPS}"
ECN_THRESHOLD_BYTES="${ECN_THRESHOLD_BYTES:-$DEFAULT_ECN_THRESHOLD_BYTES}"
PAUSE_THRESHOLD_BYTES="${PAUSE_THRESHOLD_BYTES:-$DEFAULT_PAUSE_THRESHOLD_BYTES}"
SPLIT_MINUS="${SPLIT_MINUS:-$DEFAULT_SPLIT_MINUS}"
SPLIT_ZERO="${SPLIT_ZERO:-$DEFAULT_SPLIT_ZERO}"
SPLIT_PLUS="${SPLIT_PLUS:-$DEFAULT_SPLIT_PLUS}"
TOPOLOGY="${TOPOLOGY:-$DEFAULT_TOPOLOGY}"
CLOS_LEAFS="${CLOS_LEAFS:-$DEFAULT_CLOS_LEAFS}"
CLOS_SPINES="${CLOS_SPINES:-$DEFAULT_CLOS_SPINES}"
REGULAR_HETERO_RING="${REGULAR_HETERO_RING:-$DEFAULT_REGULAR_HETERO_RING}"
REGULAR_HETERO_MULTIPLIER="${REGULAR_HETERO_MULTIPLIER:-$DEFAULT_REGULAR_HETERO_MULTIPLIER}"
SCHEDULER="${SCHEDULER:-$DEFAULT_SCHEDULER}"
WORKLOAD="${WORKLOAD:-$DEFAULT_WORKLOAD}"
ALL_TO_ALL="${ALL_TO_ALL:-$DEFAULT_ALL_TO_ALL}"
SCHEDULE_FILE="${SCHEDULE_FILE:-$DEFAULT_SCHEDULE_FILE}"
STORE_FORWARD="${STORE_FORWARD:-$DEFAULT_STORE_FORWARD}"
SRC="${SRC:-$DEFAULT_SRC}"
DST="${DST:-$DEFAULT_DST}"
QUIET="${QUIET:-$DEFAULT_QUIET}"
EFFECTIVE_SCHEDULER="$SCHEDULER"
if [[ -n "$SCHEDULE_FILE" && "$SCHEDULER" != "syccl_schedule" && "$SCHEDULER" != "taccl_schedule" ]]; then
    EFFECTIVE_SCHEDULER=teccl_schedule
fi

MODE_ARGS=()
if [[ "$ALL_TO_ALL" != "0" ]]; then
    MODE_ARGS+=("-all_to_all")
else
    MODE_ARGS+=("-src" "$SRC" "-dst" "$DST")
fi

if [[ "$QUIET" != "0" ]]; then
    MODE_ARGS+=("-quiet")
fi

if [[ -n "$SCHEDULE_FILE" ]]; then
    MODE_ARGS+=("-schedule_file" "$SCHEDULE_FILE")
fi

if [[ "$STORE_FORWARD" != "0" ]]; then
    MODE_ARGS+=("-store_forward")
fi

if [[ "$REGULAR_HETERO_RING" != "0" ]]; then
    MODE_ARGS+=("-regular_hetero_ring")
fi
MODE_ARGS+=("-regular_hetero_multiplier" "$REGULAR_HETERO_MULTIPLIER")

echo "Building htsim_smart_regular if needed..."
echo "Source entry: $SRC_ENTRY"
make -C "$ROOT_DIR/sim/tests" htsim_smart_regular

echo "Running htsim_smart_regular..."
FORWARDING_MODE="cut_through"
if [[ "$STORE_FORWARD" != "0" ]]; then
    FORWARDING_MODE="store_forward"
fi
echo "  nodes=$NODES degree=$DEGREE topology=$TOPOLOGY clos_leafs=$CLOS_LEAFS clos_spines=$CLOS_SPINES regular_hetero_ring=$REGULAR_HETERO_RING regular_hetero_multiplier=$REGULAR_HETERO_MULTIPLIER rounds=$ROUNDS message_size=$MESSAGE_SIZE_BYTES chunk_size=$CHUNK_SIZE_BYTES packet_size=$PACKET_SIZE_BYTES link_rate_gbps=$LINK_RATE_GBPS node_pacing_gbps=$LINK_RATE_GBPS ecn_threshold=$ECN_THRESHOLD_BYTES pause_threshold=$PAUSE_THRESHOLD_BYTES ecn_feedback=reverse_path feedback_drain=enabled scheduler=$EFFECTIVE_SCHEDULER forwarding=$FORWARDING_MODE workload=$WORKLOAD split=$SPLIT_MINUS/$SPLIT_ZERO/$SPLIT_PLUS all_to_all=$ALL_TO_ALL src=$SRC dst=$DST quiet=$QUIET schedule_file=${SCHEDULE_FILE:-none} args=$*"

exec "$BIN" \
    -nodes "$NODES" \
    -degree "$DEGREE" \
    -topology "$TOPOLOGY" \
    -clos_leafs "$CLOS_LEAFS" \
    -clos_spines "$CLOS_SPINES" \
    -rounds "$ROUNDS" \
    -message_size "$MESSAGE_SIZE_BYTES" \
    -chunk_size "$CHUNK_SIZE_BYTES" \
    -packet_size "$PACKET_SIZE_BYTES" \
    -link_rate_gbps "$LINK_RATE_GBPS" \
    -ecn_threshold_bytes "$ECN_THRESHOLD_BYTES" \
    -pause_threshold_bytes "$PAUSE_THRESHOLD_BYTES" \
    -split_minus "$SPLIT_MINUS" \
    -split_zero "$SPLIT_ZERO" \
    -split_plus "$SPLIT_PLUS" \
    -scheduler "$EFFECTIVE_SCHEDULER" \
    -workload "$WORKLOAD" \
    "${MODE_ARGS[@]}" \
    "$@"
