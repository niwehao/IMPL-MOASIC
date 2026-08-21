#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$EXAMPLE_DIR/../.." && pwd)"

"$ROOT_DIR/tools/teccl_generate_schedule.py" \
    --topo "$EXAMPLE_DIR/topo.json" \
    --workflow "$EXAMPLE_DIR/workflow.json" \
    --teccl-root "$ROOT_DIR/../TE-CCL" \
    --cache-dir "$ROOT_DIR/cache/teccl_schedules" \
    --output "$EXAMPLE_DIR/schedule.sched"

SCHEDULE_FILE="$EXAMPLE_DIR/schedule.sched" \
NODES=16 \
DEGREE=4 \
MESSAGE_SIZE_BYTES=1048576 \
CHUNK_SIZE_BYTES=1048576 \
PACKET_SIZE_BYTES=65535 \
QUIET=1 \
"$ROOT_DIR/run_smart_regular.sh"
