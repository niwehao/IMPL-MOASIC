#!/bin/bash
# Copyright (c) 2025, Alibaba Cloud and its affiliates
# SPDX-License-Identifier: Apache-2.0

set -e
set -u

PAIRS=(
    "h800:0.5"
    "a100:0.5"
    "a100-2:0.5"
    "h800-32:1.0"
    "h800-scale-test:0.5"
    "a100-scale-test:0.5"
    "h800-abl-ratio:0.5"
    "h800-abl-nodiff:0.5"
    "h800-abl-withdiff:0.5"
)

main() {
    # Loop through each defined pair
    for pair in "${PAIRS[@]}"; do
        local exp="${pair%:*}"
        local suffix="${pair#*:}"
        if [[ -z "$exp" || -z "$suffix" || "$exp" == "$suffix" ]]; then
            echo "Skipping invalid entry in PAIRS array: '${pair}'"
            continue
        fi
        local result_dir="syn_res/results-${suffix}"
        mkdir -p "${result_dir}"

        echo "--------------------------------------------------"
        echo "Running experiment: '${exp}'"
        echo "With suffix:        '${suffix}'"
        echo "Results directory:  '${result_dir}'"
        echo "--------------------------------------------------"

        # Run the specific experiment-suffix combination
        python3 ../scripts/runexp.py "${exp}" "${result_dir}" "${suffix}"
    done

    echo "All specified experiments completed successfully."
}

main
