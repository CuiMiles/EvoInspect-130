#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input="${1:-${root}/demo/sample_input.png}"
output="${2:-${root}/demo_output}"
python "${root}/run_demo.py" --image "${input}" --output-dir "${output}"

