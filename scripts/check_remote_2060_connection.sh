#!/usr/bin/env bash
set -Eeuo pipefail

target="${1:-}"
[[ -n "${target}" ]] || { printf 'usage: %s user@host-or-ssh-alias\n' "$0" >&2; exit 2; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "${target}" \
  'set -eu; hostname; command -v python3; command -v nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true; df -h .'
