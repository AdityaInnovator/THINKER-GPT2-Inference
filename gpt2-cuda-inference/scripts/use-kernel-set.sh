#!/usr/bin/env bash
# Switch active kernels/ between baseline (M1) and optimized (portfolio default).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-optimized}"

install_dir() {
  local src_dir="$1"
  if [[ ! -d "$src_dir" ]]; then
    echo "Error: missing $src_dir"
    exit 1
  fi
  mkdir -p kernels
  shopt -s nullglob
  for f in "$src_dir"/*.cuh; do
    cp "$f" "kernels/$(basename "$f")"
    echo "  kernels/$(basename "$f")"
  done
  shopt -u nullglob
}

case "$MODE" in
  baseline)
    echo "Installing baseline (Milestone 1) kernels..."
    install_dir "kernels_baseline"
    rm -f kernels/cp_async_utils.cuh
    ;;
  optimized)
    echo "Installing optimized kernel set..."
    install_dir "kernels_optimized"
    # attention-kv-cache references softmax in same directory
    sed -i.bak 's|#include "../kernels/softmax.cuh"|#include "softmax.cuh"|g' kernels/attention.cuh 2>/dev/null || \
      sed -i '' 's|#include "../kernels/softmax.cuh"|#include "softmax.cuh"|g' kernels/attention.cuh
    rm -f kernels/attention.cuh.bak
    ;;
  *)
    echo "Usage: $0 {baseline|optimized}"
    exit 1
    ;;
esac

echo "Done ($MODE). Rebuild with: make clean all"
