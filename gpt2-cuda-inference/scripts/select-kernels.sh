#!/usr/bin/env bash
# Install one optimization variant into active kernels/.
# Usage: ./scripts/select-kernels.sh <variant-folder-name> [apply|revert]
#
# Examples:
#   ./scripts/select-kernels.sh attention-flash apply
#   ./scripts/select-kernels.sh matmul-cp-async-pipeline apply
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Legacy course IDs -> descriptive folder (optional shorthand)
resolve_variant_dir() {
  local name="$1"
  case "$name" in
    req_0|kernels_req_0) echo "matmul-shared-register-tiling" ;;
    req_1|kernels_req_1) echo "matmul-tensor-core-wmma" ;;
    req_2|kernels_req_2) echo "matmul-cublas" ;;
    req_3|kernels_req_3) echo "layernorm-softmax-reduction" ;;
    req_4|kernels_req_4) echo "attention-flash" ;;
    req_5|kernels_req_5) echo "attention-local-window" ;;
    req_6|kernels_req_6) echo "attention-kv-cache" ;;
    op_7|kernels_op_7) echo "matmul-block-size-sweep" ;;
    op_8|kernels_op_8) echo "attention-constant-memory" ;;
    op_9|kernels_op_9) echo "attention-restrict-ptr" ;;
    op_10|kernels_op_10) echo "matmul-split-k" ;;
    op_11|kernels_op_11) echo "attention-memory-swizzle" ;;
    op_12|kernels_op_12) echo "attention-shared-memory-padding" ;;
    op_17|kernels_op_17) echo "matmul-cp-async-pipeline" ;;
    op_18|kernels_op_18) echo "matmul-block-rasterization" ;;
    *) echo "$name" ;;
  esac
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <variant-name> [apply|revert]"
  echo ""
  echo "List variants: ls variants/"
  echo "See: variants/README.md"
  exit 1
fi

RAW_NAME="$1"
VARIANT="$(resolve_variant_dir "$RAW_NAME")"
ACTION="${2:-apply}"

SRC_DIR="variants/${VARIANT}"
if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: variant not found: $SRC_DIR"
  echo "Available:"
  ls -1 variants/
  exit 1
fi

DST_DIR="kernels"
MANIFEST="${DST_DIR}/.${VARIANT}_backups.list"

if [[ "$ACTION" == "apply" ]]; then
  : > "$MANIFEST"
  shopt -s nullglob
  for src in "$SRC_DIR"/*.cuh; do
    base=$(basename "$src")
    dst="${DST_DIR}/${base}"
    bak="${dst}.org"
    if [[ -f "$dst" && ! -f "$bak" ]]; then
      cp "$dst" "$bak"
      echo "$base" >> "$MANIFEST"
    fi
    cp "$src" "$dst"
    echo "Copied: $src -> $dst"
  done
  shopt -u nullglob
  if [[ -f kernels/attention.cuh ]]; then
    sed -i.bak 's|#include "../kernels/softmax.cuh"|#include "softmax.cuh"|g' kernels/attention.cuh 2>/dev/null || \
      sed -i '' 's|#include "../kernels/softmax.cuh"|#include "softmax.cuh"|g' kernels/attention.cuh
    rm -f kernels/attention.cuh.bak
  fi
  echo "Applied variant: $VARIANT"
  echo "Revert with: $0 $RAW_NAME revert"
  exit 0
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "No manifest for $VARIANT"
  exit 0
fi

while IFS= read -r base; do
  [[ -z "$base" ]] && continue
  bak="${DST_DIR}/${base}.org"
  dst="${DST_DIR}/${base}"
  if [[ -f "$bak" ]]; then
    cp "$bak" "$dst"
    rm -f "$bak"
    echo "Restored: $base"
  fi
done < "$MANIFEST"
rm -f "$MANIFEST"
echo "Reverted: $VARIANT"
