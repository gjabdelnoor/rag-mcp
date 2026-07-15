#!/usr/bin/env bash
# Launch llama-server for the small text embedder. Device-pinned via env
# (GGML_VK_VISIBLE_DEVICES / VK_LOADER_DEVICE_SELECT) so the SAME script
# drives either GPU:
#   - 780M iGPU (default): Vulkan0, BDF 0000:c4:00.0
#   - 7700S dGPU:          Vulkan1, BDF 0000:03:00.0 (set both env vars
#     before calling, e.g. from dgpu_embed_ctl.py)
# Without VK_LOADER_DEVICE_SELECT the Vulkan loader still enumerates both
# render nodes at init even when GGML_VK_VISIBLE_DEVICES narrows compute,
# which wakes whichever GPU isn't targeted out of D3cold. Always set both.
#
# Args (positional):
#   1: model GGUF path                (required)
#   2: port                           (required)
#   3: pooling {none|mean|cls|last}   (required)
#   4: ctx-size                       (required)
#   5: batch-size                     (optional; default = ctx-size)
#   6: ubatch-size                    (optional; default = ctx-size)
#   7: parallel slots                 (optional; default = 1)
set -euo pipefail

MODEL="${1:?model GGUF path required}"
PORT="${2:?port required}"
POOL="${3:?pooling required}"
CTX="${4:?ctx-size required}"
BATCH="${5:-$CTX}"
UBATCH="${6:-$CTX}"
PARALLEL="${7:-1}"

# Locate the Vulkan llama-server: prefer $RAG_LLAMA_BIN, then LM Studio's
# bundled Vulkan build, then anything on PATH.
find_llama() {
  if [[ -n "${RAG_LLAMA_BIN:-}" && -x "$RAG_LLAMA_BIN" ]]; then
    echo "$RAG_LLAMA_BIN"; return
  fi
  local hit
  hit=$(ls -td ~/.lmstudio/extensions/backends/*vulkan*/llama-server 2>/dev/null | head -n1)
  if [[ -x "$hit" ]]; then echo "$hit"; return; fi
  command -v llama-server || true
}
BIN="$(find_llama)"
[[ -n "$BIN" && -x "$BIN" ]] || { echo "FATAL: no llama-server found (set RAG_LLAMA_BIN)" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "FATAL: model not found: $MODEL" >&2; exit 1; }

# Default to the 780M iGPU (Vulkan0) unless the caller already set both
# device-pin env vars (e.g. dgpu_embed_ctl.py targeting the 7700S).
export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-0}"
export VK_LOADER_DEVICE_SELECT="${VK_LOADER_DEVICE_SELECT:-0000:c4:00.0}"
# llama-server's --device Vulkan<N> must match GGML_VK_VISIBLE_DEVICES: once
# the loader is filtered to one device via the env vars above, that device is
# always re-enumerated as index 0 from llama-server's point of view.
DEVICE="Vulkan0"

echo "embed-launch: GGML_VK_VISIBLE_DEVICES=$GGML_VK_VISIBLE_DEVICES VK_LOADER_DEVICE_SELECT=$VK_LOADER_DEVICE_SELECT" >&2
echo "embed-launch: $BIN --model $MODEL --port $PORT --device $DEVICE --pooling $POOL --ctx-size $CTX --batch-size $BATCH --ubatch-size $UBATCH --parallel $PARALLEL --embedding -ngl 99" >&2

# Redirect both stdout/stderr to a log file so the caller can inspect startup.
LOG="/tmp/embed_server_${PORT}.log"
exec "$BIN" \
  --model "$MODEL" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --device "$DEVICE" \
  --pooling "$POOL" \
  --ctx-size "$CTX" \
  --batch-size "$BATCH" \
  --ubatch-size "$UBATCH" \
  --parallel "$PARALLEL" \
  --embedding \
  --n-gpu-layers 99 \
  --flash-attn on \
  --poll 0 \
  --no-warmup \
  >"$LOG" 2>&1
