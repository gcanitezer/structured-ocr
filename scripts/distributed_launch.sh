#!/usr/bin/env bash
# Distributed training launcher for the Structured OCR pipeline.
#
# This script wraps torchrun / accelerate launch with sensible defaults
# for the Structured OCR training scripts. Pass either ``sft`` or
# ``grpo`` as the first argument; remaining arguments are forwarded to
# the underlying Python script.
#
# Examples:
#
#   # Single-node, multi-GPU SFT
#   ./scripts/distributed_launch.sh sft --config configs/training_sft.yaml
#
#   # Multi-node GRPO (called on every node; --rdzv_endpoint should be a
#   # reachable master node)
#   ./scripts/distributed_launch.sh grpo \
#       --nnodes 2 --node_rank 0 \
#       --rdzv_endpoint master.example.com:29500 \
#       --config configs/training_grpo.yaml \
#       --model-name ./outputs/sft
#
#   # Use accelerate FSDP
#   ./scripts/distributed_launch.sh sft --launcher accelerate \
#       --accelerate-config configs/accelerate_fsdp.yaml \
#       --config configs/training_sft.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-sft}"
shift || true

LAUNCHER="torchrun"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-127.0.0.1:29500}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/deepspeed_zero2.json}"
EXTRA_TORCHRUN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launcher)
            LAUNCHER="$2"
            shift 2
            ;;
        --accelerate-config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --deepspeed-config)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --rdzv-endpoint)
            RDZV_ENDPOINT="$2"
            shift 2
            ;;
        *)
            EXTRA_TORCHRUN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$NPROC_PER_NODE" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    fi
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

case "$MODE" in
    sft)
        SCRIPT="$REPO_ROOT/scripts/train_sft.py"
        ;;
    grpo)
        SCRIPT="$REPO_ROOT/scripts/train_grpo.py"
        ;;
    *)
        echo "Unknown mode '$MODE' (expected 'sft' or 'grpo')" >&2
        exit 2
        ;;
esac

case "$LAUNCHER" in
    torchrun)
        TORCHRUN_ARGS=(
            --nnodes "$NNODES"
            --node_rank "$NODE_RANK"
            --rdzv_backend "$RDZV_BACKEND"
            --rdzv_endpoint "$RDZV_ENDPOINT"
        )
        if [[ -n "$NPROC_PER_NODE" ]]; then
            TORCHRUN_ARGS+=(--nproc_per_node "$NPROC_PER_NODE")
        fi
        echo "[distributed_launch] torchrun ${TORCHRUN_ARGS[*]} $SCRIPT ${EXTRA_TORCHRUN_ARGS[*]}" >&2
        exec torchrun "${TORCHRUN_ARGS[@]}" "$SCRIPT" "${EXTRA_TORCHRUN_ARGS[@]}"
        ;;
    accelerate)
        echo "[distributed_launch] accelerate launch --config_file $ACCELERATE_CONFIG $SCRIPT ${EXTRA_TORCHRUN_ARGS[*]}" >&2
        exec accelerate launch --config_file "$ACCELERATE_CONFIG" "$SCRIPT" "${EXTRA_TORCHRUN_ARGS[@]}"
        ;;
    python)
        echo "[distributed_launch] python $SCRIPT ${EXTRA_TORCHRUN_ARGS[*]}" >&2
        exec python "$SCRIPT" "${EXTRA_TORCHRUN_ARGS[@]}"
        ;;
    *)
        echo "Unknown launcher '$LAUNCHER' (expected 'torchrun', 'accelerate', or 'python')" >&2
        exit 2
        ;;
esac
