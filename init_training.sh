#!/bin/bash

# Exit on error
set -e

usage() {
    cat <<EOF
Usage: $0 [--config PATH] [--mode dynamic|static] [--no-auto-shutdown] [--] [accelerate launch args]

Examples:
  $0 --config config_dpo.yaml --mode dynamic
  $0 -- --num_processes 2 --mixed_precision bf16
EOF
}

CONFIG_PATH="config_dpo.yaml"
MODE="dynamic"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
EXTRA_ACCELERATE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --config requires a path."
                exit 1
            fi
            CONFIG_PATH="$2"
            shift 2
            ;;
        --mode)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --mode requires a value (dynamic or static)."
                exit 1
            fi
            MODE="$2"
            shift 2
            ;;
        --no-auto-shutdown|--no-shutdown)
            AUTO_SHUTDOWN=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ACCELERATE_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ACCELERATE_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$MODE" != "dynamic" && "$MODE" != "static" ]]; then
    echo "Error: --mode must be 'dynamic' or 'static'."
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Auto-shutdown logic
POD_ID="${RUNPOD_POD_ID:-}"
cleanup() {
    if [[ -n "$POD_ID" && "$AUTO_SHUTDOWN" == "true" ]]; then
        echo "[cleanup] Training finished (or script exited). Stopping pod $POD_ID..."
        runpodctl stop pod "$POD_ID" || true
    else
        echo "[cleanup] Skipping auto-shutdown (POD_ID unset or AUTO_SHUTDOWN=false)."
    fi
}
trap cleanup EXIT

echo "Starting training initialization..."

# 1. Create venv using uv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv' using uv..."
    uv venv
else
    echo "Virtual environment '.venv' already exists."
fi

# 2. Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install dependencies using uv
echo "Installing dependencies..."
# If uv.lock exists, use sync (fastest and most reliable)
if [ -f "uv.lock" ]; then
    echo "Found uv.lock. Syncing environment..."
    uv sync || { echo "Error: uv sync failed."; AUTO_SHUTDOWN=false; exit 1; }
# If pyproject.toml exists but no lock, install from it
elif [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml. Installing..."
    uv pip install . || { echo "Error: uv pip install . failed."; AUTO_SHUTDOWN=false; exit 1; }
# Fallback to requirements.txt
elif [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing..."
    uv pip install -r requirements.txt || { echo "Error: uv pip install requirements.txt failed."; AUTO_SHUTDOWN=false; exit 1; }
else
    echo "Error: No dependency file found (uv.lock, pyproject.toml, or requirements.txt)."
    AUTO_SHUTDOWN=false
    exit 1
fi

# 4. HuggingFace Login
echo "Logging in to Hugging Face..."
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please export HF_TOKEN='your_token_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

# 5. WandB Login
echo "Logging in to WandB..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable is not set."
    echo "Please export WANDB_API_KEY='your_key_here' before running this script."
    AUTO_SHUTDOWN=false
    exit 1
fi
wandb login "$WANDB_API_KEY"

# 6. Run Training Scripts
echo "Launching training with Accelerate..."
accelerate launch "${EXTRA_ACCELERATE_ARGS[@]}" -m dynamic_dpo.train --config "$CONFIG_PATH" --mode "$MODE"

echo "Training initialization sequence completed."
