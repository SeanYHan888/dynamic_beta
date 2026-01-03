#!/bin/bash
# Quick test script - wrapper around the unified test pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default config
CONFIG="${PROJECT_ROOT}/config_dpo.yaml"

# Parse arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Quick wrapper for the unified test pipeline"
            echo ""
            echo "Options:"
            echo "  --skip-generate    Skip generation (use existing outputs)"
            echo "  --skip-judge       Skip judging step"
            echo "  --quick            Quick test (50 examples, skip generate)"
            echo "  --models MODEL     Which models: 'all', 'dpo', 'dynamic_dpo', 'ref', or comma-separated"
            echo "  --help             Show this help"
            echo ""
            echo "All other options are passed to run_test_pipeline.py"
            echo ""
            echo "Examples:"
            echo "  $0                           # Full pipeline (all models)"
            echo "  $0 --models dpo              # Only evaluate DPO model"
            echo "  $0 --models dynamic_dpo      # Only evaluate Dynamic DPO model"
            echo "  $0 --models dpo,dynamic_dpo  # Evaluate both DPO models"
            echo "  $0 --skip-generate           # Re-judge existing outputs"
            echo "  $0 --quick                   # Quick test"
            echo "  $0 --judge-model gpt-4o      # Use different judge model"
            exit 0
            ;;
        --quick)
            ARGS+=("--skip-generate" "--max-judge-items" "50")
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if OpenAI API key is set (only if not skipping judge)
if [[ ! " ${ARGS[@]} " =~ " --skip-judge " ]]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "Warning: OPENAI_API_KEY not set. Judging step may fail."
        echo "Set it with: export OPENAI_API_KEY=your-api-key"
        echo ""
    fi
fi

# Run the pipeline
echo "Running test pipeline..."
echo "Config: $CONFIG"
echo ""

cd "$PROJECT_ROOT"
uv run python test/run_test_pipeline.py --config "$CONFIG" "${ARGS[@]}"
