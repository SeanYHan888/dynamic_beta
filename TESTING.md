# Quick Testing Guide

## Run Complete Test Pipeline (One Command)

**Option 1: Using the shell script (easiest)**
```bash
./test/test.sh
```

**Option 2: Using Python directly**
```bash
uv run python test/run_test_pipeline.py --config config_dpo.yaml
```

This single command will:
1. ✅ Generate responses from all models (ref, dpo, dynamic_dpo)
2. ✅ Build A/B comparison pairs
3. ✅ Judge pairs using GPT-4
4. ✅ Summarize results and compute win rates

## Common Commands

### Full pipeline
```bash
./test/test.sh
```

### Skip generation (use existing outputs)
```bash
./test/test.sh --skip-generate
```

### Skip judging (just generate and build pairs)
```bash
./test/test.sh --skip-judge
```

### Quick test (50 examples, use existing outputs)
```bash
./test/test.sh --quick
```

### Use different judge model
```bash
./test/test.sh --judge-model gpt-4o
```

### Run only DPO model
```bash
./test/test.sh --models dpo
```

### Run only Dynamic DPO model
```bash
./test/test.sh --models dynamic_dpo
```

### Run both DPO models (skip reference)
```bash
./test/test.sh --models dpo,dynamic_dpo
```

### Use custom model checkpoint
```bash
# Test a specific DPO checkpoint
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_10

# Test a specific Dynamic DPO checkpoint
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dynamic_dpo \
  --dynamic-dpo-model ./my_models/dynamic_dpo_best
```

## Quick Setup

1. **Set OpenAI API Key** (for judging step):
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

2. **Install dependencies**:
   ```bash
   pip install openai
   ```

3. **Run the pipeline**:
   ```bash
   ./test/test.sh
   ```

## Output Location

All results are saved in `eval_outputs/`:
- `summary.json` - Final win rates and statistics
- `*_out.jsonl` - Generated responses
- `*_pairs.jsonl` - A/B comparison pairs
- `*_judgments.jsonl` - GPT-4 judgments

## View Results

```bash
cat eval_outputs/summary.json
```

Or check the console output for a formatted summary.

## More Details

See [test/README.md](test/README.md) for complete documentation.
