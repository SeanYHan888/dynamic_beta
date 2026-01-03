# Migration Guide: Old vs New Testing Pipeline

## Before (Multiple Steps)

Previously, you had to run 4 separate scripts:

```bash
# Step 1: Generate responses (3 separate runs)
python test/gen_hh_answer.py --config config_dpo.yaml

# Step 2: Build pairs (3 separate runs)
python test/01_build_pairs.py \
  --in_jsonl eval_outputs/dpo_out.jsonl \
  --out_pairs eval_outputs/dpo_pairs.jsonl \
  --seed 42

python test/01_build_pairs.py \
  --in_jsonl eval_outputs/dynamic_dpo_out.jsonl \
  --out_pairs eval_outputs/dynamic_dpo_pairs.jsonl \
  --seed 42

python test/01_build_pairs.py \
  --in_jsonl eval_outputs/ref_out.jsonl \
  --out_pairs eval_outputs/ref_pairs.jsonl \
  --seed 42

# Step 3: Judge pairs (3 separate runs)
python test/02_gpt_judge.py \
  --pairs eval_outputs/dpo_pairs.jsonl \
  --out eval_outputs/dpo_judgments.jsonl \
  --judge_model gpt-4.1

python test/02_gpt_judge.py \
  --pairs eval_outputs/dynamic_dpo_pairs.jsonl \
  --out eval_outputs/dynamic_dpo_judgments.jsonl \
  --judge_model gpt-4.1

python test/02_gpt_judge.py \
  --pairs eval_outputs/ref_pairs.jsonl \
  --out eval_outputs/ref_judgments.jsonl \
  --judge_model gpt-4.1

# Step 4: Summarize
python test/03_summarize_results.py \
  --judgments eval_outputs/dpo_judgments.jsonl,eval_outputs/dynamic_dpo_judgments.jsonl,eval_outputs/ref_judgments.jsonl \
  --out_json eval_outputs/summary.json
```

**Total: 10 separate commands!**

## After (Single Command)

Now you can run everything with one command:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml
```

**Total: 1 command!**

## Key Improvements

### 1. **Single Command**
- Old: 10 separate commands
- New: 1 unified command

### 2. **Automatic Flow**
- Old: Manual coordination between steps
- New: Automatic pipeline with clear progress indicators

### 3. **Configuration-Based**
- Old: Hardcoded paths and parameters
- New: Everything configured in YAML

### 4. **Resume Support**
- Old: No automatic resume
- New: Automatically resumes interrupted judging

### 5. **Better Progress Tracking**
- Old: Minimal feedback
- New: Progress bars and step-by-step status

### 6. **Error Handling**
- Old: Manual error checking
- New: Automatic error handling with graceful fallbacks

### 7. **Flexible Execution**
- Old: All-or-nothing
- New: Skip steps with `--skip-generate` or `--skip-judge`

## Backward Compatibility

The old scripts still exist and work:
- `gen_hh_answer.py`
- `01_build_pairs.py`
- `02_gpt_judge.py`
- `03_summarize_results.py`

But they are now **deprecated** in favor of the unified pipeline.

## Common Migration Scenarios

### Scenario 1: Full Test Run
**Old:**
```bash
# Run 10 separate commands...
```

**New:**
```bash
python test/run_test_pipeline.py --config config_dpo.yaml
```

### Scenario 2: Re-judge Existing Outputs
**Old:**
```bash
# Manually run 01_build_pairs.py for each model
# Then manually run 02_gpt_judge.py for each model
# Then run 03_summarize_results.py
```

**New:**
```bash
python test/run_test_pipeline.py --config config_dpo.yaml --skip-generate
```

### Scenario 3: Test with Different Judge Model
**Old:**
```bash
# Re-run 02_gpt_judge.py 3 times with --judge_model parameter
# Then run 03_summarize_results.py
```

**New:**
```bash
python test/run_test_pipeline.py --config config_dpo.yaml \
  --skip-generate --judge-model gpt-4o
```

### Scenario 4: Quick Test (Limited Examples)
**Old:**
```bash
# Edit config, run all commands, restore config
```

**New:**
```bash
python test/run_test_pipeline.py --config config_dpo.yaml --max-judge-items 50
```

## What's the Same

- Output file formats (JSONL)
- Output file locations (`eval_outputs/`)
- Judgment logic (same GPT-4 prompts)
- Summary statistics (same calculations)

## What's Different

- **Everything is automated** - No manual coordination needed
- **Better defaults** - Judge model and other settings in config
- **Progress tracking** - Clear feedback on what's happening
- **Resume support** - Can restart without losing work
