# Test Pipeline Refactoring Summary

## What Was Done

Refactored the test pipeline from 3 separate scripts into a unified, single-command testing system.

### Files Created

1. **`run_test_pipeline.py`** - Main unified pipeline script
   - Combines all 4 testing steps into one script
   - Supports flexible execution (skip steps, resume, etc.)
   - Clean progress tracking and error handling

2. **`test.sh`** - Convenient shell wrapper
   - Even simpler interface for common use cases
   - Quick test mode (`--quick`)
   - Auto-checks for OpenAI API key

3. **Documentation**
   - `README.md` - Complete documentation
   - `MIGRATION.md` - Before/after comparison
   - `../TESTING.md` - Quick reference guide
   - `SUMMARY.md` - This file

### Configuration Updates

- Updated `config_dpo.yaml` with judge settings:
  - `test.judge_model` - GPT model to use
  - `test.summary_output` - Where to save results

## Pipeline Steps

The unified pipeline runs 4 steps automatically:

1. **Generate Responses** (Step 1)
   - Loads ref, dpo, and dynamic_dpo models
   - Samples test data from HH-RLHF
   - Generates responses for all models
   - Saves to JSONL files

2. **Build Pairs** (Step 2)
   - Creates A/B comparison pairs
   - Randomizes position to avoid bias
   - Generates one pair file per model

3. **Judge Pairs** (Step 3)
   - Uses GPT-4 to judge which response is better
   - Supports resume (skips already-judged items)
   - Handles API errors gracefully
   - Rate limiting support

4. **Summarize Results** (Step 4)
   - Computes win rates for each model
   - Calculates tie and invalid rates
   - Saves JSON summary
   - Prints formatted results

## Usage Comparison

### Before (Multiple Commands)
```bash
# Step 1: Generate (already done by gen_hh_answer.py)

# Step 2: Build pairs (3 times)
python test/01_build_pairs.py --in_jsonl eval_outputs/dpo_out.jsonl --out_pairs eval_outputs/dpo_pairs.jsonl --seed 42
python test/01_build_pairs.py --in_jsonl eval_outputs/dynamic_dpo_out.jsonl --out_pairs eval_outputs/dynamic_dpo_pairs.jsonl --seed 42
python test/01_build_pairs.py --in_jsonl eval_outputs/ref_out.jsonl --out_pairs eval_outputs/ref_pairs.jsonl --seed 42

# Step 3: Judge (3 times)
python test/02_gpt_judge.py --pairs eval_outputs/dpo_pairs.jsonl --out eval_outputs/dpo_judgments.jsonl --judge_model gpt-4.1
python test/02_gpt_judge.py --pairs eval_outputs/dynamic_dpo_pairs.jsonl --out eval_outputs/dynamic_dpo_judgments.jsonl --judge_model gpt-4.1
python test/02_gpt_judge.py --pairs eval_outputs/ref_pairs.jsonl --out eval_outputs/ref_judgments.jsonl --judge_model gpt-4.1

# Step 4: Summarize
python test/03_summarize_results.py --judgments eval_outputs/dpo_judgments.jsonl,eval_outputs/dynamic_dpo_judgments.jsonl,eval_outputs/ref_judgments.jsonl --out_json eval_outputs/summary.json
```

**Total: ~10 commands**

### After (Single Command)
```bash
./test/test.sh
```

**Total: 1 command**

## Key Features

### 1. Single Command Execution
Run the entire pipeline with one command

### 2. Step Skipping
- `--skip-generate` - Use existing model outputs
- `--skip-judge` - Just generate and build pairs

### 3. Resume Support
Automatically resumes interrupted judging from where it left off

### 4. Flexible Configuration
- Configure via YAML
- Override with CLI arguments
- Different judge models
- Rate limiting control

### 5. Progress Tracking
- Progress bars for generation
- Step-by-step status messages
- Clear section headers

### 6. Error Handling
- Graceful API error handling
- Invalid judgment detection
- Missing dependency warnings

## Output Structure

All outputs in `eval_outputs/`:

```
eval_outputs/
├── dpo_out.jsonl              # DPO model responses
├── dpo_pairs.jsonl            # DPO comparison pairs
├── dpo_judgments.jsonl        # DPO GPT-4 judgments
├── dynamic_dpo_out.jsonl      # Dynamic DPO responses
├── dynamic_dpo_pairs.jsonl    # Dynamic DPO pairs
├── dynamic_dpo_judgments.jsonl # Dynamic DPO judgments
├── ref_out.jsonl              # Reference model responses
├── ref_pairs.jsonl            # Reference pairs
├── ref_judgments.jsonl        # Reference judgments
└── summary.json               # Final statistics
```

## Example Results Format

```json
{
  "dpo": {
    "total": 200,
    "valid": 180,
    "win": 120,
    "loss": 60,
    "tie": 15,
    "invalid": 5,
    "win_rate": 0.6667,
    "tie_rate": 0.075,
    "invalid_rate": 0.025
  },
  "dynamic_dpo": {
    "total": 200,
    "valid": 185,
    "win": 140,
    "loss": 45,
    "tie": 12,
    "invalid": 3,
    "win_rate": 0.7568,
    "tie_rate": 0.06,
    "invalid_rate": 0.015
  },
  "ref": {
    "total": 200,
    "valid": 175,
    "win": 85,
    "loss": 90,
    "tie": 20,
    "invalid": 5,
    "win_rate": 0.4857,
    "tie_rate": 0.1,
    "invalid_rate": 0.025
  }
}
```

## Backward Compatibility

Old scripts still exist and work:
- `gen_hh_answer.py` - Original generation script
- `01_build_pairs.py` - Original pair builder
- `02_gpt_judge.py` - Original judging script
- `03_summarize_results.py` - Original summarizer

These are **deprecated** but maintained for compatibility.

## Dependencies

- PyTorch
- Transformers
- Datasets
- PyYAML
- tqdm
- openai (optional, for judging step)

## Testing

Script has been syntax-checked and help output verified:
```bash
$ uv run python test/run_test_pipeline.py --help
# Shows complete help with all options
```

## Future Improvements

Potential enhancements:
1. Support for additional judge models (Claude, etc.)
2. Parallel judging (concurrent API calls)
3. Visualization of results (charts, graphs)
4. Streaming output during generation
5. Docker container for reproducibility
6. CI/CD integration

## Notes

- Uses existing `split_prompt_and_response` from `dataset_process_hh.py`
- Maintains same JSONL format as original scripts
- Compatible with existing workflows
- No changes to config structure (only additions)
