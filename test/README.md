# Testing Pipeline

This directory contains a unified testing pipeline for evaluating DPO models against baseline models.

## Overview

The testing pipeline consists of 4 steps, all automated in a single script:

1. **Generate Responses** - Generate responses from ref, dpo, and dynamic_dpo models
2. **Build Pairs** - Create A/B comparison pairs with randomization to avoid position bias
3. **Judge Pairs** - Use GPT-4 to judge which response is better
4. **Summarize Results** - Compute win rates and statistics

## Quick Start

### Basic Usage

Run the complete pipeline with a single command:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml
```

This will:
- Generate responses from all 3 models (ref, dpo, dynamic_dpo)
- Build comparison pairs
- Judge pairs using GPT-4
- Summarize results and save to `eval_outputs/summary.json`

### Skip Generation (Use Existing Outputs)

If you already have generated outputs and just want to re-run judging:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --skip-generate
```

### Skip Judging (Just Generate and Build Pairs)

If you want to generate responses and pairs but skip the GPT-4 judging step:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --skip-judge
```

### Custom Judge Model

Use a different GPT model for judging:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --judge-model gpt-4o
```

### Run Specific Models Only

Evaluate only the DPO model:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --models dpo
```

Evaluate only the Dynamic DPO model:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --models dynamic_dpo
```

Evaluate both DPO models (skip reference):

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --models dpo,dynamic_dpo
```

Valid model options: `all`, `dpo`, `dynamic_dpo`, `ref`, or comma-separated combinations.

### Use Custom Model Paths

Override the model paths from config with custom checkpoint directories:

```bash
# Use a specific DPO checkpoint
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --dpo-model ./checkpoints/dpo_epoch_5

# Use a specific Dynamic DPO checkpoint
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --dynamic-dpo-model ./checkpoints/dynamic_dpo_step_1000

# Use a different reference model
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --ref-model meta-llama/Llama-3.2-3B

# Override multiple models
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --dpo-model ./dpo_v2 \
  --dynamic-dpo-model ./dynamic_dpo_v2 \
  --models dpo,dynamic_dpo
```

These flags override the paths specified in the config file, allowing you to test different checkpoints without modifying the config.

### Limit Number of Judgments

For testing, limit the number of items to judge:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --max-judge-items 50
```

### Add Sleep Between API Calls

To avoid rate limits, add a delay between judge API calls:

```bash
python test/run_test_pipeline.py --config config_dpo.yaml --judge-sleep 1.0
```

## Configuration

Edit `config_dpo.yaml` to configure the test pipeline:

```yaml
test:
  subset: test                          # Dataset split to use
  test_num: 200                         # Number of test examples
  seed: 42                              # Random seed
  max_new_tokens: 256                   # Max tokens to generate
  temperature: 0.7                      # Sampling temperature
  top_p: 0.9                           # Top-p sampling
  batch_size: 4                        # Generation batch size
  dpo_out_dir: eval_outputs/dpo_out.jsonl
  dynamic_dpo_out_dir: eval_outputs/dynamic_dpo_out.jsonl
  ref_out_dir: eval_outputs/ref_out.jsonl
  judge_model: gpt-4.1                 # GPT model for judging
  summary_output: eval_outputs/summary.json
```

## Output Files

The pipeline creates the following files in `eval_outputs/`:

### Generation Outputs
- `dpo_out.jsonl` - DPO model responses
- `dynamic_dpo_out.jsonl` - Dynamic DPO model responses
- `ref_out.jsonl` - Reference model responses

### Comparison Pairs
- `dpo_pairs.jsonl` - A/B pairs for DPO
- `dynamic_dpo_pairs.jsonl` - A/B pairs for Dynamic DPO
- `ref_pairs.jsonl` - A/B pairs for Reference

### Judgments
- `dpo_judgments.jsonl` - GPT-4 judgments for DPO
- `dynamic_dpo_judgments.jsonl` - GPT-4 judgments for Dynamic DPO
- `ref_judgments.jsonl` - GPT-4 judgments for Reference

### Summary
- `summary.json` - Final statistics and win rates

## Understanding Results

The summary JSON contains statistics for each model:

```json
{
  "dpo": {
    "total": 200,           // Total judgments
    "valid": 180,           // Valid A/B comparisons
    "win": 120,             // Model wins
    "loss": 60,             // Model losses
    "tie": 15,              // Ties
    "invalid": 5,           // Invalid comparisons
    "win_rate": 0.6667,     // win / (win + loss)
    "tie_rate": 0.075,      // tie / total
    "invalid_rate": 0.025   // invalid / total
  },
  ...
}
```

**Win Rate** = `wins / (wins + losses)` - Higher is better

This metric shows how often the model's response was preferred over the reference (chosen) response.

## Requirements

- PyTorch
- Transformers
- Datasets
- OpenAI Python client (for judging step)
- CUDA (recommended for generation)

Install OpenAI client for judging:
```bash
pip install openai
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-api-key-here
```

## Individual Scripts (Legacy)

The old pipeline scripts are still available for manual execution:

1. `gen_hh_answer.py` - Generate model responses
2. `01_build_pairs.py` - Build comparison pairs
3. `02_gpt_judge.py` - Judge pairs with GPT-4
4. `03_summarize_results.py` - Summarize results

These are now deprecated in favor of the unified `run_test_pipeline.py`.

## Troubleshooting

### CUDA Out of Memory

Reduce `test.batch_size` in the config:
```yaml
test:
  batch_size: 2  # Reduce from 4
```

### OpenAI Rate Limits

Add sleep between API calls:
```bash
python test/run_test_pipeline.py --config config_dpo.yaml --judge-sleep 2.0
```

### Resume Interrupted Judging

The judging step automatically resumes from where it left off. Just re-run the same command - it will skip already-judged items.

## Advanced Usage

### Custom Summary Output Path

```bash
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --summary-output my_results/summary.json
```

### Complete Example with All Options

```bash
python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --judge-model gpt-4o \
  --max-judge-items 100 \
  --judge-sleep 1.0 \
  --summary-output results/test_summary.json
```
