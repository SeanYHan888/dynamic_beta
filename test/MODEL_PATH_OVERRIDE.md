# Model Path Override Feature

This document explains how to use custom model paths with the test pipeline.

## Overview

The test pipeline now supports command-line arguments to override model paths from the config file. This is useful for:

- Testing different model checkpoints without editing the config
- Comparing multiple training runs
- Quick iteration during model development
- Testing specific epochs or steps

## Available Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--dpo-model` | Override DPO model path | `--dpo-model ./checkpoints/dpo_epoch_5` |
| `--dynamic-dpo-model` | Override Dynamic DPO model path | `--dynamic-dpo-model ./models/dynamic_dpo_best` |
| `--ref-model` | Override reference model path | `--ref-model meta-llama/Llama-3.2-3B` |

## Usage Examples

### Test a Specific DPO Checkpoint

```bash
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_10
```

This will:
1. Use `./checkpoints/dpo_epoch_10` instead of the path in config
2. Only evaluate the DPO model
3. Skip dynamic_dpo and ref models

### Compare Two Different DPO Checkpoints

```bash
# Test epoch 5
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_5 \
  --summary-output results/epoch_5.json

# Test epoch 10
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_10 \
  --summary-output results/epoch_10.json

# Compare results
cat results/epoch_5.json
cat results/epoch_10.json
```

### Test Dynamic DPO with Custom Checkpoint

```bash
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dynamic_dpo \
  --dynamic-dpo-model ./my_experiments/dynamic_dpo_lr_0.001
```

### Use Different Reference Model

```bash
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --ref-model meta-llama/Llama-3.2-3B
```

### Override Multiple Models

```bash
uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --dpo-model ./experiments/dpo_v2 \
  --dynamic-dpo-model ./experiments/dynamic_dpo_v2 \
  --models dpo,dynamic_dpo
```

## How It Works

1. **Config is loaded first**: The pipeline reads model paths from `config_dpo.yaml`
2. **Command-line args override**: If you provide `--dpo-model`, `--dynamic-dpo-model`, or `--ref-model`, those paths override the config
3. **Display confirmation**: The pipeline prints "Using custom path for {model}: {path}" when overrides are active
4. **Everything else stays the same**: Output paths, generation settings, etc. still come from config

## Example Output

When using a custom model path, you'll see:

```
================================================================================
STEP 1: Generating Model Responses (dpo)
================================================================================
Using custom path for dpo: ./checkpoints/dpo_epoch_5

Loading tokenizer from meta-llama/Llama-3.2-1B...
Loading test dataset: Anthropic/hh-rlhf (test)...
Sampled 200 test examples

Loading dpo model from ./checkpoints/dpo_epoch_5...
...
```

## Workflow for Model Development

### 1. During Training

Train your model and save checkpoints:
```bash
# Your training saves to: ./checkpoints/dpo_epoch_{1,2,3,...}
```

### 2. Quick Test Checkpoints

Test different epochs without editing config:
```bash
# Test epoch 3
./test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_3 \
  --max-judge-items 50

# Test epoch 5
./test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_5 \
  --max-judge-items 50
```

### 3. Full Evaluation of Best Checkpoint

Once you identify the best checkpoint:
```bash
./test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_5
```

Or update your config permanently:
```yaml
dpo_training:
  save_dpo_dir: checkpoints/dpo_epoch_5  # Update config
```

## Tips

1. **Use relative paths**: Paths are relative to where you run the script
2. **Combine with `--models`**: Use `--models dpo` to only test the model you're overriding
3. **Use `--skip-generate`**: If you already generated outputs, use `--skip-generate` to just re-judge
4. **Organize outputs**: Use `--summary-output` to save results to different files for comparison

## Error Handling

If you specify a model path that doesn't exist:
```bash
./test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --dpo-model ./nonexistent/path
```

The pipeline will fail with a clear error when trying to load the model:
```
OSError: ./nonexistent/path does not appear to be a valid model directory
```

## Config File vs Command Line

| Aspect | Config File | Command Line |
|--------|-------------|--------------|
| **Persistence** | Permanent until edited | Only for this run |
| **Best for** | Default/production models | Testing, experiments |
| **Precedence** | Lower | Higher (overrides config) |
| **Use case** | Stable model paths | Quick iterations, comparisons |
