# Example Pipeline Output

This is what you'll see when running the test pipeline.

## Running the Full Pipeline

```bash
$ ./test/test.sh

Running test pipeline...
Config: /home/feng/github/dynamic_beta/config_dpo.yaml

================================================================================
STEP 1: Generating Model Responses
================================================================================

Loading tokenizer from meta-llama/Llama-3.2-1B...
Loading DPO model from dpo_model...
Loading Dynamic DPO model from dynamic_dpo_model...
Loading reference model from meta-llama/Llama-3.2-1B...

Loading test dataset: Anthropic/hh-rlhf (test)...
Sampled 200 test examples

Generating from DPO model...
Generating: 100%|████████████████████████████| 50/50 [02:15<00:00,  2.70s/it]

Generating from Dynamic DPO model...
Generating: 100%|████████████████████████████| 50/50 [02:18<00:00,  2.77s/it]

Generating from Reference model...
Generating: 100%|████████████████████████████| 50/50 [02:12<00:00,  2.65s/it]

✓ Saved DPO outputs: eval_outputs/dpo_out.jsonl
✓ Saved Dynamic DPO outputs: eval_outputs/dynamic_dpo_out.jsonl
✓ Saved Reference outputs: eval_outputs/ref_out.jsonl

================================================================================
STEP 2: Building Comparison Pairs
================================================================================

✓ Built 200 pairs for dpo: eval_outputs/dpo_pairs.jsonl
✓ Built 200 pairs for dynamic_dpo: eval_outputs/dynamic_dpo_pairs.jsonl
✓ Built 200 pairs for ref: eval_outputs/ref_pairs.jsonl

================================================================================
STEP 3: Judging with GPT-4
================================================================================

Judging dpo pairs...
✓ Judged 200 new pairs for dpo: eval_outputs/dpo_judgments.jsonl

Judging dynamic_dpo pairs...
✓ Judged 200 new pairs for dynamic_dpo: eval_outputs/dynamic_dpo_judgments.jsonl

Judging ref pairs...
✓ Judged 200 new pairs for ref: eval_outputs/ref_judgments.jsonl

================================================================================
STEP 4: Summarizing Results
================================================================================

=== Win Rate (wins / (wins + losses)) ===
dpo              win_rate=0.6234 (win=96, loss=58, valid=154)  tie=41 invalid=5 total=200
dynamic_dpo      win_rate=0.7297 (win=108, loss=40, valid=148)  tie=47 invalid=5 total=200
ref              win_rate=0.4722 (win=85, loss=95, valid=180)  tie=15 invalid=5 total=200

✓ Saved summary: eval_outputs/summary.json

================================================================================
PIPELINE COMPLETE
================================================================================

All outputs saved in: eval_outputs/
Summary JSON: eval_outputs/summary.json
```

## Running with --skip-generate

```bash
$ ./test/test.sh --skip-generate

Running test pipeline...
Config: /home/feng/github/dynamic_beta/config_dpo.yaml

Skipping generation step (using existing outputs)

================================================================================
STEP 2: Building Comparison Pairs
================================================================================

✓ Built 200 pairs for dpo: eval_outputs/dpo_pairs.jsonl
✓ Built 200 pairs for dynamic_dpo: eval_outputs/dynamic_dpo_pairs.jsonl
✓ Built 200 pairs for ref: eval_outputs/ref_pairs.jsonl

================================================================================
STEP 3: Judging with GPT-4
================================================================================

Judging dpo pairs...
✓ Judged 0 new pairs for dpo: eval_outputs/dpo_judgments.jsonl

Judging dynamic_dpo pairs...
✓ Judged 0 new pairs for dynamic_dpo: eval_outputs/dynamic_dpo_judgments.jsonl

Judging ref pairs...
✓ Judged 0 new pairs for ref: eval_outputs/ref_judgments.jsonl

================================================================================
STEP 4: Summarizing Results
================================================================================

=== Win Rate (wins / (wins + losses)) ===
dpo              win_rate=0.6234 (win=96, loss=58, valid=154)  tie=41 invalid=5 total=200
dynamic_dpo      win_rate=0.7297 (win=108, loss=40, valid=148)  tie=47 invalid=5 total=200
ref              win_rate=0.4722 (win=85, loss=95, valid=180)  tie=15 invalid=5 total=200

✓ Saved summary: eval_outputs/summary.json

================================================================================
PIPELINE COMPLETE
================================================================================

All outputs saved in: eval_outputs/
Summary JSON: eval_outputs/summary.json
```

## Running with Custom Model Path

```bash
$ uv run python test/run_test_pipeline.py \
  --config config_dpo.yaml \
  --models dpo \
  --dpo-model ./checkpoints/dpo_epoch_5

Loaded config from: config_dpo.yaml
Models to evaluate: ['dpo']
Using device: cuda

================================================================================
STEP 1: Generating Model Responses (dpo)
================================================================================
Using custom path for dpo: ./checkpoints/dpo_epoch_5

Loading tokenizer from meta-llama/Llama-3.2-1B...
Loading test dataset: Anthropic/hh-rlhf (test)...
Sampled 200 test examples

Loading dpo model from ./checkpoints/dpo_epoch_5...
Generating from dpo model...
Generating: 100%|████████████████████████████| 50/50 [02:15<00:00,  2.70s/it]
✓ Saved dpo outputs: eval_outputs/dpo_out.jsonl

================================================================================
STEP 2: Building Comparison Pairs (dpo)
================================================================================

✓ Built 200 pairs for dpo: eval_outputs/dpo_pairs.jsonl

...
```

## Running Quick Test

```bash
$ ./test/test.sh --quick

Running test pipeline...
Config: /home/feng/github/dynamic_beta/config_dpo.yaml

Skipping generation step (using existing outputs)

================================================================================
STEP 2: Building Comparison Pairs
================================================================================

✓ Built 200 pairs for dpo: eval_outputs/dpo_pairs.jsonl
✓ Built 200 pairs for dynamic_dpo: eval_outputs/dynamic_dpo_pairs.jsonl
✓ Built 200 pairs for ref: eval_outputs/ref_pairs.jsonl

================================================================================
STEP 3: Judging with GPT-4
================================================================================

Judging dpo pairs...
✓ Judged 50 new pairs for dpo: eval_outputs/dpo_judgments.jsonl

Judging dynamic_dpo pairs...
✓ Judged 50 new pairs for dynamic_dpo: eval_outputs/dynamic_dpo_pairs.jsonl

Judging ref pairs...
✓ Judged 50 new pairs for ref: eval_outputs/ref_judgments.jsonl

================================================================================
STEP 4: Summarizing Results
================================================================================

=== Win Rate (wins / (wins + losses)) ===
dpo              win_rate=0.6190 (win=26, loss=16, valid=42)  tie=7 invalid=1 total=50
dynamic_dpo      win_rate=0.7391 (win=34, loss=12, valid=46)  tie=3 invalid=1 total=50
ref              win_rate=0.4783 (win=22, loss=24, valid=46)  tie=3 invalid=1 total=50

✓ Saved summary: eval_outputs/summary.json

================================================================================
PIPELINE COMPLETE
================================================================================

All outputs saved in: eval_outputs/
Summary JSON: eval_outputs/summary.json
```

## Error Handling Examples

### Missing OpenAI API Key

```bash
$ ./test/test.sh --skip-generate
Warning: OPENAI_API_KEY not set. Judging step may fail.
Set it with: export OPENAI_API_KEY=your-api-key

Running test pipeline...
...
```

### OpenAI Not Installed

```bash
$ uv run python test/run_test_pipeline.py --config config_dpo.yaml
Warning: OpenAI not installed. Install with: pip install openai

...

================================================================================
STEP 3: SKIPPED - OpenAI not installed
================================================================================
Install with: pip install openai
```

## Viewing Results

```bash
$ cat eval_outputs/summary.json
{
  "dpo": {
    "total": 200,
    "valid": 154,
    "win": 96,
    "loss": 58,
    "tie": 41,
    "invalid": 5,
    "win_rate": 0.6233766233766234,
    "tie_rate": 0.205,
    "invalid_rate": 0.025
  },
  "dynamic_dpo": {
    "total": 200,
    "valid": 148,
    "win": 108,
    "loss": 40,
    "tie": 47,
    "invalid": 5,
    "win_rate": 0.7297297297297297,
    "tie_rate": 0.235,
    "invalid_rate": 0.025
  },
  "ref": {
    "total": 200,
    "valid": 180,
    "win": 85,
    "loss": 95,
    "tie": 15,
    "invalid": 5,
    "win_rate": 0.47222222222222215,
    "tie_rate": 0.075,
    "invalid_rate": 0.025
  }
}
```

## Help Output

```bash
$ ./test/test.sh --help
Usage: ./test/test.sh [OPTIONS]

Quick wrapper for the unified test pipeline

Options:
  --skip-generate    Skip generation (use existing outputs)
  --skip-judge       Skip judging step
  --quick            Quick test (50 examples, skip generate)
  --help             Show this help

All other options are passed to run_test_pipeline.py

Examples:
  ./test/test.sh                           # Full pipeline
  ./test/test.sh --skip-generate           # Re-judge existing outputs
  ./test/test.sh --quick                   # Quick test
  ./test/test.sh --judge-model gpt-4o      # Use different judge model
```

```bash
$ uv run python test/run_test_pipeline.py --help
usage: run_test_pipeline.py [-h] --config CONFIG [--skip-generate]
                            [--skip-judge] [--judge-model JUDGE_MODEL]
                            [--max-judge-items MAX_JUDGE_ITEMS]
                            [--judge-sleep JUDGE_SLEEP]
                            [--summary-output SUMMARY_OUTPUT]

Unified testing pipeline for DPO models

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --skip-generate       Skip generation step (use existing outputs)
  --skip-judge          Skip judging step
  --judge-model JUDGE_MODEL
                        Judge model to use (default: gpt-4.1)
  --max-judge-items MAX_JUDGE_ITEMS
                        Max items to judge (0=all)
  --judge-sleep JUDGE_SLEEP
                        Sleep between judge API calls (seconds)
  --summary-output SUMMARY_OUTPUT
                        Path to save summary JSON
```
