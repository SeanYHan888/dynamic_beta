Dynamic DPO training with optional FSDP via Hugging Face Accelerate.

Quick start

- Edit `config_dpo.yaml` (models, dataset, and FSDP settings).
- Configure Accelerate once if you have not: `accelerate config`.
- Launch training:

```bash
accelerate launch -m dynamic_dpo.train --config config_dpo.yaml --mode dynamic
```

FSDP notes

- FSDP is controlled by `fsdp.enabled` in `config_dpo.yaml`.
- `fsdp.auto_wrap_layers` should match your model’s transformer block class name (for Llama: `LlamaDecoderLayer`).
- Full state dict checkpoints are saved on rank 0 to `dpo_training.save_dir`.
