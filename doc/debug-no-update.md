# Diagnostics: Why Loss/Reward Isn't Updating

If your model seems unchanged after training (loss is flat, or evaluation metrics match the start), here are the top 3 potential causes based on your current configuration.

## 1. Learning Rate is Too Low (Most Likely)

**Current Config:** `learning_rate: 5e-7`
**Diagnosis:** This is extremely conservative. For a 1B model on DPO, specificially with only 1 epoch (~900 steps), a learning rate of `5e-7` often results in changes so small they look like zero.
**Recommended Fix:**

- Increase to `5e-6` or `1e-5`.
- `5e-7` is usually for fine-tuning 70B models or very sensitive stabilization phases, not initial DPO adaptation.

## 2. Beta Collapse (Specific to Dynamic DPO)

**Current Config:** `beta_min: 0.0`
**Diagnosis:** If the Dynamic Beta logic drives `beta` down to 0, the gradients effectively vanish.

- The gradient of DPO loss scales with $\beta$.
- If $\beta \approx 0$, then $\nabla Loss \approx 0$.
**Recommended Fix:**
- Set `beta_min: 0.001` or `0.01` in `config_dpo.yaml` to prevent total collapse.
- Check your `risk_test_and_beta_log.jsonl`. If you see `beta` values like `0.0` or `1e-9`, this is the culprit.

## 3. FSDP Saving Logic (If "Not Updating" means Saved Model is Base Model)

**Current Logic:**

```python
state_dict = accelerator.get_state_dict(policy)
unwrapped = accelerator.unwrap_model(policy)
unwrapped.save_pretrained(save_dir, state_dict=state_dict)
```

**Diagnosis:** This logic is generally correct for Accelerate. However, if `offload_to_cpu` didn't gather all shards correctly (e.g. OOM during gather), you might be saving an empty or partial state.
**Verification:**

- Check if the `pbar` loss was decreasing during training.
- If `loss` went down in logs but the saved model is bad, it's a Saving Issue.
- If `loss` stayed flat in logs, it's a Learning Rate issue.

---

## Suggested Immediate Actions

1. **Check Logs:** Look at `wandb` or `tqdm` output.
   - **Flat Loss?** -> Increase LR.
   - **Loss Decreased?** -> Debug Saving.

2. **Modify Config (`config_dpo.yaml`):**

```yaml
dpo_training:
  learning_rate: 5e-6    # INCREASE THIS (was 5e-7)
  ...

beta_update:
  beta_min: 0.01         # INCREASE THIS (was 0.0)
```
