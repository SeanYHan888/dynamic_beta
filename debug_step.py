import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from torch.optim import AdamW
import numpy as np
import logging

try:
    from src.dynamic_dpo.data import build_train_val
    from src.dynamic_dpo.modeling import compute_batch_log_prob, margin_compute, dpo_loss
except ImportError:
    # If running from root, adjust path
    import sys
    import os
    sys.path.append(os.getcwd())
    from src.dynamic_dpo.data import build_train_val
    from src.dynamic_dpo.modeling import compute_batch_log_prob, margin_compute, dpo_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEBUG_STEP")

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

def debug_step():
    print("=== STARTING DEBUG STEP ===")
    
    # 1. Load Config
    with open("config_dpo.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config. LR: {config['dpo_training']['learning_rate']}, BetaMin: {config['beta_update'].get('beta_min', 'N/A')}")

    # 2. Load Model & Tokenizer
    model_name = config['policy_name']
    print(f"Loading model: {model_name}...")
    try:
        policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        print("WARNING: Could not load actual model. Falling back to gpt2 for logic check.")
        policy = AutoModelForCausalLM.from_pretrained("gpt2")
        ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        policy.config.pad_token_id = tokenizer.eos_token_id
        ref_model.config.pad_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    policy.to(device)
    ref_model.to(device)
    ref_model.eval()
    policy.train()

    # 3. Load Data
    print("Building dataloader...")
    # Mock distributed args
    config["distributed"] = {"world_size": 1, "rank": 0}
    # Reduce batch size for debug
    config['dpo_training']['batch_size'] = 2
    
    train_loader, _, _, _ = build_train_val(config, tokenizer)
    batch = next(iter(train_loader))
    batch = to_device(batch, device)
    print(f"Batch loaded. Keys: {batch.keys()}")
    
    # Check masks
    chosen_mask = batch['chosen_attention_mask']
    print(f"Chosen Mask Sum: {chosen_mask.sum().item()} / {chosen_mask.numel()}")
    if chosen_mask.sum() == 0:
        print("CRITICAL: Input Mask is all zeros!")

    # 4. Forward Pass
    print("Running forward pass...")
    policy_chosen, policy_rejected, ref_chosen, ref_rejected = compute_batch_log_prob(batch, policy, ref_model)
    
    print(f"Policy Cho Mean: {policy_chosen.mean().item():.4f}")
    print(f"Ref Cho Mean:    {ref_chosen.mean().item():.4f}")
    
    # 5. Compute Margins
    margins = margin_compute(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"Margins (Policy - Ref): Mean={margins.mean().item():.6f}, Std={margins.std().item():.6f}")
    print(f"Margins Raw: {margins.detach().cpu().numpy()}")

    # 6. Compute Loss
    beta = float(config['beta_update'].get('beta_0', 0.1))
    print(f"Using Beta: {beta}")
    
    loss, rewards_chosen, rewards_rejected = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    print(f"Loss: {loss.mean().item():.6f}")
    print(f"Rewards Chosen: {rewards_chosen.mean().item():.6f}")
    
    # 7. Backward Pass
    print("Running backward pass...")
    optimizer = AdamW(policy.parameters(), lr=1e-5)
    loss.mean().backward()
    
    # 8. Check Gradients
    total_norm = 0.0
    has_grad = False
    for p in policy.parameters():
        if p.grad is not None:
            has_grad = True
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"\n=== GRADIENT CHECK ===")
    print(f"Total Gradient Norm: {total_norm:.6f}")
    if total_norm == 0.0:
        print("FAIL: Gradients are ZERO.")
    elif not has_grad:
        print("FAIL: No gradients found.")
    else:
        print("SUCCESS: Gradients exist.")

if __name__ == "__main__":
    debug_step()
