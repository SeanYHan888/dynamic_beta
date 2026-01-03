import torch
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import os
import json
import logging
from typing import Dict, Any

try:
    from .data import build_train_val
    from .modeling import (
        dpo_loss,
        margin_compute,
        compute_and_log_model_margin,
        compute_batch_log_prob,
        empirical_over_threshold_proportion,
        risk_test,
        update_beta,
        WarmupQuantileAccumulator,
        EMAUpdate,
    )
except ImportError:  # Allows running as a script: uv run src/dynamic_dpo/train.py
    import os
    import sys

    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    from data import build_train_val
    from modeling import (
        dpo_loss,
        margin_compute,
        compute_and_log_model_margin,
        compute_batch_log_prob,
        empirical_over_threshold_proportion,
        risk_test,
        update_beta,
        WarmupQuantileAccumulator,
        EMAUpdate,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def to_device_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(policy, ref_model, val_loader, beta, device):
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    total_loss = 0.0
    total_count = 0
    sum_chosen_rewards = 0.0
    sum_rejected_rewards = 0.0
    correct = 0  

    pbar = tqdm(val_loader, desc="Evaluating", leave=False, dynamic_ncols=True)

    for batch in pbar:
        batch = to_device_batch(batch, device)

        policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(
            batch, policy=policy, ref_model=ref_model
        )

        loss_vec, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_log_prob,
            policy_rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
            beta
        )

        batches = loss_vec.shape[0]
        total_loss += loss_vec.mean().item() * batches
        total_count += batches
        sum_chosen_rewards += chosen_rewards.mean().item() * batches
        sum_rejected_rewards += rejected_rewards.mean().item() * batches
        correct += (chosen_rewards > rejected_rewards).sum().item()
        
    metrics = {
        "eval_loss": total_loss / max(1, total_count),
        "eval_chosen_rewards": sum_chosen_rewards / max(1, total_count),
        "eval_rejected_rewards": sum_rejected_rewards / max(1, total_count),
        "eval_reward_accuracy": correct / max(1, total_count),
    }
            
    if policy_was_training:
        policy.train()

    return metrics

def train(config_path: str, mode: str = "dynamic"):
    """
    Main training function.
    mode: 'dynamic' or 'static'
    """
    config = load_yaml_config(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed_everything(config['dataset'].get('seed', 42))

    wandb.init(
        project=config.get('wandb_project', 'handwritten-dpo'),
        name=config.get('run_name', 'run'),
        config=config
    )

    # Load models
    policy_name = config['policy_name']
    ref_name = config['ref_name']
    policy = AutoModelForCausalLM.from_pretrained(policy_name).to(device)
    tok = AutoTokenizer.from_pretrained(policy_name)
    policy.config.pad_token_id = tok.pad_token_id

    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    
    train_loader, val_loader = build_train_val(config=config, tokenizer=tok)

    policy.train()
    ref_model.eval()

    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    use_bf16 = config['precision'] == 'bf16'
    if use_bf16:
        policy.to(dtype=torch.bfloat16)
        ref_model.to(dtype=torch.bfloat16)

    # Logging setup
    # Determine log dir based on mode or config
    if mode == 'dynamic':
         LOG_DIR = config['margin_log'].get('log_dir', 'logs/margins')
    else:
         LOG_DIR = config['margin_log'].get('dpo_log_dir', 'logs/dpo_margins')
         
    os.makedirs(LOG_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")

    # Dynamic DPO specific setup
    is_dynamic = (mode == 'dynamic')
    beta = float(config.get('dpo_training', {}).get('beta', 0.1)) # default static beta

    if is_dynamic:
        # risk test parameter
        delta = float(config['risk_test']['delta'])
        eplison_0 = float(config['risk_test']['eplison_0'])
        momentum = float(config['risk_test']['lambda'])
        q = 1.0 - delta
        threshold_accumulator = WarmupQuantileAccumulator(q=q)
        
        # beta update
        gamma = float(config['beta_update']['gamma'])
        alpha = float(config['beta_update']['alpha'])
        beta_0 = float(config['beta_update']['beta_0'])
        beta_max = float(config['beta_update']['beta_max'])
        beta_min = float(config['beta_update']['beta_min'])
        
        beta = beta_0 # start with beta_0
        
        risk_stat = {"total": 0, "fail": 0}
        log_f = open("risk_test_and_beta_log.jsonl", "w", encoding="utf-8")
        warmup_steps = int(config['dpo_training']['warmup_steps'])
        warmup_done = False
        warmup_count = 0
        ema = None
    else:
        # Static DPO
        # For static, we might still have warmup for LR, but not for beta
        warmup_steps = int(config['dpo_training'].get('warmup_steps', 0)) # Might be used for scheduler, but not beta
    
    epochs = config['dpo_training']['epochs']
    log_steps = config['dpo_training']['log_steps']
    global_steps = 0
    max_grad_norm = float(config['dpo_training']['max_grad_norm'])

    for epoch in range(epochs):
        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train | epoch {epoch+1}/{epochs}", dynamic_ncols=True, leave=False)
        
        running_loss = 0.0
        
        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16): 
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(
                    batch, policy=policy, ref_model=ref_model
                )

                model_margin = margin_compute(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob
                )

                # Dynamic DPO Logic
                beta_used = beta
                if is_dynamic:
                    if not warmup_done:
                        threshold_accumulator.update(model_margin)
                        warmup_count += 1
                        if warmup_count == warmup_steps:
                            tau_0 = threshold_accumulator.finalize()
                            ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                            warmup_done = True
                            log_f.write(json.dumps({
                                "type": "warmup_end",
                                "tau_0": float(tau_0),
                                "beta_0": float(beta)
                            }) + "\n")
                            log_f.flush()
                        beta_used = beta_0
                    else:
                        tau = ema.update_tau(model_margin)
                        num_margin = int(model_margin.numel())
                        p_hat = empirical_over_threshold_proportion(model_margin, tau)
                        is_over_risk, eplison, delta_prime = risk_test(p_hat=p_hat, eplison_0=eplison_0, delta=delta, n=num_margin)

                        beta, u_k, s_k, alpha_used = update_beta(
                            beta, p_hat, delta_prime, eplison, alpha, gamma, beta_min, beta_max
                        )
                        beta_used = beta

                        risk_stat["total"] += 1
                        if is_over_risk:
                            risk_stat["fail"] += 1

                        log_f.write(json.dumps({
                            "step": int(global_steps),
                            "tau": float(tau),
                            "p_hat": float(p_hat),
                            "risk_over": bool(is_over_risk),
                            "beta": float(beta),
                            "u_k": float(u_k),
                            "s_k": float(s_k),
                            "alpha": float(alpha_used),
                        }) + "\n")
                        log_f.flush()

                compute_and_log_model_margin(
                    model_margin=model_margin, epoch_dir=epoch_dir, epoch=epoch, step=step, JSONL_PATH=JSONL_PATH
                )

                loss_raw, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta=beta_used
                )
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()

            optimizer.zero_grad()
            loss.backward()
            
            # Dynamic often clips grad only after warmup, but static usually always clips. 
            # In original training.py (dynamic), clip is always on. 
            # Wait, original training.py: lines 205-211 had `if global_steps >= warmup_steps:` wrapper, but that was probably for stepping optimizer?
            # Re-reading original `training.py` code provided in view_file:
            # Lines 297: `torch.nn.utils.clip_grad_norm_` is UNCONDITIONALLY called.
            # Lines 298: `optimizer.step()` is UNCONDITIONALLY called.
            # Ah, wait. In `dpo_training.py` (static), lines 206-210 had `if global_steps >= warmup_steps: clip; step;`.
            # This logic seems divergent. I will unify to standard DPO practice: always step. "Warmup" usually refers to LR schedule or beta warmup.
            # However, if the user intended to freeze updates during warmup, I should respect that.
            # Let's check `training.py` (dynamic) loop again.
            # lines 295-298: just zero_grad, backward, clip, step. No warmup check.
            # So dynamic DPO *updates* weights during beta warmup.
            # Now `dpo_training.py` (static):
            # lines 205: `if global_steps >= warmup_steps:` then clip and step.
            # This implies static DPO *skips* updates for the first N steps? That's odd. Usually warmup means LR warmup.
            # Maybe `warmup_steps` here means "reference model warmup" or something? No, it's DPO.
            # I will follow `training.py` behavior (always update) as it seems more standard, unless `dpo_training.py` has a specific reason.
            # Given `dpo_training.py` was likely an earlier or alternative experiment, and `training.py` is the main dynamic one, I'll stick to `training.py`'s update logic (always update).
            # If `dpo_training.py` really wanted to skip steps, that's a very specific behavior. I'll make it always update for now to be safe and consistent.
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            global_steps += 1 
            running_loss += loss.item()
            
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                wandb.log({
                    'loss': avg_loss,
                    'chosen_rewards': avg_chosen_rewards.item(),
                    'rejected_rewards': avg_rejected_rewards.item(),
                    'model_margin': avg_model_margin.item(),
                    'beta': beta_used
                })
                running_loss = 0.0

        eval_metrics = evaluate(policy, ref_model, val_loader, beta=beta_used, device=device)
        logger.info(f"[eval] loss={eval_metrics['eval_loss']:.4f} acc={eval_metrics['eval_reward_accuracy']:.3f}")
        wandb.log(eval_metrics)

        if is_dynamic:
             logger.info(f"[RISK] fail {risk_stat['fail']} / {risk_stat['total']}")

    if is_dynamic:
        log_f.close()

    save_dir = config['dpo_training'].get('save_dir', 'dpo_model')
    policy.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "static"], help="Training mode")
    args = parser.parse_args()
    
    train(args.config, args.mode)
