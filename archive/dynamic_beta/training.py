import torch
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_process_hh import build_train_val
from dpo_loss import dpo_loss, margin_compute, compute_and_log_model_margin, empirical_over_threshold_proportion, risk_test, update_beta
from batch_log_prob import compute_batch_log_prob
import wandb
import os
import json

from mean_and_var import WarmupQuantileAccumulator, EMAUpdate

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
# transform the batch to the device
def to_device_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# control randomness
def random_controler(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

# eval()
@torch.no_grad()
def evaluate(policy, ref_model, val_loader, beta, device):
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    total_loss = 0.0
    total_count = 0

    sum_chosen_rewards = 0.0
    sum_rejected_rewards = 0.0

    # chosen_rewards > rejected_rewards
    correct = 0  

    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        leave=False,        
        dynamic_ncols=True
    )

    for batch in pbar:
        batch = to_device_batch(batch, device)

        # compute the log prob
        policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(batch, policy=policy, ref_model=ref_model)

        # dpo loss
        loss_vec, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_log_prob,
            policy_rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
            beta
        )

        # loss_vec: (bs,)
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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random_controler()

    # initial wandb
    wandb.init(project=config.get('wandb_project','handwritten-dpo'),
               name=config.get('run_name','run'),
               config=config)

    # load model and tokenizer
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
    
    # load dataset
    train_loader, val_loader = build_train_val(config=config, tokenizer=tok)

    policy.train()
    ref_model.eval()

    # define optimizer
    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    # using bf 16
    use_bf16 = config['precision'] == 'bf16'
    if use_bf16:
        policy.to(dtype=torch.bfloat16)
        ref_model.to(dtype=torch.bfloat16)

    # define margin log
    LOG_DIR = config['margin_log']['log_dir']
    os.makedirs(LOG_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")

    # risk test parameter
    delta = float(config['risk_test']['delta'])
    eplison_0 = float(config['risk_test']['eplison_0'])
    momentum = float(config['risk_test']['lambda'])
    q = 1.0 -delta

    threshold = WarmupQuantileAccumulator(q=q)

    # beta_update
    gamma = float(config['beta_update']['gamma'])
    alpha = float(config['beta_update']['alpha'])
    beta_0 = float(config['beta_update']['beta_0'])
    beta_max = float(config['beta_update']['beta_max'])
    beta_min = float(config['beta_update']['beta_min'])

    # log risk test
    risk_stat = {
        "total": 0,
        "fail": 0,
        }
    
    log_f = open("risk_test_and_beta_log.jsonl", "w", encoding="utf-8")

    # training loop
    # every epoch create a folder to save the model_margin
    epochs = config['dpo_training']['epochs']
    log_steps = config['dpo_training']['log_steps']
    
    warmup_steps = int(config['dpo_training']['warmup_steps'])
    warmup_done = False
    warmup_count = 0
    global_steps = 0

    for epoch in range(epochs):
        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        pbar = tqdm(enumerate(train_loader),
                total=len(train_loader),
                desc=f"train | epoch {epoch+1}/{epochs}",
                dynamic_ncols=True, leave=False)
        
        running_loss = 0.0
        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            # generate logits for each part
            # using bf16
            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16): 
                # compute log_prob
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(batch, policy=policy, ref_model=ref_model)

                # compute the margin M
                model_margin = margin_compute(
                    policy_chosen_log_prob=policy_chosen_log_prob,
                    policy_rejected_log_prob=policy_rejected_log_prob,
                    ref_chosen_log_prob=ref_chosen_log_prob,
                    ref_rejected_log_prob=ref_rejected_log_prob,
                )

                
                # determine the initial mean and variance
                if not warmup_done:
                    threshold.update(model_margin)
                    warmup_count += 1   
                    
                    if (not warmup_done) and (warmup_count == warmup_steps):
                        tau_0 = threshold.finalize()
                        ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                        beta=beta_0
                        warmup_done = True
                        log_f.write(json.dumps({
                            "type": "warmup_end",
                            "tau_0": float(tau_0),
                            "beta_0": float(beta)
                            }) + "\n")
                        log_f.flush()
                        
                    # warmup phase: use fixed beta
                    beta_used = beta_0
                    u_k = None
                    s_k = None
                
                else:        
                    # EMA update the threshold
                    tau = ema.update_tau(model_margin)
                    
                    # tail proportion compute and risk test
                    num_margin = int(model_margin.numel())
                    p_hat = empirical_over_threshold_proportion(model_margin, tau)
                    is_over_risk, eplison, delta_prime = risk_test(p_hat=p_hat, eplison_0=eplison_0, delta=delta, n=num_margin)

                    # beta adjust equation
                    beta_new, u_k, s_k, alpha_used = update_beta(
                        beta=beta,
                        p_hat=p_hat,
                        delta_prime=delta_prime,
                        eplison=eplison,
                        alpha=alpha,
                        gamma=gamma,
                        beta_min=beta_min,
                        beta_max=beta_max)
                    
                    beta = beta_new         
                    beta_used = beta

                        
                    # count
                    risk_stat["total"] += 1
                    if is_over_risk:
                        risk_stat["fail"] += 1
                     
                    # log
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
                            model_margin=model_margin,
                            epoch_dir=epoch_dir,
                            epoch=epoch,
                            step=step,
                            JSONL_PATH=JSONL_PATH
                            )

                # compute loss
                loss_raw, chosen_rewards, rejected_rewards= dpo_loss(
                     policy_chosen_log_prob=policy_chosen_log_prob,
                     policy_rejected_log_prob=policy_rejected_log_prob,
                     ref_chosen_log_prob=ref_chosen_log_prob,
                     ref_rejected_log_prob=ref_rejected_log_prob,
                     beta=beta_used
                     )
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float(config['dpo_training']['max_grad_norm']))
            optimizer.step()
            global_steps += 1 
            running_loss += loss.item()
            
            
            # log the training info
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
        print(f"[eval] loss={eval_metrics['eval_loss']:.4f} "
              f"acc={eval_metrics['eval_reward_accuracy']:.3f} "
              f"cr={eval_metrics['eval_chosen_rewards']:.3f} "
              f"rr={eval_metrics['eval_rejected_rewards']:.3f}")

        print(
            f"[RISK] fail {risk_stat['fail']} / {risk_stat['total']} "
            f"= {risk_stat['fail'] / max(1, risk_stat['total']):.4f}"
            )
    log_f.close()

    # save model
    save_dir = config['dpo_training']['save_dir']
    policy.save_pretrained(save_dir)

if __name__ == "__main__":
    train()
