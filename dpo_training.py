import torch
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_process_hh import build_train_val
from dpo_loss import dpo_loss, margin_compute, compute_and_log_model_margin
from batch_log_prob import compute_batch_log_prob
import wandb
import os

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
    LOG_DIR = config['margin_log']['dpo_log_dir']
    os.makedirs(LOG_DIR, exist_ok=True)
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")

    # training loop
    # every epoch create a folder to save the model_margin
    epochs = config['dpo_training']['epochs']
    log_steps = config['dpo_training']['log_steps']
    
    warmup_steps = int(config['dpo_training']['warmup_steps'])
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
                     beta=config['dpo_training']['beta']
                     )
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()

            optimizer.zero_grad()
            loss.backward()
            # after warmup, update parameters
            if global_steps >= warmup_steps:
                 torch.nn.utils.clip_grad_norm_(
                     policy.parameters(),
                     max_norm=float(config['dpo_training']['max_grad_norm'])
                 )
                 optimizer.step()

            running_loss += loss.item()
            global_steps += 1

            # log the training info
            if (step + 1) % log_steps == 0:
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                wandb.log({
                    'loss': avg_loss,
                    'chosen_rewards': avg_chosen_rewards.item(),
                    'rejected_rewards': avg_rejected_rewards.item(),
                    'model_margin': avg_model_margin.item()
                })
                running_loss = 0.0

        eval_metrics = evaluate(policy, ref_model, val_loader, beta=config['dpo_training']['beta'], device=device)
        wandb.log({"eval_loss": eval_metrics["eval_loss"]})
        print(f"[eval] loss={eval_metrics['eval_loss']:.4f} "
              f"acc={eval_metrics['eval_reward_accuracy']:.3f} "
              f"cr={eval_metrics['eval_chosen_rewards']:.3f} "
              f"rr={eval_metrics['eval_rejected_rewards']:.3f}")

    # save model
    save_dir = config['dpo_training']['save_dpo_dir']
    policy.save_pretrained(save_dir)

if __name__ == "__main__":
    train()
