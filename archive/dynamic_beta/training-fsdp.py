import os
import json
import random
import argparse
import yaml
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset_process_hh import build_train_val_distributed
from batch_log_prob import compute_batch_log_prob
from dpo_loss import (
    dpo_loss,
    margin_compute,
    compute_and_log_model_margin,
    empirical_over_threshold_proportion,
    risk_test,
    update_beta,
)
from mean_and_var import WarmupQuantileAccumulator, EMAUpdate

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.api import (
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from functools import partial

import wandb


# config
def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def to_device_batch(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

# ranfom setting
def random_controller(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# distribution setting
def initial_distribution():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return device, rank, world_size, local_rank

# gather 1d tensor from all ranks
def gather_1d_tensor(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    x: (bs,) on current rank GPU
    return: (bs * world_size,) concatenated across all ranks
    """
    x = x.contiguous()
    gather_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gather_list, x)
    return torch.cat(gather_list, dim=0)


def broadcast_float(v: float, src: int = 0, device: torch.device | None = None) -> float:
    t = torch.tensor([v], device=device if device is not None else torch.device("cuda"))
    dist.broadcast(t, src=src)
    return float(t.item())


def fsdp_wrap(model, use_bf16: bool):
    mp = MixedPrecision(
        param_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        reduce_dtype=torch.float32,  # reduce in fp32
        buffer_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    return FSDP(
        model,
        auto_wrap_policy=partial(size_based_auto_wrap_policy, min_num_params=int(1e7)),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
    )


# eval
@torch.no_grad()
def evaluate(policy, ref_model, val_loader, beta, device, is_rank0: bool):
    """
    IMPORTANT (FSDP):
      - All ranks MUST enter evaluate() to participate in FSDP collectives.
      - We all_reduce totals to get true global metrics.
    """
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    # local accumulators (as python floats/ints)
    local_total_loss_sum = 0.0   # sum of (mean_loss_per_sample * bs) over samples
    local_total_count = 0        # number of samples
    local_sum_chosen_rewards = 0.0
    local_sum_rejected_rewards = 0.0
    local_correct = 0            # count(chosen_reward > rejected_reward)

    # only rank0 shows progress bar
    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        leave=False,
        dynamic_ncols=True,
        disable=not is_rank0,   
    )

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
            beta,
        )

        bs = int(loss_vec.shape[0])
        local_total_loss_sum += float(loss_vec.mean().item()) * bs
        local_total_count += bs
        local_sum_chosen_rewards += float(chosen_rewards.mean().item()) * bs
        local_sum_rejected_rewards += float(rejected_rewards.mean().item()) * bs
        local_correct += int((chosen_rewards > rejected_rewards).sum().item())

    # pack into one tensor to reduce number of collectives
    t = torch.tensor(
        [
            local_total_loss_sum,
            float(local_total_count),
            local_sum_chosen_rewards,
            local_sum_rejected_rewards,
            float(local_correct),
        ],
        device=device,
        dtype=torch.float32,
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_loss_sum = float(t[0].item())
    total_count = int(t[1].item())
    sum_chosen_rewards = float(t[2].item())
    sum_rejected_rewards = float(t[3].item())
    correct = float(t[4].item())

    metrics = {
        "eval_loss": total_loss_sum / max(1, total_count),
        "eval_chosen_rewards": sum_chosen_rewards / max(1, total_count),
        "eval_rejected_rewards": sum_rejected_rewards / max(1, total_count),
        "eval_reward_accuracy": correct / max(1, total_count),
        "eval_total_count": total_count,
    }

    if policy_was_training:
        policy.train()
    return metrics


# train 
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    device, rank, world_size, local_rank = initial_distribution()
    is_rank0 = (rank == 0)

    random_controller()

    # initial wandb (rank0 only)
    if is_rank0:
        wandb.init(
            project=config.get("wandb_project", "handwritten-dpo"),
            name=config.get("run_name", "run"),
            config=config,
        )
    else:
        wandb.init(mode="disabled")

    # model and tokenizer
    policy_name = config["policy_name"]
    ref_name = config["ref_name"]

    tok = AutoTokenizer.from_pretrained(policy_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    use_bf16 = (config["precision"] == "bf16")

    # policy: load on CPU then wrap with FSDP (do NOT .to(device) before FSDP)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    policy.config.pad_token_id = tok.pad_token_id
    policy = fsdp_wrap(policy, use_bf16=use_bf16)
    policy.train()

    # ref model: no grad, shard with FSDP as well (do NOT .to(device) before FSDP)
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    ref_model = fsdp_wrap(ref_model, use_bf16=use_bf16)
    ref_model.requires_grad_(False)
    ref_model.eval()

    # load data
    train_loader, val_loader = build_train_val_distributed(
        config=config, tokenizer=tok, rank=rank, world_size=world_size
    )

    # optimizer (MUST be after FSDP wrap) 
    optimizer = AdamW(params=policy.parameters(), lr=float(config["dpo_training"]["learning_rate"]))

    # margin log dir 
    LOG_DIR = config["margin_log"]["log_dir"]
    os.makedirs(LOG_DIR, exist_ok=True)

    # only rank0 writes global logs
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log_global.jsonl")
    risk_log_path = os.path.join(LOG_DIR, "risk_test_and_beta_log_global.jsonl")
    log_f = open(risk_log_path, "w", encoding="utf-8") if is_rank0 else None

    # risk parameters 
    delta = float(config["risk_test"]["delta"])
    eplison_0 = float(config["risk_test"]["eplison_0"])
    momentum = float(config["risk_test"]["lambda"])
    q = 1.0 - delta

    threshold = WarmupQuantileAccumulator(q=q)

    # beta update 
    gamma = float(config["beta_update"]["gamma"])
    alpha = float(config["beta_update"]["alpha"])
    beta_0 = float(config["beta_update"]["beta_0"])
    beta_max = float(config["beta_update"]["beta_max"])
    beta_min = float(config["beta_update"]["beta_min"])
    beta = beta_0  

    risk_stat = {"total": 0, "fail": 0}

    # training loop 
    epochs = int(config["dpo_training"]["epochs"])
    log_steps = int(config["dpo_training"]["log_steps"])
    warmup_steps = int(config["dpo_training"]["warmup_steps"])

    warmup_done = False
    warmup_count = 0
    global_steps = 0
    running_loss = 0.0

    # placeholder for EMA object
    ema = None

    for epoch in range(epochs):
        # distributed sampler needs set_epoch
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        if is_rank0:
            os.makedirs(epoch_dir, exist_ok=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train | epoch {epoch+1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not is_rank0,  
        )

        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            # autocast for bf16
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
            with autocast_ctx:
                # log_prob
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(
                    batch, policy=policy, ref_model=ref_model
                )

                # margin (local)
                model_margin = margin_compute(
                    policy_chosen_log_prob=policy_chosen_log_prob,
                    policy_rejected_log_prob=policy_rejected_log_prob,
                    ref_chosen_log_prob=ref_chosen_log_prob,
                    ref_rejected_log_prob=ref_rejected_log_prob,
                )  # (bs,)

                # GLOBALIZE margin across all GPUs 
                margin_global = gather_1d_tensor(model_margin, world_size)  

                # warmup / dynamic beta (GLOBAL)
                if not warmup_done:
                    # count steps on all ranks to keep same warmup boundary
                    warmup_count += 1

                    # only rank0 accumulates the global margins
                    if is_rank0:
                        threshold.update(margin_global)

                        if warmup_count == warmup_steps:
                            tau_0 = threshold.finalize()
                        else:
                            tau_0 = 0.0
                    else:
                        tau_0 = 0.0

                    if warmup_count == warmup_steps:
                        # broadcast tau_0, init ema, broadcast beta (fixed beta_0)
                        tau_0 = broadcast_float(tau_0, src=0, device=device)
                        ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                        beta = beta_0
                        warmup_done = True

                        if is_rank0:
                            log_f.write(
                                json.dumps(
                                    {
                                        "type": "warmup_end",
                                        "tau_0": float(tau_0),
                                        "beta_0": float(beta_0),
                                        "warmup_steps": int(warmup_steps),
                                        "global_margin_n": int(margin_global.numel()),
                                    }
                                )
                                + "\n"
                            )
                            log_f.flush()

                    beta_used = beta_0
                    u_k, s_k, alpha_used = None, None, None
                    tau, p_hat, is_over_risk = None, None, None

                else:
                    # only rank0 computes tau/p_hat/beta from GLOBAL margins
                    if is_rank0:
                        tau = ema.update_tau(margin_global)
                        num_margin = int(margin_global.numel())
                        p_hat = empirical_over_threshold_proportion(margin_global, tau)
                        is_over_risk, eplison, delta_prime = risk_test(
                            p_hat=p_hat, eplison_0=eplison_0, delta=delta, n=num_margin
                        )

                        beta_new, u_k, s_k, alpha_used = update_beta(
                            beta=beta,
                            p_hat=p_hat,
                            delta_prime=delta_prime,
                            eplison=eplison,
                            alpha=alpha,
                            gamma=gamma,
                            beta_min=beta_min,
                            beta_max=beta_max,
                        )

                        # stats
                        risk_stat["total"] += 1
                        if is_over_risk:
                            risk_stat["fail"] += 1

                        # log global
                        log_f.write(
                            json.dumps(
                                {
                                    "step": int(global_steps),
                                    "tau": float(tau),
                                    "p_hat": float(p_hat),
                                    "risk_over": bool(is_over_risk),
                                    "beta_old": float(beta),
                                    "beta_new": float(beta_new),
                                    "u_k": float(u_k),
                                    "s_k": float(s_k),
                                    "alpha": float(alpha_used),
                                    "global_margin_n": int(num_margin),
                                }
                            )
                            + "\n"
                        )
                        log_f.flush()
                    else:
                        beta_new = 0.0

                    # broadcast beta_new -> all ranks
                    beta = broadcast_float(beta_new, src=0, device=device)
                    beta_used = beta

                # log global margin distribution 
                if is_rank0:
                    compute_and_log_model_margin(
                        model_margin=margin_global,  # GLOBAL margin!
                        epoch_dir=epoch_dir,
                        epoch=epoch,
                        step=step,
                        JSONL_PATH=JSONL_PATH,
                    )

                # DPO loss uses beta_used (same on all ranks)
                loss_raw, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_log_prob=policy_chosen_log_prob,
                    policy_rejected_log_prob=policy_rejected_log_prob,
                    ref_chosen_log_prob=ref_chosen_log_prob,
                    ref_rejected_log_prob=ref_rejected_log_prob,
                    beta=beta_used,
                )

                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = margin_global.mean()  

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # FSDP grad clip 
            policy.clip_grad_norm_(max_norm=float(config["dpo_training"]["max_grad_norm"]))

            optimizer.step()

            global_steps += 1
            running_loss += loss.item()

            # log train info (rank0 only)
            if is_rank0 and ((step + 1) % log_steps == 0):
                avg_loss = running_loss / log_steps
                pbar.set_postfix(loss=f"{avg_loss:.3f}")
                wandb.log(
                    {
                        "loss": avg_loss,
                        "chosen_rewards": float(avg_chosen_rewards.item()),
                        "rejected_rewards": float(avg_rejected_rewards.item()),
                        "model_margin_global_mean": float(avg_model_margin.item()),
                        "beta": float(beta_used),
                    },
                    step=global_steps,
                )
                running_loss = 0.0

        # eval (FIXED): ALL ranks participate to avoid FSDP deadlock 
        dist.barrier()
        eval_metrics = evaluate(
            policy, ref_model, val_loader,
            beta=float(beta_used),
            device=device,
            is_rank0=is_rank0,
        )
        dist.barrier()

        if is_rank0:
            print(
                f"[eval] loss={eval_metrics['eval_loss']:.4f} "
                f"acc={eval_metrics['eval_reward_accuracy']:.3f} "
                f"cr={eval_metrics['eval_chosen_rewards']:.3f} "
                f"rr={eval_metrics['eval_rejected_rewards']:.3f} "
                f"(N={eval_metrics['eval_total_count']})"
            )
            print(
                f"[RISK] fail {risk_stat['fail']} / {risk_stat['total']} "
                f"= {risk_stat['fail'] / max(1, risk_stat['total']):.4f}"
            )

    if is_rank0 and log_f is not None:
        log_f.close()

    # save model (rank0 only, gather full state dict)
    dist.barrier()
    save_dir = config["dpo_training"]["save_dir"]
    if is_rank0:
        os.makedirs(save_dir, exist_ok=True)

    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        policy,
        StateDictType.FULL_STATE_DICT,
        full_state_cfg,
        full_optim_cfg,
    ):
        cpu_state = policy.state_dict()

    if is_rank0:
        base_model = AutoModelForCausalLM.from_pretrained(
            policy_name,
            torch_dtype=torch.float32,
        )
        base_model.load_state_dict(cpu_state, strict=False)
        base_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"[SAVE] saved to: {save_dir}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
