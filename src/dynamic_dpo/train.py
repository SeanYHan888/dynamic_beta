import torch
import torch.distributed as dist
from torch.optim import AdamW
import argparse
import numpy as np
import yaml
from tqdm import tqdm
import random
from functools import partial
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import os
import json
import logging
import time
from typing import Dict, Any

try:
    from .data import build_train_val
    from .modeling import (
        dpo_loss,
        margin_compute,
        compute_and_log_model_margin,
        compute_batch_log_prob,
        WarmupQuantileAccumulator,
        EMAUpdate,
        compute_dynamic_beta_update,
        gather_global_margins_and_broadcast_scalars,
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
        WarmupQuantileAccumulator,
        EMAUpdate,
        compute_dynamic_beta_update,
        gather_global_margins_and_broadcast_scalars,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def to_device_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}

def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def evaluate(policy, ref_model, val_loader, beta, accelerator):
    device = accelerator.device
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    total_loss = 0.0
    total_count = 0
    sum_chosen_rewards = 0.0
    sum_rejected_rewards = 0.0
    correct = 0  

    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        leave=False,
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
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
            beta
        )

        batches = loss_vec.shape[0]
        total_loss += loss_vec.mean().item() * batches
        total_count += batches
        sum_chosen_rewards += chosen_rewards.mean().item() * batches
        sum_rejected_rewards += rejected_rewards.mean().item() * batches
        correct += (chosen_rewards > rejected_rewards).sum().item()
        
    totals = torch.tensor(
        [total_loss, total_count, sum_chosen_rewards, sum_rejected_rewards, correct],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    total_loss, total_count, sum_chosen_rewards, sum_rejected_rewards, correct = totals.tolist()
    denom = max(1.0, total_count)
    metrics = {
        "eval_loss": total_loss / denom,
        "eval_chosen_rewards": sum_chosen_rewards / denom,
        "eval_rejected_rewards": sum_rejected_rewards / denom,
        "eval_reward_accuracy": correct / denom,
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
    use_bf16 = config['precision'] == 'bf16'
    import inspect

    fsdp_cfg = config.get('fsdp', {})
    fsdp_enabled = bool(fsdp_cfg.get('enabled', True))
    fsdp_state_offload = bool(fsdp_cfg.get('state_dict_offload', True))
    fsdp_rank0_only = bool(fsdp_cfg.get('save_on_rank0_only', True))
    fsdp_shard_ref_model = bool(fsdp_cfg.get("shard_ref_model", False))

    fsdp_plugin = None
    fsdp_missing_params: list[str] = []
    if fsdp_enabled:
        fsdp_kwargs = {
            "state_dict_config": FullStateDictConfig(
                offload_to_cpu=fsdp_state_offload,
                rank0_only=fsdp_rank0_only,
            ),
            "optim_state_dict_config": FullOptimStateDictConfig(
                offload_to_cpu=fsdp_state_offload,
                rank0_only=fsdp_rank0_only,
            ),
        }
        # Compatibility knobs: only pass when supported by the installed accelerate.
        plugin_sig = None
        try:
            plugin_sig = inspect.signature(FullyShardedDataParallelPlugin.__init__)
        except Exception:
            plugin_sig = None
        if plugin_sig is not None:
            # Enforce native FSDP settings to match `archive/dynamic_beta/training-fsdp.py`.
            if "sharding_strategy" in plugin_sig.parameters:
                fsdp_kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
            else:
                fsdp_missing_params.append("sharding_strategy")
            if "mixed_precision_policy" in plugin_sig.parameters:
                mp = MixedPrecision(
                    param_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16 if use_bf16 else torch.float32,
                )
                fsdp_kwargs["mixed_precision_policy"] = mp
            else:
                fsdp_missing_params.append("mixed_precision_policy")
            if "auto_wrap_policy" in plugin_sig.parameters:
                min_num_params = int(fsdp_cfg.get("min_num_params", int(1e7)))
                fsdp_kwargs["auto_wrap_policy"] = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
            else:
                fsdp_missing_params.append("auto_wrap_policy")
            if "use_orig_params" in plugin_sig.parameters:
                fsdp_kwargs["use_orig_params"] = bool(fsdp_cfg.get("use_orig_params", True))
            if "sync_module_states" in plugin_sig.parameters:
                fsdp_kwargs["sync_module_states"] = bool(fsdp_cfg.get("sync_module_states", True))
        else:
            fsdp_missing_params = ["sharding_strategy", "mixed_precision_policy", "auto_wrap_policy"]

        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_kwargs)

    accel_cfg = config.get("accelerate", {})
    accel_kwargs: Dict[str, Any] = {}
    try:
        accel_sig = inspect.signature(Accelerator.__init__)
    except Exception:
        accel_sig = None
    if accel_sig is not None:
        if "split_batches" in accel_sig.parameters:
            # Enforce global batch = (per-rank batch_size) * (num_processes).
            accel_kwargs["split_batches"] = False
        if "dispatch_batches" in accel_sig.parameters and "dispatch_batches" in accel_cfg:
            # If True, only rank0 builds batches and dispatches them to other ranks (can bottleneck on CPU/tokenization).
            accel_kwargs["dispatch_batches"] = bool(accel_cfg.get("dispatch_batches"))
        if "even_batches" in accel_sig.parameters and "even_batches" in accel_cfg:
            accel_kwargs["even_batches"] = bool(accel_cfg.get("even_batches"))

    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        mixed_precision="bf16" if use_bf16 else "no",
        **accel_kwargs,
    )
    device = accelerator.device
    if accelerator.is_main_process:
        logger.info(
            "accelerate: num_processes=%s split_batches=%s dispatch_batches=%s",
            accelerator.num_processes,
            getattr(accelerator, "split_batches", None),
            getattr(accelerator, "dispatch_batches", None),
        )
        if fsdp_enabled and fsdp_missing_params:
            logger.warning(
                "accelerate FSDP plugin does not expose %s; cannot fully enforce wrap/mixed-precision/sharding parity.",
                fsdp_missing_params,
            )

    seed_everything(config['dataset'].get('seed', 42))

    if accelerator.is_main_process:
        wandb.init(
            project=config.get('wandb_project', 'handwritten-dpo'),
            name=config.get('run_name', 'run'),
            config=config
        )
    accelerator.wait_for_everyone()

    # Load models
    policy_name = config['policy_name']
    ref_name = config['ref_name']
    policy = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )
    tok = AutoTokenizer.from_pretrained(policy_name)
    policy.config.pad_token_id = tok.pad_token_id

    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    # Load ref model in the compute dtype to reduce activation memory 
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)

    train_loader, val_loader = build_train_val(config=config, tokenizer=tok, accelerator=accelerator)

    # Important:
    # - Use `accelerator.prepare(...)` for model/optimizer (FSDP wrapping, DDP sync, etc.)
    # - Do NOT pass dataloaders into `accelerator.prepare(...)` here since we already construct
    #   a `DistributedSampler` in `build_train_val(...)` for multi-process runs; double-sharding
    #   can otherwise change the effective sample stream.
    if fsdp_enabled and fsdp_shard_ref_model:
        policy, ref_model = accelerator.prepare(policy, ref_model)
        ref_model.requires_grad_(False)
    else:
        policy = accelerator.prepare(policy)

    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))
    optimizer = accelerator.prepare(optimizer)
    if not (fsdp_enabled and fsdp_shard_ref_model):
        ref_model.to(device)

    if accelerator.is_main_process:
        logger.info("dataloader types: train=%s val=%s", type(train_loader).__name__, type(val_loader).__name__)

        def _sampler_name(dl):
            sampler_candidates = []
            sampler_candidates.append(getattr(dl, "sampler", None))
            sampler_candidates.append(getattr(getattr(dl, "batch_sampler", None), "sampler", None))
            for s in sampler_candidates:
                if s is not None:
                    return type(s).__name__
            return None

        logger.info("samplers: train=%s val=%s", _sampler_name(train_loader), _sampler_name(val_loader))
        per_rank_bs = int(config['dpo_training']['batch_size'])
        logger.info("expected global batch size=%s", per_rank_bs * accelerator.num_processes)

    policy.train()
    ref_model.eval()

    # Logging setup
    # Determine log dir based on mode or config
    if mode == 'dynamic':
         LOG_DIR = config['margin_log'].get('log_dir', 'logs/margins')
    else:
         LOG_DIR = config['margin_log'].get('dpo_log_dir', 'logs/dpo_margins')
         
    if accelerator.is_main_process:
        os.makedirs(LOG_DIR, exist_ok=True)
    accelerator.wait_for_everyone()
    JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")
    margin_log_cfg = config.get("margin_log", {})
    margin_log_enabled = bool(margin_log_cfg.get("enabled", True))
    margin_log_every = max(1, int(margin_log_cfg.get("every_steps", 1)))
    margin_log_save_npy = bool(margin_log_cfg.get("save_npy", True))
    margin_log_save_jsonl = bool(margin_log_cfg.get("save_jsonl", True))
    margin_log_sample_size = margin_log_cfg.get("sample_size", None)

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
        if beta_min <= 0.0:
            # If beta reaches 0, DPO gradients become exactly 0 and training will appear frozen.
            if accelerator.is_main_process:
                logger.warning("beta_min<=0 will zero gradients if beta hits 0; clamping beta_min to 1e-6.")
            beta_min = 1e-6
        
        beta = beta_0 # start with beta_0
        
        risk_stat = {"total": 0, "fail": 0}
        log_f = None
        if accelerator.is_main_process:
            log_f = open("risk_test_and_beta_log.jsonl", "w", encoding="utf-8")
        warmup_steps = max(0, int(config['dpo_training']['warmup_steps']))
        # warmup_steps==0 means we skip buffering and initialize EMA tau from the first batch.
        warmup_done = warmup_steps == 0
        needs_ema_init = warmup_steps == 0
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
    debug_cfg = config.get("debug", {})
    debug_log_updates = bool(debug_cfg.get("log_param_updates", False))
    debug_param_n = int(debug_cfg.get("param_update_n", 1024))
    profile_cfg = config.get("profile", {})
    profile_timing = bool(profile_cfg.get("timing", False))
    debug_param = None
    if debug_log_updates and accelerator.is_main_process:
        try:
            debug_param = next(iter(policy.parameters()))
        except StopIteration:
            debug_param = None

    for epoch in range(epochs):
        # Ensure DistributedSampler is reshuffled deterministically across ranks each epoch (if present).
        sampler_candidates = []
        sampler_candidates.append(getattr(train_loader, "sampler", None))
        sampler_candidates.append(getattr(getattr(train_loader, "batch_sampler", None), "sampler", None))
        if hasattr(train_loader, "dataloader"):
            sampler_candidates.append(getattr(train_loader.dataloader, "sampler", None))
            sampler_candidates.append(getattr(getattr(train_loader.dataloader, "batch_sampler", None), "sampler", None))
        for sampler in sampler_candidates:
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
                break

        epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
        if accelerator.is_main_process:
            os.makedirs(epoch_dir, exist_ok=True)

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train | epoch {epoch+1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not accelerator.is_main_process,
        )
        
        window_start_t = time.perf_counter()
        running_loss_sum = 0.0
        running_loss_count = 0
        param_before = None
        time_data_s = 0.0
        time_fwd_s = 0.0
        time_bwd_s = 0.0
        
        for step, batch in pbar:
            step_t0 = time.perf_counter()
            batch = to_device_batch(batch, device)
            if profile_timing:
                time_data_s += time.perf_counter() - step_t0

            if debug_param is not None and accelerator.is_main_process and (step + 1) % log_steps == 0:
                with torch.no_grad():
                    flat = debug_param.detach().float().view(-1)
                    n = min(int(flat.numel()), max(1, debug_param_n))
                    param_before = flat[:n].cpu()

            fwd_t0 = time.perf_counter()
            with accelerator.autocast():
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(
                    batch, policy=policy, ref_model=ref_model
                )

                model_margin = margin_compute(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob
                )

                # Dynamic DPO Logic
                beta_used = beta
                global_margins = None
                tau_for_log = None
                if is_dynamic:
                    local_margins = model_margin.detach().float().view(-1)
                    if not warmup_done:
                        # Warmup: gather global margins and buffer them for tau_0.
                        global_margins = accelerator.gather_for_metrics(local_margins)
                        if accelerator.is_main_process:
                            threshold_accumulator.update(global_margins)
                        warmup_count += 1
                        if warmup_count == warmup_steps:
                            # Warmup end: tau_0 from global margins, then initialize EMA.
                            if accelerator.is_main_process:
                                tau_0 = threshold_accumulator.finalize()
                                ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                                if log_f is not None:
                                    log_f.write(json.dumps({
                                        "type": "warmup_end",
                                        "tau_0": float(tau_0),
                                        "beta_0": float(beta)
                                    }) + "\n")
                                    log_f.flush()
                            warmup_done = True
                            needs_ema_init = False
                        beta_used = beta_0
                    else:
                        # Post-warmup: compute tau/p_hat/risk/beta on rank0 using global margins.
                        compute_fn = partial(
                            compute_dynamic_beta_update,
                            beta=beta,
                            ema=ema,
                            needs_ema_init=needs_ema_init,
                            q=q,
                            momentum=momentum,
                            eplison_0=eplison_0,
                            delta=delta,
                            alpha=alpha,
                            gamma=gamma,
                            beta_min=beta_min,
                            beta_max=beta_max,
                        )

                        # Collect global margins on all ranks, compute tau/beta on rank0, then broadcast scalars.
                        global_margins, tau, beta_used, stats = gather_global_margins_and_broadcast_scalars(
                            accelerator, local_margins, compute_fn
                        )
                        beta = beta_used
                        tau_for_log = tau

                        if accelerator.is_main_process and stats is not None:
                            # Update EMA state and log risk stats on rank0 only.
                            if stats["warmup_end"] and log_f is not None:
                                log_f.write(json.dumps({
                                    "type": "warmup_end",
                                    "tau_0": float(stats["tau_0"]),
                                    "beta_0": float(stats["beta_before_update"]),
                                }) + "\n")
                                log_f.flush()

                            ema = stats["ema"]
                            needs_ema_init = stats["needs_ema_init"]
                            risk_stat["total"] += 1
                            if stats["risk_over"]:
                                risk_stat["fail"] += 1

                            if log_f is not None:
                                log_f.write(json.dumps({
                                    "step": int(global_steps),
                                    "tau": float(stats["tau"]),
                                    "p_hat": float(stats["p_hat"]),
                                    "risk_over": bool(stats["risk_over"]),
                                    "beta": float(stats["beta"]),
                                    "u_k": float(stats["u_k"]),
                                    "s_k": float(stats["s_k"]),
                                    "alpha": float(stats["alpha"]),
                                }) + "\n")
                                log_f.flush()

                if accelerator.is_main_process:
                    margins_for_logging = model_margin
                    if is_dynamic and global_margins is not None:
                        # Log global margins to reflect true distributed batch stats.
                        margins_for_logging = global_margins
                    if margin_log_enabled and (step % margin_log_every == 0):
                        compute_and_log_model_margin(
                            model_margin=margins_for_logging,
                            epoch_dir=epoch_dir,
                            epoch=epoch,
                            step=step,
                            JSONL_PATH=JSONL_PATH,
                            save_npy=margin_log_save_npy,
                            save_jsonl=margin_log_save_jsonl,
                            sample_size=margin_log_sample_size,
                        )

                loss_raw, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta=beta_used
                )
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()
                batch_loss_sum = float(loss_raw.detach().float().sum().item())
                batch_loss_count = int(loss_raw.numel())
            if profile_timing:
                time_fwd_s += time.perf_counter() - fwd_t0

            bwd_t0 = time.perf_counter()
            optimizer.zero_grad()
            accelerator.backward(loss)
            
            # Dynamic often clips grad only after warmup, but static usually always clips. 
           
            grad_norm = None
            if max_grad_norm and max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            if profile_timing:
                time_bwd_s += time.perf_counter() - bwd_t0
            
            global_steps += 1 
            running_loss_sum += batch_loss_sum
            running_loss_count += batch_loss_count
            
            if (step + 1) % log_steps == 0:
                elapsed = time.perf_counter() - window_start_t
                avg_loss_local = running_loss_sum / max(1.0, float(running_loss_count))
                totals = torch.tensor(
                    [running_loss_sum, float(running_loss_count)],
                    device=device,
                    dtype=torch.float64,
                )
                if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                    elapsed_t = torch.tensor([elapsed], device=device, dtype=torch.float64)
                    dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)
                    elapsed = float(elapsed_t.item())
                avg_loss_global = float(totals[0].item()) / max(1.0, float(totals[1].item()))
                examples_global = float(totals[1].item())
                examples_per_step_global = examples_global / float(log_steps)
                examples_per_sec_global = examples_global / max(1e-12, elapsed)

                if accelerator.is_main_process:
                    pbar.set_postfix(loss=f"{avg_loss_global:.3f}")
                param_delta_mean = None
                if debug_param is not None and param_before is not None:
                    with torch.no_grad():
                        flat = debug_param.detach().float().view(-1)
                        n = min(int(flat.numel()), param_before.numel())
                        after = flat[:n].cpu()
                        param_delta_mean = (after - param_before).abs().mean().item()
                log_payload = {
                    'loss': avg_loss_global,
                    'loss_local_rank0': avg_loss_local,
                    'chosen_rewards': avg_chosen_rewards.item(),
                    'rejected_rewards': avg_rejected_rewards.item(),
                    'model_margin': avg_model_margin.item(),
                    'beta': beta_used,
                    'grad_norm': float(grad_norm) if grad_norm is not None else None,
                    'performance/examples_per_step_global': examples_per_step_global,
                    'performance/examples_per_sec_global': examples_per_sec_global,
                }
                if profile_timing:
                    timing = torch.tensor(
                        [time_data_s, time_fwd_s, time_bwd_s],
                        device=device,
                        dtype=torch.float64,
                    )
                    if dist.is_available() and dist.is_initialized() and accelerator.num_processes > 1:
                        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    log_payload.update(
                        {
                            "performance/time_data_s_per_step_max": float(timing[0].item()) / float(log_steps),
                            "performance/time_fwd_s_per_step_max": float(timing[1].item()) / float(log_steps),
                            "performance/time_bwd_s_per_step_max": float(timing[2].item()) / float(log_steps),
                            "performance/time_total_s_per_step_max": float(elapsed) / float(log_steps),
                        }
                    )
                if param_delta_mean is not None:
                    log_payload["param_delta_mean_abs"] = float(param_delta_mean)
                if is_dynamic and tau_for_log is not None:
                    log_payload['tau'] = tau_for_log
                if accelerator.is_main_process:
                    wandb.log(log_payload)
                running_loss_sum = 0.0
                running_loss_count = 0
                window_start_t = time.perf_counter()
                time_data_s = 0.0
                time_fwd_s = 0.0
                time_bwd_s = 0.0

        eval_metrics = evaluate(policy, ref_model, val_loader, beta=beta_used, accelerator=accelerator)
        if accelerator.is_main_process:
            logger.info(f"[eval] loss={eval_metrics['eval_loss']:.4f} acc={eval_metrics['eval_reward_accuracy']:.3f}")
            wandb.log(eval_metrics)

            if is_dynamic:
                logger.info(f"[RISK] fail {risk_stat['fail']} / {risk_stat['total']}")

    if is_dynamic and log_f is not None:
        log_f.close()

    # Save final model: mirror `archive/dynamic_beta/training-fsdp.py`
    # - Gather FULL_STATE_DICT
    # - Offload to CPU
    # - Save weights as fp32 (load into a fresh fp32 base model)
    save_dir = config['dpo_training'].get('save_dir', 'dpo_model')
    accelerator.wait_for_everyone()

    cpu_state = None
    if fsdp_enabled and isinstance(policy, FSDP):
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        full_optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            policy,
            StateDictType.FULL_STATE_DICT,
            full_state_cfg,
            full_optim_cfg,
        ):
            cpu_state = policy.state_dict()
    else:
        # Non-FSDP: only rank0 needs the full model weights; move to CPU for saving.
        if accelerator.is_main_process:
            state = accelerator.unwrap_model(policy).state_dict()
            cpu_state = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()}

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            policy_name,
            torch_dtype=torch.float32,
        )
        base_model.load_state_dict(cpu_state, strict=False)
        base_model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        logger.info("[SAVE] saved fp32 model to: %s", save_dir)

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "static"], help="Training mode")
    args = parser.parse_args()
    
    train(args.config, args.mode)
