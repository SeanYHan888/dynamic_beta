import argparse
import inspect
import json
import logging
import os
import random
from functools import partial
from typing import Any, Dict

import numpy as np
import torch
import wandb
import yaml
from accelerate import Accelerator
try:
    from accelerate.utils import FullyShardedDataParallelPlugin
except ImportError:
    from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import FullStateDictConfig, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from .data import build_train_val
from .modeling import (
    dpo_loss, 
    margin_compute, 
    compute_and_log_model_margin, 
    compute_batch_log_prob,
    risk_test, 
    update_beta,
    empirical_over_threshold_proportion,
    WarmupQuantileAccumulator,
    EMAUpdate,
    build_debug_payload
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

def resolve_fsdp_layer_classes(model, class_names):
    layer_classes = set()
    for class_name in class_names:
        for module in model.modules():
            if module.__class__.__name__ == class_name:
                layer_classes.add(module.__class__)
                break
    return layer_classes

def build_accelerator(config: Dict[str, Any], policy, mixed_precision: str) -> Accelerator:
    fsdp_config = config.get("fsdp", {})
    fsdp_enabled = bool(fsdp_config.get("enabled", False))
    if not fsdp_enabled:
        return Accelerator(mixed_precision=mixed_precision)
    if not torch.cuda.is_available():
        logger.warning("FSDP enabled but CUDA is not available. Disabling FSDP.")
        return Accelerator(mixed_precision=mixed_precision)

    auto_wrap_policy = None
    layer_cls_names = fsdp_config.get("auto_wrap_layers", [])
    if isinstance(layer_cls_names, str):
        layer_cls_names = [name.strip() for name in layer_cls_names.split(",") if name.strip()]
    elif layer_cls_names is None:
        layer_cls_names = []
    if layer_cls_names:
        layer_classes = resolve_fsdp_layer_classes(policy, layer_cls_names)
        if layer_classes:
            auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)
        else:
            logger.warning(
                "FSDP auto_wrap_layers not found in model; wrapping full model. auto_wrap_layers=%s",
                layer_cls_names,
            )

    fsdp_version = int(fsdp_config.get("version", fsdp_config.get("fsdp_version", 2)))
    use_fsdp2 = fsdp_version == 2

    def build_mp_policy(fsdp2: bool):
        if mixed_precision != "bf16":
            return None
        if fsdp2:
            try:
                from torch.distributed.fsdp import MixedPrecisionPolicy
            except ImportError as exc:
                raise RuntimeError(
                    "FSDP2 mixed precision requires torch.distributed.fsdp.MixedPrecisionPolicy."
                ) from exc
            mp_kwargs = {
                "param_dtype": torch.bfloat16,
                "reduce_dtype": torch.bfloat16,
                "buffer_dtype": torch.bfloat16,
            }
            mp_sig = inspect.signature(MixedPrecisionPolicy)
            supported = {k: v for k, v in mp_kwargs.items() if k in mp_sig.parameters}
            return MixedPrecisionPolicy(**supported)
        return MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    mp_policy = build_mp_policy(use_fsdp2)

    reshard_cfg = fsdp_config.get("reshard_after_forward", None)
    if use_fsdp2:
        if reshard_cfg is None:
            reshard_after_forward = True
        elif isinstance(reshard_cfg, bool):
            reshard_after_forward = reshard_cfg
        elif isinstance(reshard_cfg, str):
            key = reshard_cfg.strip().lower()
            if key in ("true", "1", "yes", "full_shard", "full"):
                reshard_after_forward = True
            elif key in ("false", "0", "no", "no_shard", "none"):
                reshard_after_forward = False
            else:
                raise ValueError(f"Unsupported fsdp.reshard_after_forward value: {reshard_cfg!r}")
        else:
            raise ValueError(f"Unsupported fsdp.reshard_after_forward type: {type(reshard_cfg).__name__}")
    else:
        if reshard_cfg is None:
            reshard_after_forward = ShardingStrategy.FULL_SHARD
        elif isinstance(reshard_cfg, ShardingStrategy):
            reshard_after_forward = reshard_cfg
        elif isinstance(reshard_cfg, str):
            key = reshard_cfg.strip().lower()
            if key in ("full_shard", "full", "fsdp", "true"):
                reshard_after_forward = ShardingStrategy.FULL_SHARD
            elif key in ("shard_grad_op", "shard_grad", "grad", "hybrid"):
                reshard_after_forward = ShardingStrategy.SHARD_GRAD_OP
            elif key in ("no_shard", "none", "false"):
                reshard_after_forward = ShardingStrategy.NO_SHARD
            else:
                raise ValueError(f"Unsupported fsdp.reshard_after_forward value: {reshard_cfg!r}")
        elif isinstance(reshard_cfg, bool):
            reshard_after_forward = ShardingStrategy.FULL_SHARD if reshard_cfg else ShardingStrategy.NO_SHARD
        else:
            raise ValueError(f"Unsupported fsdp.reshard_after_forward type: {type(reshard_cfg).__name__}")

    fsdp_plugin_kwargs = dict(
        reshard_after_forward=reshard_after_forward,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision_policy=mp_policy,
        state_dict_type=StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        limit_all_gathers=bool(fsdp_config.get("limit_all_gathers", True)),
    )
    plugin_sig = inspect.signature(FullyShardedDataParallelPlugin)
    actual_fsdp_version = fsdp_version if use_fsdp2 else 1
    if "fsdp_version" in plugin_sig.parameters:
        fsdp_plugin_kwargs["fsdp_version"] = fsdp_version
    elif use_fsdp2:
        logger.warning(
            "FSDP2 requested but accelerate.FullyShardedDataParallelPlugin has no fsdp_version; falling back to FSDP1."
        )
        actual_fsdp_version = 1
        fsdp_plugin_kwargs["mixed_precision_policy"] = build_mp_policy(False)
        if isinstance(reshard_after_forward, bool):
            fsdp_plugin_kwargs["reshard_after_forward"] = (
                ShardingStrategy.FULL_SHARD if reshard_after_forward else ShardingStrategy.NO_SHARD
            )

    try:
        fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if use_fsdp2 and "FSDP1" in msg and "reshard_after_forward" in msg:
            logger.warning(
                "FSDP2 requested but plugin rejected bool reshard_after_forward; falling back to FSDP1."
            )
            fsdp_plugin_kwargs.pop("fsdp_version", None)
            raf = fsdp_plugin_kwargs.get("reshard_after_forward")
            if isinstance(raf, bool):
                fsdp_plugin_kwargs["reshard_after_forward"] = (
                    ShardingStrategy.FULL_SHARD if raf else ShardingStrategy.NO_SHARD
                )
            fsdp_plugin_kwargs["mixed_precision_policy"] = build_mp_policy(False)
            actual_fsdp_version = 1
            fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_plugin_kwargs)
        else:
            raise
    accelerator = Accelerator(mixed_precision=mixed_precision, fsdp_plugin=fsdp_plugin)
    accelerator._fsdp_version_used = actual_fsdp_version
    return accelerator

@torch.no_grad()
def evaluate(policy, ref_model, val_loader, beta, accelerator):
    policy_was_training = policy.training
    policy.eval()
    ref_model.eval()

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_count = torch.tensor(0.0, device=accelerator.device)
    sum_chosen_rewards = torch.tensor(0.0, device=accelerator.device)
    sum_rejected_rewards = torch.tensor(0.0, device=accelerator.device)
    correct = torch.tensor(0.0, device=accelerator.device)

    pbar = tqdm(
        val_loader,
        desc="Evaluating",
        leave=False,
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    for batch in pbar:
        batch = to_device_batch(batch, accelerator.device)

        with accelerator.autocast():
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

        total_loss += loss_vec.sum()
        total_count += loss_vec.shape[0]
        sum_chosen_rewards += chosen_rewards.sum()
        sum_rejected_rewards += rejected_rewards.sum()
        correct += (chosen_rewards > rejected_rewards).sum()

    total_loss = accelerator.reduce(total_loss, reduction="sum")
    total_count = accelerator.reduce(total_count, reduction="sum")
    sum_chosen_rewards = accelerator.reduce(sum_chosen_rewards, reduction="sum")
    sum_rejected_rewards = accelerator.reduce(sum_rejected_rewards, reduction="sum")
    correct = accelerator.reduce(correct, reduction="sum")

    metrics = {}
    if accelerator.is_main_process:
        denom = max(1.0, total_count.item())
        metrics = {
            "eval_loss": total_loss.item() / denom,
            "eval_chosen_rewards": sum_chosen_rewards.item() / denom,
            "eval_rejected_rewards": sum_rejected_rewards.item() / denom,
            "eval_reward_accuracy": correct.item() / denom,
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
    # Load models
    policy_name = config['policy_name']
    ref_name = config['ref_name']
    policy = AutoModelForCausalLM.from_pretrained(policy_name)
    tok = AutoTokenizer.from_pretrained(policy_name)
    policy.config.pad_token_id = tok.pad_token_id

    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(ref_name)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)

    use_bf16 = config['precision'] == 'bf16' and torch.cuda.is_available()
    mixed_precision = "bf16" if use_bf16 else "no"
    accelerator = build_accelerator(config, policy, mixed_precision=mixed_precision)
    device = accelerator.device
    if accelerator.is_main_process:
        logger.info(
            "precision config=%s cuda_available=%s mixed_precision=%s accelerator_mixed_precision=%s device=%s num_processes=%s",
            config.get("precision"),
            torch.cuda.is_available(),
            mixed_precision,
            accelerator.mixed_precision,
            device,
            accelerator.num_processes,
        )
        logger.info(
            "CUDA_VISIBLE_DEVICES=%s torch.cuda.device_count=%s",
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch.cuda.device_count(),
        )
        fsdp_cfg = config.get("fsdp", {})
        logger.info(
            "fsdp_enabled=%s fsdp_version_config=%s fsdp_version_used=%s",
            fsdp_cfg.get("enabled", False),
            fsdp_cfg.get("version", fsdp_cfg.get("fsdp_version")),
            getattr(accelerator, "_fsdp_version_used", None),
        )

    seed = config['dataset'].get('seed', 42)
    seed_everything(seed + accelerator.process_index)

    if accelerator.is_main_process:
        wandb.init(
            project=config.get('wandb_project', 'handwritten-dpo'),
            name=config.get('run_name', 'run'),
            config=config
        )

    config.setdefault("distributed", {})
    config["distributed"]["world_size"] = accelerator.num_processes
    config["distributed"]["rank"] = accelerator.process_index

    train_loader, val_loader, train_sampler, val_sampler = build_train_val(config=config, tokenizer=tok)

    optimizer = AdamW(params=policy.parameters(), lr=float(config['dpo_training']['learning_rate']))

    if use_bf16:
        ref_model.to(dtype=torch.bfloat16)
    if getattr(accelerator, "_fsdp_version_used", None) == 2:
        policy, optimizer, train_loader, val_loader = accelerator.prepare(
            policy, optimizer, train_loader, val_loader
        )
        ref_model.to(device)
    else:
        policy, ref_model, optimizer, train_loader, val_loader = accelerator.prepare(
            policy, ref_model, optimizer, train_loader, val_loader
        )
    policy.train()
    ref_model.eval()
    if accelerator.is_main_process:
        policy_dtype = next(policy.parameters()).dtype
        ref_dtype = next(ref_model.parameters()).dtype
        logger.info(
            "policy_param_dtype=%s ref_param_dtype=%s (autocast=%s)",
            policy_dtype,
            ref_dtype,
            accelerator.mixed_precision,
        )

    # Logging setup
    # Determine log dir based on mode or config
    if mode == 'dynamic':
         LOG_DIR = config['margin_log'].get('log_dir', 'logs/margins')
    else:
         LOG_DIR = config['margin_log'].get('dpo_log_dir', 'logs/dpo_margins')
    
    if accelerator.is_main_process:
        os.makedirs(LOG_DIR, exist_ok=True)
        JSONL_PATH = os.path.join(LOG_DIR, "margins_log.jsonl")
    else:
        JSONL_PATH = None

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

        risk_stat = {"total": 0, "fail": 0} if accelerator.is_main_process else None
        log_f = None
        if accelerator.is_main_process:
            log_f = open("risk_test_and_beta_log.jsonl", "w", encoding="utf-8")
        warmup_steps = int(config['dpo_training']['warmup_steps'])
        warmup_done = warmup_steps <= 0
        warmup_count = 0
        ema = EMAUpdate(tau_0=0.0, q=q, momentum=momentum) if warmup_done else None
    else:
        # Static DPO
        # For static, we might still have warmup for LR, but not for beta
        warmup_steps = int(config['dpo_training'].get('warmup_steps', 0)) # Might be used for scheduler, but not beta
    
    epochs = config['dpo_training']['epochs']
    log_steps = config['dpo_training']['log_steps']
    global_steps = 0
    max_grad_norm = float(config['dpo_training']['max_grad_norm'])

    margin_log_cfg = config.get('margin_log', {})
    margin_log_every = int(margin_log_cfg.get('log_every', 1))
    margin_log_sample_size = int(margin_log_cfg.get('sample_size', 0))
    margin_log_save_npy = bool(margin_log_cfg.get('save_npy', True))

    debug_cfg = config.get("debug", {})
    debug_max_batches = int(debug_cfg.get("max_batches", 1))
    debug_max_preview = int(debug_cfg.get("max_preview_tokens", 64))
    debug_print_max = int(debug_cfg.get("print_batches", 3))
    debug_batches_left = debug_max_batches
    debug_print_left = min(debug_print_max, debug_max_batches)
    debug_log_f = None
    if accelerator.is_main_process and debug_max_batches > 0:
        debug_log_path = debug_cfg.get("log_path")
        if not debug_log_path:
            debug_log_path = os.path.join(LOG_DIR, "debug_batches.jsonl")
        os.makedirs(os.path.dirname(debug_log_path) or ".", exist_ok=True)
        debug_log_f = open(debug_log_path, "a", encoding="utf-8")

    for epoch in range(epochs):
        # Handle shuffling for Accelerate-prepared loaders or manual samplers
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(epoch)
        elif train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        if hasattr(val_loader, "set_epoch"):
            val_loader.set_epoch(epoch)
        elif val_sampler is not None:
            val_sampler.set_epoch(epoch)

        if accelerator.is_main_process:
            epoch_dir = os.path.join(LOG_DIR, f"epoch_{epoch:03d}")
            os.makedirs(epoch_dir, exist_ok=True)
        else:
            epoch_dir = None

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train | epoch {epoch+1}/{epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not accelerator.is_main_process,
        )
        
        running_loss_sum = torch.tensor(0.0, device=device)
        running_loss_count = torch.tensor(0.0, device=device)
        
        for step, batch in pbar:
            batch = to_device_batch(batch, device)

            if accelerator.is_main_process and debug_batches_left > 0:
                debug_payload = build_debug_payload(batch, tok, max_preview_tokens=debug_max_preview)
                if debug_log_f is not None:
                    debug_record = {
                        "epoch": int(epoch),
                        "step": int(step),
                        "global_step": int(global_steps),
                        **debug_payload,
                    }
                    debug_log_f.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
                    debug_log_f.flush()
                if debug_print_left > 0:
                    logger.info("raw_record: %s", repr(debug_payload["raw_record"]))
                    logger.info("chosen_input_ids: %s", debug_payload["chosen_input_ids"])
                    logger.info("chosen_attention_mask: %s", debug_payload["chosen_attention_mask"])
                    logger.info("chosen_labels: %s", debug_payload["chosen_labels"])
                    logger.info("rejected_input_ids: %s", debug_payload["rejected_input_ids"])
                    logger.info("rejected_attention_mask: %s", debug_payload["rejected_attention_mask"])
                    logger.info("rejected_labels: %s", debug_payload["rejected_labels"])
                    logger.info("debug_stats: %s", json.dumps(debug_payload["stats"]))
                    debug_print_left -= 1
                debug_batches_left -= 1

            with accelerator.autocast():
                policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob = compute_batch_log_prob(
                    batch, policy=policy, ref_model=ref_model
                )

                model_margin = margin_compute(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob
                )

                # Dynamic DPO Logic
                beta_used = beta
                margins_for_log = model_margin
                if is_dynamic:
                    # Sync margins for correct global statistics; compute risk only on rank0.
                    all_margins = accelerator.gather(model_margin)
                    margins_for_log = all_margins

                    if accelerator.is_main_process:
                        if not warmup_done:
                            threshold_accumulator.update(all_margins)
                            warmup_count += 1
                            if warmup_count >= warmup_steps:
                                tau_0 = threshold_accumulator.finalize()
                                ema = EMAUpdate(tau_0=tau_0, q=q, momentum=momentum)
                                warmup_done = True
                                if log_f is not None:
                                    log_f.write(json.dumps({
                                        "type": "warmup_end",
                                        "tau_0": float(tau_0),
                                        "beta_0": float(beta)
                                    }) + "\n")
                                    log_f.flush()
                            beta_used = beta_0
                        else:
                            tau = ema.update_tau(all_margins)

                            # Calculate p_hat exactly using the global batch
                            num_margin = all_margins.numel()
                            p_hat = empirical_over_threshold_proportion(all_margins, tau)

                            is_over_risk, eplison, delta_prime = risk_test(
                                p_hat=p_hat,
                                eplison_0=eplison_0,
                                delta=delta,
                                n=num_margin,
                            )

                            beta, u_k, s_k, alpha_used = update_beta(
                                beta, p_hat, delta_prime, eplison, alpha, gamma, beta_min, beta_max
                            )
                            beta_used = beta

                            if risk_stat is not None:
                                risk_stat["total"] += 1
                                if is_over_risk:
                                    risk_stat["fail"] += 1

                            if log_f is not None:
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

                    if accelerator.num_processes > 1 and torch.distributed.is_initialized():
                        sync = torch.tensor(
                            [beta, beta_used, float(warmup_done)],
                            device=device,
                        )
                        torch.distributed.broadcast(sync, src=0)
                        beta = float(sync[0].item())
                        beta_used = float(sync[1].item())
                        warmup_done = bool(int(sync[2].item()))

                if accelerator.is_main_process and margin_log_every > 0 and (global_steps % margin_log_every == 0):
                    compute_and_log_model_margin(
                        model_margin=margins_for_log,
                        epoch_dir=epoch_dir,
                        epoch=epoch,
                        step=step,
                        JSONL_PATH=JSONL_PATH,
                        sample_size=margin_log_sample_size,
                        save_npy=margin_log_save_npy,
                    )

                loss_raw, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta=beta_used
                )
                
                loss = loss_raw.mean()
                avg_chosen_rewards = chosen_rewards.mean()
                avg_rejected_rewards = rejected_rewards.mean()
                avg_model_margin = model_margin.mean()

            optimizer.zero_grad()
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(policy.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            global_steps += 1 
            running_loss_sum += loss_raw.detach().sum()
            running_loss_count += loss_raw.new_tensor(loss_raw.numel())
            
            if (step + 1) % log_steps == 0:
                global_loss_sum = accelerator.reduce(running_loss_sum, reduction="sum")
                global_loss_count = accelerator.reduce(running_loss_count, reduction="sum")
                if accelerator.is_main_process:
                    denom = max(global_loss_count.item(), 1.0)
                    avg_loss = global_loss_sum.item() / denom
                    pbar.set_postfix(loss=f"{avg_loss:.3f}")
                    wandb.log({
                        'loss': avg_loss,
                        'chosen_rewards': avg_chosen_rewards.item(),
                        'rejected_rewards': avg_rejected_rewards.item(),
                        'model_margin': avg_model_margin.item(),
                        'beta': beta_used
                    })
                running_loss_sum.zero_()
                running_loss_count.zero_()

        eval_metrics = evaluate(policy, ref_model, val_loader, beta=beta_used, accelerator=accelerator)
        if accelerator.is_main_process:
            logger.info(f"[eval] loss={eval_metrics['eval_loss']:.4f} acc={eval_metrics['eval_reward_accuracy']:.3f}")
            wandb.log(eval_metrics)

        if is_dynamic:
            if accelerator.is_main_process and risk_stat is not None:
                logger.info(f"[RISK] fail {risk_stat['fail']} / {risk_stat['total']}")

    if is_dynamic:
        if log_f is not None:
            log_f.close()
    if debug_log_f is not None:
        debug_log_f.close()

    save_dir = config['dpo_training'].get('save_dir', 'dpo_model')
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(policy)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(save_dir, state_dict=state_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="dynamic", choices=["dynamic", "static"], help="Training mode")
    args = parser.parse_args()
    
    train(args.config, args.mode)
