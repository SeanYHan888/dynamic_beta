import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import json
from typing import Tuple, Dict

# --- Log Probability Calculation ---

def compute_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probability of the labels given the logits.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:].clone()

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    per_token_log_prob = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
   
    return (per_token_log_prob * loss_mask).sum(-1)

def compute_batch_log_prob(batch: Dict[str, torch.Tensor], policy, ref_model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities for chosen and rejected responses for both policy and reference models.
    """
    # policy forward (needs grad)
    policy_chosen_logits = policy(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    ).logits

    policy_rejected_logits = policy(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    ).logits

    policy_chosen_log_prob = compute_log_prob(
        logits=policy_chosen_logits,
        labels=batch["chosen_labels"],
    )
    policy_rejected_log_prob = compute_log_prob(
        logits=policy_rejected_logits,
        labels=batch["rejected_labels"],
    )

    # ref forward (no grad)
    with torch.no_grad():
        ref_chosen_logits = ref_model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits

        ref_rejected_logits = ref_model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits

        ref_chosen_log_prob = compute_log_prob(
            logits=ref_chosen_logits,
            labels=batch["chosen_labels"],
        )
        ref_rejected_log_prob = compute_log_prob(
            logits=ref_rejected_logits,
            labels=batch["rejected_labels"],
        )

    return policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob


# --- DPO Loss & Margin ---

def margin_compute(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob):
    """
    Compute the margin: (log(pi(yw|x)) - log(pi(yl|x))) - (log(ref(yw|x)) - log(ref(yl|x)))
    """
    policy_diff = policy_chosen_log_prob - policy_rejected_log_prob
    ref_diff =  ref_chosen_log_prob - ref_rejected_log_prob
    model_margin = (policy_diff - ref_diff).detach()
    return model_margin

def dpo_loss(policy_chosen_log_prob, policy_rejected_log_prob, ref_chosen_log_prob, ref_rejected_log_prob, beta):
    """
    Compute the DPO loss.
    """
    chosen_log_prob = policy_chosen_log_prob - ref_chosen_log_prob
    rejected_log_prob = policy_rejected_log_prob - ref_rejected_log_prob

    # compute the loss
    loss = - F.logsigmoid(beta * (chosen_log_prob - rejected_log_prob))

    chosen_rewards = (beta * chosen_log_prob).detach()
    rejected_rewards = (beta * rejected_log_prob).detach()

    return loss, chosen_rewards, rejected_rewards


def compute_and_log_model_margin(model_margin, epoch_dir, epoch, step, JSONL_PATH, sample_size=0, save_npy=True):
    """
    Log model margins to a JSONL file and save raw margins as .npy.
    """
    # full array
    # using numpy to process, so only on cpu
    m = model_margin.detach().float().cpu().numpy()  
                
    npy_path = None
    if save_npy:
        # step: batch index
        npy_path = os.path.join(epoch_dir, f"step_{step:05d}.npy")
        np.save(npy_path, m)

    # 2) write a readable per-batch record to ONE jsonl file
    # summary stats
    # quantiles
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
                
    record = {
        "epoch": int(epoch),
        "step": int(step),
        "batch_size": int(m.shape[0]),
        "mean": float(m.mean()),
        "std": float(m.std(ddof=0)),
        "min": float(m.min()),
        "p10": float(p10),
        "median": float(p50),
        "p90": float(p90),
        "max": float(m.max()),
        "pos_frac": float((m > 0).mean()),
        "npy": npy_path,
    }
    if sample_size > 0:
        sample = m[: min(sample_size, m.shape[0])]
        record["sample"] = [float(x) for x in sample]

    # save in the jsonl file       
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- Stats & Risk Utilities ---

class WarmupQuantileAccumulator:
    """
    Accumulate margins during warmup and estimate initial quantile threshold (tau_0).
    We store warmup margins (optionally winsorized) and compute:
      tau_0 = quantile(margins, q)
    where typically q = 1 - delta, so that P(M >= tau) ~ delta.
    """
    def __init__(self, q):
        self.q = q
        self._buf: list[torch.Tensor] = []

    # margins: Tensor([batch_size]) for this warmup step
    @torch.no_grad()
    def update(self, batch_margins: torch.Tensor):
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return
        self._buf.append(t.cpu())
        
    def finalize(self):
        if len(self._buf) == 0:
            return 0.0
        all_m = torch.cat(self._buf, dim=0)
        tau0 = torch.quantile(all_m, self.q).item()
        return float(tau0)  
    
class EMAUpdate:
    def __init__(self, tau_0, q, momentum):
        self.tau = tau_0
        self.q = q
        self.lam = momentum

    # threshold tau equations    
    def update_tau(self, batch_margins: torch.Tensor):
        t = batch_margins.detach().float().view(-1)
        if t.numel() == 0:
            return self.tau
        batch_tau = torch.quantile(t, self.q).item()
        self.tau = (1.0 - self.lam) * self.tau + self.lam * batch_tau
        return self.tau

    def update_tau_from_value(self, batch_tau: float):
        self.tau = (1.0 - self.lam) * self.tau + self.lam * float(batch_tau)
        return self.tau

def empirical_over_threshold_proportion(margins: torch.Tensor, threshold):
    return (margins >= threshold).float().mean().item()

def risk_test(p_hat, eplison_0, delta, n):
    eplison = math.sqrt(math.log(1 / eplison_0) / (2 * n))
    delta_prime = delta + eplison
    return p_hat > delta_prime, eplison, delta_prime

def update_beta(beta, p_hat, delta_prime, eplison, alpha, gamma, beta_min, beta_max):
    u_k = (p_hat - delta_prime) / eplison
    s_k = math.tanh(gamma * u_k)
    beta_new = beta * math.exp(alpha * s_k)
    beta_new = max(beta_min, min(beta_new, beta_max))
    return beta_new, u_k, s_k, alpha
