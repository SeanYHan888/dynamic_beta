import torch
from dpo_loss import compute_log_prob

def compute_batch_log_prob(batch, policy, ref_model):
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
