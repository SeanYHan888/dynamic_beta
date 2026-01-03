import logging
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

logger = logging.getLogger("DPO_DATA")

ASSISTANT_TAG = "\n\nAssistant:"

# delete the \n at the beginning of the response
def strip_one_leading_newline(s): 
    return s[1:] if s.startswith("\n") else s

def split_prompt_and_response(input_text):
    """
    HH format: multi-turn text containing many "\n\nAssistant:".
    We take the LAST Assistant tag as the start of the final assistant response.

    Returns:
    prompt: everything up to and including the final "\n\nAssistant:"
    response: the assistant completion after that tag (no leading newline)
    
    """
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = input_text[:index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG):]
    response = strip_one_leading_newline(response)
    return prompt, response


def convert_to_triples(chosen_text, rejected_text):
    """
    convert one HH row into an explicit triplet:
      {prompt, chosen, rejected}

    """
    # get prompt and response from chosen_text
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)

    # assume the chosen and rejected prompts are same
    if not rejected_text.startswith(chosen_prompt):
        return None
    
    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt):])
    
    
    if len(chosen_prompt.strip()) == 0:
        return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0:
        return None
    
    return {"prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response}

# special token helpers
def get_bos_eos_lists(tokenizer):
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    bos = [bos_id] if bos_id is not None else []
    eos = [eos_id] if eos_id is not None else []
    return bos, eos

# process entire dataset, build hh dataset
def build_HH_dataset(ds):
    def map_fn(batch):
        prompts, chosens, rejecteds, valids = [], [], [], []
        for chosen_text, rejected_text in zip(batch["chosen"], batch["rejected"]):
            output = convert_to_triples(
                chosen_text=chosen_text,
                rejected_text=rejected_text,
            )
            if output is None:
                prompts.append("")
                chosens.append("")
                rejecteds.append("")
                valids.append(False)
            else:
                prompts.append(output["prompt"])
                chosens.append(output["chosen"])
                rejecteds.append(output["rejected"])
                valids.append(True)
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds, "is_valid": valids}

    ds_triple = ds.map(map_fn, batched=True, remove_columns=ds.column_names)
    ds_triple = ds_triple.filter(lambda x: x["is_valid"])
    ds_triple = ds_triple.remove_columns(["is_valid"])
    return ds_triple


def collate_fn(batch, tokenizer, max_len, truncate_prompt=False, debug_state=None):
    """
    Input: list of triplets:
      {prompt, chosen, rejected}

    Output tensors:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length

    """

    if tokenizer.pad_token_id is None:
        # decoder-only models often have no pad token; reuse eos as pad for batching
        tokenizer.pad_token = tokenizer.eos_token

    bos, eos = get_bos_eos_lists(tokenizer)
    bos_len = len(bos)
    eos_len = len(eos)

    chosen_input_ids_list = []
    rejected_input_ids_list = []
    chosen_attention_mask_list = []
    rejected_attention_mask_list = []
    prompt_lens = []
    max_prompt_lens = []

    for item in batch:
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

        if truncate_prompt:
            max_prompt_len = max_len - bos_len - max(len(chosen_ids), len(rejected_ids)) - eos_len
            if max_prompt_len < 0:
                max_prompt_len = 0
            prompt_ids = prompt_ids[-max_prompt_len:] if max_prompt_len > 0 else []
        else:
            max_prompt_len = max_len - bos_len - eos_len
            if max_prompt_len < 0:
                max_prompt_len = 0
            if len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[:max_prompt_len]

        max_response_len = max_len - bos_len - len(prompt_ids) - eos_len
        if max_response_len < 0:
            max_response_len = 0
        chosen_ids = chosen_ids[:max_response_len]
        rejected_ids = rejected_ids[:max_response_len]

        prompt_lens.append(bos_len + len(prompt_ids))
        max_prompt_lens.append(max_prompt_len)

        chosen_seq = bos + prompt_ids + chosen_ids + eos
        rejected_seq = bos + prompt_ids + rejected_ids + eos

        chosen_attention = [1] * len(chosen_seq)
        rejected_attention = [1] * len(rejected_seq)

        chosen_pad = max_len - len(chosen_seq)
        rejected_pad = max_len - len(rejected_seq)

        chosen_input_ids_list.append(chosen_seq + [tokenizer.pad_token_id] * chosen_pad)
        rejected_input_ids_list.append(rejected_seq + [tokenizer.pad_token_id] * rejected_pad)
        chosen_attention_mask_list.append(chosen_attention + [0] * chosen_pad)
        rejected_attention_mask_list.append(rejected_attention + [0] * rejected_pad)

    enc_chosen_input_ids = torch.tensor(chosen_input_ids_list, dtype=torch.long)
    enc_chosen_attention_mask = torch.tensor(chosen_attention_mask_list, dtype=torch.long)
    enc_rejected_input_ids = torch.tensor(rejected_input_ids_list, dtype=torch.long)
    enc_rejected_attention_mask = torch.tensor(rejected_attention_mask_list, dtype=torch.long)
    prompt_length = torch.tensor(prompt_lens, dtype=torch.long)
    if debug_state is not None and debug_state.get("print_left", 0) > 0:
        logger.info("data_debug prompt_len=%s", prompt_lens)
        logger.info("data_debug max_prompt_len=%s", max_prompt_lens)
        logger.info(
            "data_debug bos_id=%s eos_id=%s add_bos_token=%s",
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            getattr(tokenizer, "add_bos_token", None),
        )
        debug_state["print_left"] -= 1

    # Build labels: input_ids with prompt tokens masked to -100
    chosen_labels = enc_chosen_input_ids.clone()
    rejected_labels = enc_rejected_input_ids.clone()
    max_seq_len = chosen_labels.size(1)
    for i, pl in enumerate(prompt_length.tolist()):
        pl = min(pl, max_seq_len)
        chosen_labels[i, :pl] = -100
        rejected_labels[i, :pl] = -100

    # mask padding
    chosen_labels[enc_chosen_attention_mask == 0] = -100
    rejected_labels[enc_rejected_attention_mask == 0] = -100

    return {
        "chosen_input_ids": enc_chosen_input_ids,
        "chosen_attention_mask": enc_chosen_attention_mask,
        "chosen_labels": chosen_labels,

        "rejected_input_ids": enc_rejected_input_ids,
        "rejected_attention_mask": enc_rejected_attention_mask,
        "rejected_labels": rejected_labels,

        # optional: useful for debugging / analysis
        "prompt_length": prompt_length,
    }

# build train, eval dataloader
def build_train_val(config, tokenizer):
    raw_dataset = config['dataset']['dataset_name']
    split = config['dataset']['subset']
    val_ratio = float(config['dataset']['val_ratio'])
    seed = int(config['dataset']['seed'])
    max_len = int(config['dataset']['max_len'])
    min_response_tokens = int(config['dataset'].get('min_response_tokens', 0))
    batch_size = int(config['dpo_training']['batch_size'])

    ds = load_dataset(raw_dataset, split=split)
    ds_triple = build_HH_dataset(ds)
# bos shifting, llama tokenizer adds extra bos
    bos, _ = get_bos_eos_lists(tokenizer)
    bos_shift = len(bos)

    def add_prompt_len(batch):
        lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"]) + bos_shift
            for p in batch["prompt"]
        ]
        return {"prompt_len": lens}

    ds_triple = ds_triple.map(add_prompt_len, batched=True)
    max_prompt_len = max_len - min_response_tokens
    if max_prompt_len < 1:
        raise ValueError("min_response_tokens must be smaller than max_len.")
    ds_triple = ds_triple.filter(lambda x: x["prompt_len"] < max_prompt_len)
    ds_triple = ds_triple.remove_columns(["prompt_len"])

    split_ds = ds_triple.train_test_split(test_size=val_ratio, seed=seed)
    train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]

    truncate_prompt = bool(config['dataset'].get('truncate_prompt', False))
    debug_cfg = config.get("debug", {})
    data_debug_print = int(debug_cfg.get("print_batches", 0))
    debug_state = {"print_left": data_debug_print} if data_debug_print > 0 else None
    ds_collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_len=max_len,
        truncate_prompt=truncate_prompt,
        debug_state=debug_state,
    )

    # When using Accelerate, we do NOT need to manually use DistributedSampler.
    # Accelerate will automatically shard the DataLoader for us during `prepare`.
    # Using DistributedSampler + Accelerate causes double-sharding (half data).
    
    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=True, # Accelerate handles sampler creation
        collate_fn=ds_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds_collate,
        pin_memory=True,
    )

    # Return None for samplers as they are now handled implicitly or internally
    return train_loader, val_loader, None, None
