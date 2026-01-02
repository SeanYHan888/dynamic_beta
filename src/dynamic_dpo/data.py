import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial


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


def collate_fn(batch, tokenizer, max_len):
    """
    Input: list of triplets:
      {prompt, chosen, rejected}

    Output tensors:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length

    """

    chosen_txt, rejected_txt = [], []
    prompt_lens = []

    if tokenizer.pad_token_id is None:
        # decoder-only models often have no pad token; reuse eos as pad for batching
        tokenizer.pad_token = tokenizer.eos_token

    for item in batch:
        prompt = str(item['prompt'])
        chosen = str(item['chosen'])
        rejected = str(item['rejected'])

        chosen_txt.append(prompt + chosen)
        rejected_txt.append(prompt + rejected)

        # prompt_length without special tokens
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_lens.append(len(prompt_ids))

    # chosen, rejected tokens
    enc_chosen = tokenizer(
        chosen_txt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    enc_rejected = tokenizer(
        rejected_txt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # If tokenizer adds a BOS token at position 0, prompt length should include it.
    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.size(1) > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0
    prompt_length = torch.tensor([pl + bos_shift for pl in prompt_lens], dtype=torch.long)

    max_seq_len = enc_chosen.input_ids.size(1)
    # If the tokenizer truncated prompts (which happens if truncation=True), we must clamp prompt_length
    prompt_length = torch.clamp(prompt_length, max=max_seq_len - 1) 
    
    keep_mask = prompt_length < max_seq_len
    if not keep_mask.all():
         # This should now be impossible due to clamp, but keeping as safeguard
        raise ValueError("Batch contains prompts that consume full sequence length.")
    enc_chosen_input_ids = enc_chosen.input_ids
    enc_chosen_attention_mask = enc_chosen.attention_mask
    enc_rejected_input_ids = enc_rejected.input_ids
    enc_rejected_attention_mask = enc_rejected.attention_mask

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
    batch_size = int(config['dpo_training']['batch_size'])

    ds = load_dataset(raw_dataset, split=split)
    ds_triple = build_HH_dataset(ds)

    bos_shift = 1 if (getattr(tokenizer, "add_bos_token", False) and tokenizer.bos_token_id is not None) else 0

    def add_prompt_len(batch):
        lens = [
            len(tokenizer(p, add_special_tokens=False)["input_ids"]) + bos_shift
            for p in batch["prompt"]
        ]
        return {"prompt_len": lens}

    ds_triple = ds_triple.map(add_prompt_len, batched=True)
    ds_triple = ds_triple.filter(lambda x: x["prompt_len"] < max_len)
    ds_triple = ds_triple.remove_columns(["prompt_len"])

    split_ds = ds_triple.train_test_split(test_size=val_ratio, seed=seed)
    train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]

    ds_collate = partial(collate_fn, tokenizer=tokenizer, max_len=max_len)

    # When using Accelerate, we do NOT need to manually use DistributedSampler.
    # Accelerate will automatically shard the DataLoader for us during `prepare`.
    # Using DistributedSampler + Accelerate causes double-sharding (half data).
    
    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=True, # Accelerate handles sampler creation
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    # Return None for samplers as they are now handled implicitly or internally
    return train_loader, val_loader, None, None




