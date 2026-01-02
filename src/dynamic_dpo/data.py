from functools import partial
from typing import Any, Dict, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ASSISTANT_TAG = "\n\nAssistant:"


def strip_one_leading_newline(text: str) -> str:
    return text[1:] if text.startswith("\n") else text


def split_prompt_and_response(input_text: str) -> Tuple[str, str]:
    """
    HH format: multi-turn text containing many "\n\nAssistant:".
    The last Assistant tag marks the start of the final assistant response.
    """
    normalized = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = normalized.rfind(ASSISTANT_TAG)
    if index < 0:
        raise ValueError("No '\\n\\nAssistant:' tag found in HH input.")
    prompt = normalized[: index + len(ASSISTANT_TAG)]
    response = strip_one_leading_newline(normalized[index + len(ASSISTANT_TAG) :])
    return prompt, response


def _row_to_triplet(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        chosen_text = row["chosen"]
        rejected_text = row["rejected"]
        prompt, chosen_response = split_prompt_and_response(chosen_text)
        if not str(rejected_text).startswith(prompt):
            return {"prompt": "", "chosen": "", "rejected": "", "is_valid": False}
        rejected_response = strip_one_leading_newline(str(rejected_text)[len(prompt) :])
        if not prompt.strip() or not chosen_response.strip() or not rejected_response.strip():
            return {"prompt": "", "chosen": "", "rejected": "", "is_valid": False}
        return {
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
            "is_valid": True,
        }
    except Exception:
        return {"prompt": "", "chosen": "", "rejected": "", "is_valid": False}


def build_hh_dataset(ds: Dataset) -> Dataset:
    ds = ds.map(_row_to_triplet)
    ds = ds.filter(lambda row: row["is_valid"])
    keep_cols = {"prompt", "chosen", "rejected"}
    drop_cols = [col for col in ds.column_names if col not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


def collate_fn(batch, tokenizer, max_len: int) -> Dict[str, torch.Tensor]:
    """
    Input: list of triplets {prompt, chosen, rejected}
    Output:
      chosen_input_ids, chosen_attention_mask, chosen_labels
      rejected_input_ids, rejected_attention_mask, rejected_labels
      prompt_length
    """
    chosen_txt, rejected_txt = [], []
    prompt_lens = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for item in batch:
        prompt = str(item["prompt"])
        chosen = str(item["chosen"])
        rejected = str(item["rejected"])

        chosen_txt.append(prompt + chosen)
        rejected_txt.append(prompt + rejected)

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        prompt_lens.append(len(prompt_ids))

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

    has_bos = (
        tokenizer.bos_token_id is not None
        and enc_chosen.input_ids.size(1) > 0
        and enc_chosen.input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    bos_shift = 1 if has_bos else 0
    prompt_length = torch.tensor([pl + bos_shift for pl in prompt_lens], dtype=torch.long)

    chosen_labels = enc_chosen.input_ids.clone()
    rejected_labels = enc_rejected.input_ids.clone()

    max_seq_len = chosen_labels.size(1)
    for i, pl in enumerate(prompt_length.tolist()):
        pl = min(pl, max_seq_len)
        chosen_labels[i, :pl] = -100
        rejected_labels[i, :pl] = -100

    chosen_labels[enc_chosen.attention_mask == 0] = -100
    rejected_labels[enc_rejected.attention_mask == 0] = -100

    return {
        "chosen_input_ids": enc_chosen.input_ids,
        "chosen_attention_mask": enc_chosen.attention_mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": enc_rejected.input_ids,
        "rejected_attention_mask": enc_rejected.attention_mask,
        "rejected_labels": rejected_labels,
        "prompt_length": prompt_length,
    }


def _distributed_samplers(config: Dict[str, Any], dataset, shuffle: bool, seed: int):
    dist_cfg = config.get("distributed", {})
    world_size = int(dist_cfg.get("world_size", 1))
    rank = int(dist_cfg.get("rank", 0))
    if world_size <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )


def build_train_val(config: Dict[str, Any], tokenizer):
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("dpo_training", {})

    raw_dataset = dataset_cfg.get("dataset_name", "Anthropic/hh-rlhf")
    split = dataset_cfg.get("subset", dataset_cfg.get("split", "train"))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.1))
    seed = int(dataset_cfg.get("seed", 42))
    max_len = int(dataset_cfg.get("max_len", 512))
    batch_size = int(training_cfg.get("batch_size", 1))
    num_workers = int(dataset_cfg.get("num_workers", training_cfg.get("num_workers", 0)))
    drop_last = bool(training_cfg.get("drop_last", False))

    ds = load_dataset(raw_dataset, split=split)
    ds_triple = build_hh_dataset(ds)

    if 0.0 < val_ratio < 1.0:
        split_ds = ds_triple.train_test_split(test_size=val_ratio, seed=seed)
        train_ds_raw, val_ds_raw = split_ds["train"], split_ds["test"]
    else:
        train_ds_raw = ds_triple
        val_ds_raw = ds_triple.select([])

    collate = partial(collate_fn, tokenizer=tokenizer, max_len=max_len)

    train_sampler = _distributed_samplers(config, train_ds_raw, shuffle=True, seed=seed)
    val_sampler = _distributed_samplers(config, val_ds_raw, shuffle=False, seed=seed)

    train_loader = DataLoader(
        train_ds_raw,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds_raw,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def build_eval_loader(config: Dict[str, Any], tokenizer):
    dataset_cfg = config.get("dataset", {})
    test_cfg = config.get("test", {})
    training_cfg = config.get("dpo_training", {})

    raw_dataset = test_cfg.get("dataset_name", dataset_cfg.get("dataset_name", "Anthropic/hh-rlhf"))
    split = test_cfg.get("subset", "test")
    seed = int(test_cfg.get("seed", dataset_cfg.get("seed", 42)))
    max_len = int(test_cfg.get("max_len", dataset_cfg.get("max_len", 512)))
    batch_size = int(test_cfg.get("batch_size", training_cfg.get("batch_size", 1)))
    num_workers = int(test_cfg.get("num_workers", dataset_cfg.get("num_workers", training_cfg.get("num_workers", 0))))
    limit = int(test_cfg.get("test_num", 0))

    ds = load_dataset(raw_dataset, split=split)
    ds_triple = build_hh_dataset(ds)
    if limit > 0:
        ds_triple = ds_triple.select(range(min(limit, len(ds_triple))))

    collate = partial(collate_fn, tokenizer=tokenizer, max_len=max_len)
    eval_sampler = _distributed_samplers(config, ds_triple, shuffle=False, seed=seed)

    eval_loader = DataLoader(
        ds_triple,
        batch_size=batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        drop_last=False,
    )

    return eval_loader, eval_sampler
