import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# Config values
dataset_name = "Anthropic/hh-rlhf"
subset = "train[:20%]"
max_len = 512
policy_name = "meta-llama/Llama-3.2-1B"

print(f"\n=== Debugging Dataset Processing ===")
print(f"Dataset: {dataset_name} | Subset: {subset}")
print(f"Max Len: {max_len} | Tokenizer: {policy_name}")

print(f"\n1. Loading Dataset...")
ds = load_dataset(dataset_name, split=subset)
initial_count = len(ds)
print(f"   Initial Size: {initial_count:,} records")

# --- tokenizer setup ---
try:
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    print(f"   Loaded tokenizer: {policy_name}")
except OSError:
    print(f"   [WARNING] Could not access {policy_name} (likely gated). Falling back to 'gpt2' for approximation.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
bos_shift = 1 if (getattr(tokenizer, "add_bos_token", False) and tokenizer.bos_token_id is not None) else 0

# --- Structure Filter ---
ASSISTANT_TAG = "\n\nAssistant:"
def strip_one_leading_newline(s): 
    return s[1:] if s.startswith("\n") else s

def split_prompt_and_response(input_text):
    input_text = str(input_text).replace("\r\n", "\n").replace("\r", "\n")
    index = input_text.rfind(ASSISTANT_TAG)
    if index < 0:
        return None, None
    prompt = input_text[:index + len(ASSISTANT_TAG)]
    response = input_text[index + len(ASSISTANT_TAG):]
    response = strip_one_leading_newline(response)
    return prompt, response

def convert_to_triples(chosen_text, rejected_text):
    chosen_prompt, chosen_response = split_prompt_and_response(chosen_text)
    if chosen_prompt is None: return None
    
    if not rejected_text.startswith(chosen_prompt):
        return None
    
    rejected_response = strip_one_leading_newline(rejected_text[len(chosen_prompt):])
    
    if len(chosen_prompt.strip()) == 0: return None
    if len(chosen_response.strip()) == 0 or len(rejected_response.strip()) == 0: return None
    
    return {"prompt": chosen_prompt}

print(f"\n2. Applying Structure Filter (HH formatting)...")
def filter_structure(batch):
    valids = []
    for chosen, rejected in zip(batch["chosen"], batch["rejected"]):
        res = convert_to_triples(chosen, rejected)
        valids.append(res is not None)
    return {"is_valid": valids}

ds_triple = ds.map(filter_structure, batched=True, desc="Checking structure")
ds_triple = ds_triple.filter(lambda x: x["is_valid"], desc="Filtering invalid")
structure_count = len(ds_triple)
dropped_structure = initial_count - structure_count
print(f"   Valid Records: {structure_count:,}")
print(f"   Dropped: {dropped_structure:,} ({dropped_structure/initial_count:.1%})")

# --- Length Filter ---
print(f"\n3. Applying Length Filter (Prompt < {max_len} tokens)...")

# We need to extract prompts first to check length exactly as code does
def map_prompts(batch):
    prompts = []
    for chosen in batch["chosen"]:
        p, _ = split_prompt_and_response(chosen)
        prompts.append(p)
    return {"prompt": prompts}

ds_triple = ds_triple.map(map_prompts, batched=True, desc="Extracting prompts")

def check_len(batch):
    lens = []
    for p in batch["prompt"]:
        l = len(tokenizer(p, add_special_tokens=False)["input_ids"]) + bos_shift
        lens.append(l)
    return {"prompt_len": lens}

ds_triple = ds_triple.map(check_len, batched=True, desc="Tokenizing prompts")
ds_triple = ds_triple.filter(lambda x: x["prompt_len"] < max_len, desc="Filtering too long")
len_count = len(ds_triple)
dropped_len = structure_count - len_count
print(f"   Records Fitting Length: {len_count:,}")
print(f"   Dropped: {dropped_len:,} ({dropped_len/structure_count:.1%})")

# --- Final Calculation ---
print(f"\n4. Training Set Calculation")
val_ratio = 0.1
train_size = int(len_count * (1 - val_ratio))
val_size = len_count - train_size
print(f"   Total Available: {len_count:,}")
print(f"   Validation ({val_ratio:.0%}): {val_size:,}")
print(f"   Training Data: {train_size:,}")

print(f"\n5. Step Count for 2 GPUs")
batch_size = 16
num_gpus = 2
global_batch_size = batch_size * num_gpus
steps = train_size // global_batch_size

print(f"   Batch Size (per GPU): {batch_size}")
print(f"   Global Batch Size: {global_batch_size}")
print(f"   Calculation: {train_size} / {global_batch_size} = {steps:.2f}")
print(f"   Expected Steps: {int(steps)}")
