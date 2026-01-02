import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataset_process_hh import split_prompt_and_response
import os
import json
import random
import yaml
from tqdm import tqdm
import argparse

# load yaml config
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# sample N data for test
def sample(dataset, N, seed):
    random_num = random.Random(seed)
    index = list(range(len(dataset)))
    random_num.shuffle(index)
    index = index[:N]
    test = []
    for i in index:
        prompt, response = split_prompt_and_response(dataset[i]['chosen'])
        if prompt.strip() and response.strip():
            test.append({
                "id": i,
                "prompt": prompt,
                "response": response
            })

    return test

# generate responses
@torch.no_grad()
def generate(model, tokenizer, prompts, device, max_new_tokens, temperature, top_p, batch_size):
    model.eval()        
    output = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    for s in tqdm(
        range(0, len(prompts), batch_size),
        total=num_batches,
        desc="Generating",
        leave=True,
        ):
        batch_prompts = prompts[ s : s + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

        # decode only continuation
        for j in range(gen.size(0)):
            true_pl = int(enc["attention_mask"][j].sum().item())
            text = tokenizer.decode(gen[j, true_pl:], skip_special_tokens=True).strip()
            output.append(text)
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args=parser.parse_args()

    config = load_yaml_config(args.config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load policy and reference
    policy1_path = config['dpo_training']['save_dpo_dir']
    policy2_path = config['dpo_training']['save_dir']
    ref_name = config['ref_name']

    tok = AutoTokenizer.from_pretrained(ref_name)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    # dpo
    # policy1 = AutoModelForCausalLM.from_pretrained(policy1_path).to(device)
    # policy1.config.pad_token_id = tok.pad_token_id
    # policy1.eval()

    # our method
    policy2 = AutoModelForCausalLM.from_pretrained(policy2_path).to(device)
    policy2.config.pad_token_id = tok.pad_token_id
    policy2.eval()

    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.config.pad_token_id = tok.pad_token_id
    ref_model.requires_grad_(False)
    ref_model.eval()

    # load dataset and build sample test
    raw_test_ds_name = config['dataset']['dataset_name']
    split = config['test']['subset']
    test_num = int(config['test']['test_num'])
    seed = config['test']['seed']

    raw_test_ds = load_dataset(raw_test_ds_name, split=split)

    test_sample = sample(dataset=raw_test_ds, N=test_num, seed=seed)

    prompts = [p['prompt'] for p in test_sample]

    # output
    max_new_tokens = int(config['test']['max_new_tokens'])
    temperature = float(config['test']['temperature'])
    top_p = float(config['test']['top_p'])
    batch_size = int(config['test']['batch_size'])

    # policy1 output
    # policy1_outs = generate(policy1, tok, prompts, device, 
    #                        max_new_tokens=max_new_tokens, 
    #                        temperature=temperature, 
    #                        top_p=top_p,
    #                        batch_size=batch_size)
    
    # policy2 output
    policy2_outs = generate(policy2, tok, prompts, device, 
                           max_new_tokens=max_new_tokens, 
                           temperature=temperature, 
                           top_p=top_p,
                           batch_size=batch_size)
    
    ref_outs = generate(ref_model, tok, prompts, device, 
                           max_new_tokens=max_new_tokens, 
                           temperature=temperature, 
                           top_p=top_p,
                           batch_size=batch_size)
    
    # policy1 out
    # policy1_outs_path = config['test']['dpo_out_dir']
    # os.makedirs(os.path.dirname(policy1_outs_path) or ".", exist_ok=True)
    # with open(policy1_outs_path, "w", encoding="utf-8") as f:
    #     for it, pol in zip(test_sample, policy1_outs):
    #         rec = {
    #             "id": it["id"],
    #             "prompt": it["prompt"],
    #             "chosen_response": it["response"],
    #             "model_response": pol,
    #             "model_tag": "dpo",
    #             "gen": {
    #                 "max_new_tokens": max_new_tokens,
    #                 "temperature": temperature,
    #                 "top_p": top_p,
    #                 },
    #                 "seed": seed,
    #                 }
    #         f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # print("Saved policy:", policy1_outs_path)

    # policy2 out
    policy2_outs_path = config['test']['dynamic_dpo_out_dir']
    os.makedirs(os.path.dirname(policy2_outs_path) or ".", exist_ok=True)
    with open(policy2_outs_path, "w", encoding="utf-8") as f:
        for it, pol in zip(test_sample, policy2_outs):
            rec = {
                "id": it["id"],
                "prompt": it["prompt"],
                "chosen_response": it["response"],
                "model_response": pol,
                "model_tag": "dynamic_dpo",
                "gen": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    },
                    "seed": seed,
                    }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Saved policy:", policy2_outs_path)

    # ref out
    ref_outs_path = config['test']['ref_out_dir']
    os.makedirs(os.path.dirname(ref_outs_path) or ".", exist_ok=True)
    with open(ref_outs_path, "w", encoding="utf-8") as f:
        for it, ref in zip(test_sample, ref_outs):
            rec = {
                "id": it["id"],
                "prompt": it["prompt"],
                "chosen_response": it["response"],
                "model_response": ref,
                "model_tag": "ref",
                "gen": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    },
                    "seed": seed,
                    }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Saved ref:", ref_outs_path)


if __name__ == "__main__":
    main()