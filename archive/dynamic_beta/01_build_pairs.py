# 01_build_pairs.py
import argparse, json, os, random
from typing import Dict, Any, Iterable, List

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_pairs(in_path: str, out_path: str, seed: int):
    rng = random.Random(seed)
    rows = []
    for ex in read_jsonl(in_path):
        prompt = ex.get("prompt","")
        chosen = ex.get("chosen_response","")
        model_resp = ex.get("model_response","")
        if not prompt.strip() or not chosen.strip() or not model_resp.strip():
            continue

        # Randomize A/B to avoid position bias
        if rng.random() < 0.5:
            A, B = model_resp, chosen
            a_is_model = True
        else:
            A, B = chosen, model_resp
            a_is_model = False

        rows.append({
            "id": ex.get("id"),
            "model_tag": ex.get("model_tag"),   # ref / dpo / dynamic_dpo
            "prompt": prompt,
            "response_a": A,
            "response_b": B,
            "a_is_model": a_is_model,
            # keep originals for debugging
            "chosen_response": chosen,
            "model_response": model_resp,
            "gen": ex.get("gen", {}),
            "seed": ex.get("seed", None),
        })

    write_jsonl(out_path, rows)
    print(f"[OK] {len(rows)} pairs -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, required=True, help="your generated output jsonl")
    ap.add_argument("--out_pairs", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build_pairs(args.in_jsonl, args.out_pairs, args.seed)

if __name__ == "__main__":
    main()
