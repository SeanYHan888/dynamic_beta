# 03_summarize_results.py
import argparse
import json
import os
from typing import Dict, Any, Iterable


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=str, required=True,
                    help="comma-separated paths, e.g. a.jsonl,b.jsonl,c.jsonl")
    ap.add_argument("--out_json", type=str, required=True)
    args = ap.parse_args()

    files = [x.strip() for x in args.judgments.split(",") if x.strip()]

    stats = {}  # model_tag -> counters

    for fp in files:
        for r in read_jsonl(fp):
            tag = r.get("model_tag", "unknown")
            stats.setdefault(tag, {"total": 0, "valid": 0, "win": 0, "loss": 0, "tie": 0, "invalid": 0})
            st = stats[tag]
            st["total"] += 1

            w = (r.get("winner") or "invalid").lower()
            if w == "invalid":
                st["invalid"] += 1
                continue
            if w == "tie":
                st["tie"] += 1
                continue

            # valid: a/b
            a_is_model = bool(r.get("a_is_model"))
            is_win = (w == "a" and a_is_model) or (w == "b" and (not a_is_model))
            st["valid"] += 1
            if is_win:
                st["win"] += 1
            else:
                st["loss"] += 1

    summary = {}
    for tag, st in stats.items():
        denom = max(1, st["win"] + st["loss"])
        summary[tag] = {
            **st,
            "win_rate": st["win"] / denom,
            "tie_rate": st["tie"] / max(1, st["total"]),
            "invalid_rate": st["invalid"] / max(1, st["total"]),
        }

    print("=== WinRate (wins/(wins+losses)) ===")
    for tag, s in sorted(summary.items()):
        print(
            f"{tag:12s}  win_rate={s['win_rate']:.4f} "
            f"(win={s['win']}, loss={s['loss']}, valid={s['valid']})  "
            f"tie={s['tie']} invalid={s['invalid']} total={s['total']}"
        )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {args.out_json}")


if __name__ == "__main__":
    main()
