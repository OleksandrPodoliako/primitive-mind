"""Merge model/corpus_validated.json (Claude-validated dialogues from the
original seed set) with model/corpus_extra_validated.json (the regenerated
danger / fire / shelter / us-vs-them dialogues from the expanded 15-seed
pools, also Claude-validated), deduplicating exact-match dialogues, and
write model/corpus_final.json.
"""

import json
from collections import Counter
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
VALIDATED_PATH = MODEL_DIR / "corpus_validated.json"
EXTRA_PATH = MODEL_DIR / "corpus_extra_validated.json"
OUTPUT_PATH = MODEL_DIR / "corpus_final.json"


def dialogue_signature(dialogue):
    """Exact-match signature: situation + the full 4-turn token sequence."""
    return (
        dialogue["situation"],
        tuple(tuple(turn["tokens"]) for turn in dialogue["turns"]),
    )


def main():
    with open(VALIDATED_PATH, "r", encoding="utf-8") as f:
        validated = json.load(f)["dialogues"]
    with open(EXTRA_PATH, "r", encoding="utf-8") as f:
        extra = json.load(f)["dialogues"]

    merged = []
    seen = set()
    duplicates = 0

    for dialogue in validated + extra:
        sig = dialogue_signature(dialogue)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        merged.append(dialogue)

    situations = sorted({d["situation"] for d in merged})
    final_corpus = {
        "meta": {"total": len(merged), "situations": len(situations)},
        "dialogues": merged,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_corpus, f, indent=2)

    print(f"corpus_validated.json:       {len(validated)} dialogues")
    print(f"corpus_extra_validated.json: {len(extra)} dialogues")
    print(f"Exact duplicates dropped: {duplicates}")
    print(f"Wrote {len(merged)} dialogues to {OUTPUT_PATH}")
    print()
    print("Final per-situation counts:")
    counts = Counter(d["situation"] for d in merged)
    for situation, count in sorted(counts.items()):
        print(f"  {situation:20s} {count}")


if __name__ == "__main__":
    main()
