"""Validate model/corpus_raw.json dialogues for semantic sense using the
Anthropic API, then write model/corpus_validated.json containing only the
dialogues Claude judged VALID (same format as corpus_raw.json).

NOTE ON MODEL: "claude-haiku-3-5" is a retired model ID no longer served by
the API. This script uses the current cheapest/fastest tier instead,
claude-haiku-4-5 ($1/$5 per 1M tokens).
"""

import json
import random
import time
from collections import Counter
from pathlib import Path

import anthropic

MODEL_DIR = Path(__file__).resolve().parent
INPUT_PATH = MODEL_DIR / "corpus_raw.json"
OUTPUT_PATH = MODEL_DIR / "corpus_validated.json"

MODEL = "claude-haiku-4-5"
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 2.0
MAX_TOKENS = 60
MAX_RETRIES = 5

SYSTEM_PROMPT = """You are validating dialogues for a primitive language model. \
Vocabulary is limited to ~120 basic English words. \
A dialogue is VALID if:
1. Each response logically follows from the previous turn
2. No token is used in a semantically impossible way \
(e.g. fire cannot run, stone cannot eat)
3. The exchange makes sense as a primitive conversation

Reply with only: VALID or INVALID: <one short reason>"""


def dialogue_to_text(dialogue):
    lines = [f"{turn['speaker']}: {' '.join(turn['tokens'])}" for turn in dialogue["turns"]]
    return "\n".join(lines)


def parse_verdict(text):
    stripped = text.strip()
    upper = stripped.upper()
    if upper.startswith("VALID"):
        return True, None
    if upper.startswith("INVALID"):
        reason = stripped.split(":", 1)[1].strip() if ":" in stripped else "no reason given"
        return False, reason
    # Model didn't follow the requested format -- treat as invalid but keep
    # the raw text so it's visible in the rejection-reason summary.
    return False, f"unparseable response: {stripped[:80]!r}"


def classify_dialogue(client, dialogue_text):
    """Call the API for one dialogue with retry/backoff.
    Returns (is_valid, reason_or_None)."""
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": dialogue_text}],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            return parse_verdict(text)
        except anthropic.NotFoundError:
            raise  # bad model ID / endpoint -- retrying can't fix this
        except anthropic.RateLimitError as e:
            last_exception = e
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
        except anthropic.APIStatusError as e:
            if e.status_code < 500:
                raise  # non-retryable client error (bad request, auth, ...)
            last_exception = e
            delay = min(2 ** attempt, 30)
        except anthropic.APIConnectionError as e:
            last_exception = e
            delay = min(2 ** attempt, 30)

        delay += random.uniform(0, 1)
        print(f"    retry {attempt + 1}/{MAX_RETRIES} after {type(last_exception).__name__}, waiting {delay:.1f}s...")
        time.sleep(delay)

    raise last_exception


def main(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    dialogues = corpus["dialogues"]
    total = len(dialogues)

    client = anthropic.Anthropic()

    valid_dialogues = []
    reason_counter = Counter()
    valid_count = 0
    invalid_count = 0

    for i, dialogue in enumerate(dialogues, start=1):
        text = dialogue_to_text(dialogue)
        is_valid, reason = classify_dialogue(client, text)

        if is_valid:
            valid_count += 1
            valid_dialogues.append(dialogue)
        else:
            invalid_count += 1
            reason_counter[reason] += 1

        if i % BATCH_SIZE == 0 or i == total:
            print(f"Progress: {i}/{total} checked  (valid={valid_count}, invalid={invalid_count})")
            if i % BATCH_SIZE == 0 and i != total:
                time.sleep(BATCH_DELAY_SECONDS)

    situations = sorted({d["situation"] for d in valid_dialogues})
    situation_valid_counts = Counter(d["situation"] for d in valid_dialogues)
    validated_corpus = {
        "meta": {"total": len(valid_dialogues), "situations": len(situations)},
        "dialogues": valid_dialogues,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validated_corpus, f, indent=2)

    print()
    print(f"Total checked: {total}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {invalid_count}")
    print(f"Wrote {len(valid_dialogues)} valid dialogues to {output_path}")
    print()
    print("Per-situation valid counts:")
    for situation in situations:
        print(f"  {situation:20s} {situation_valid_counts[situation]}")
    print()
    print("Most common rejection reasons:")
    for reason, count in reason_counter.most_common(10):
        print(f"  {count:4d}  {reason}")


if __name__ == "__main__":
    import sys

    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_PATH
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_PATH
    main(in_path, out_path)
