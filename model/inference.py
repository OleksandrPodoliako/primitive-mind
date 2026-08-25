"""Interactive CLI for manually testing the Primitive Mind model.

Usage: python model/inference.py

Loads the trained checkpoint + tokenizer, then runs a REPL where you type
HUM turns (space-separated vocabulary words) and the model generates a CRO
reply token by token, followed by the top-5 next-token logits -- showing
what the model actually considered, to confirm it's predicting rather than
replaying memorized text.

Context accumulates across turns exactly like a training example -- a full
[HUM] [CRO] [HUM] [CRO] 4-turn window -- and auto-resets after the 4th turn,
same as the corpus_final.json dialogue shape the model was trained on.
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))  # so `python inference.py` works from any cwd
from train import PrimitiveMindGPT, HUM_ID, CRO_ID  # noqa: E402

EMBEDDINGS_PATH = MODEL_DIR / "embeddings.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CHECKPOINT_PATH = MODEL_DIR / "primitive_mind.pt"

MAX_TURNS = 4  # matches the [HUM][CRO][HUM][CRO] shape the model was trained on
MAX_INPUT_WORDS = 8
GEN_DELAY_SECONDS = 0.15  # small pause per generated token, for a visible "typing" effect
TOP_K = 5
TEMPERATURE = 0.8
REPETITION_PENALTY = 1.3  # logit /= 1.3 for a token already emitted this turn
EOS_TOKENS = ("[PAD]", "[HUM]")  # stop when the top predicted token is one of these
EOS_MAX_LENGTH = 4  # stop once the response reaches this many tokens


def load_model():
    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    embeddings_data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    vocab_words = {
        w for cat, entries in embeddings_data.items() if cat != "_meta" for w in entries
    }
    expected = tokenizer["vocab_size"] - 3
    if len(vocab_words) != expected:
        print(f"Warning: embeddings.json has {len(vocab_words)} words but tokenizer expects {expected}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = PrimitiveMindGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def top_k_logits(model, context_ids, k=TOP_K):
    """Top-k next-token probabilities at the current end of context (i.e.
    right after the [CRO] tag, before any token has been sampled)."""
    idx = torch.tensor([context_ids], dtype=torch.long)
    logits = model(idx)[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=min(k, probs.shape[-1]))
    return topk.values.tolist(), topk.indices.tolist()


@torch.no_grad()
def generate_turn(model, context_ids, id_to_token, temperature=TEMPERATURE):
    """Sample the CRO reply one token at a time, printing each as it's
    generated. At every step: apply a repetition penalty to tokens already
    emitted this turn (divide their logit by REPETITION_PENALTY -- this is
    what stops degenerate loops like "now now now now"), then stop if the
    model's top predicted token is [PAD]/[HUM] or EOS_MAX_LENGTH tokens have
    been generated; otherwise sample the next token at `temperature`.
    Returns (generated_words, updated_context_ids)."""
    generated_words = []
    generated_ids = []
    idx = torch.tensor([context_ids], dtype=torch.long)

    print("CRO > ", end="", flush=True)
    while True:
        if len(generated_ids) >= EOS_MAX_LENGTH:
            break

        logits = model(idx)[0, -1, :].clone()
        for tid in set(generated_ids):
            logits[tid] = logits[tid] / REPETITION_PENALTY

        top_id = int(torch.argmax(logits).item())
        if id_to_token[str(top_id)] in EOS_TOKENS:
            break

        probs = F.softmax(logits / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        word = id_to_token[str(next_id)]

        print(word, end=" ", flush=True)
        time.sleep(GEN_DELAY_SECONDS)
        generated_words.append(word)
        generated_ids.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

    if not generated_words:
        print("(nothing)", end="")
    print()
    return generated_words, idx[0].tolist()


def main():
    model, tokenizer = load_model()
    token_to_id = tokenizer["token_to_id"]
    id_to_token = tokenizer["id_to_token"]
    vocab_size = tokenizer["vocab_size"]

    print("Primitive Mind ready. Type tokens separated by spaces.")
    print(f'({vocab_size - 3}-word vocabulary -- type "reset" to clear context, "quit" to exit)\n')

    context_ids = []
    turn_count = 0

    while True:
        try:
            raw = input("HUM > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() == "quit":
            print("Bye.")
            break
        if raw.lower() == "reset":
            context_ids = []
            turn_count = 0
            print("(context reset)\n")
            continue

        words = raw.lower().split()
        if len(words) > MAX_INPUT_WORDS:
            print(f"(too many words -- max {MAX_INPUT_WORDS})\n")
            continue

        unknown = [w for w in words if w.startswith("[") or w not in token_to_id]
        if unknown:
            print(f"(unknown word(s): {unknown} -- not in the {vocab_size - 3}-word vocabulary)\n")
            continue

        context_ids += [HUM_ID] + [token_to_id[w] for w in words]
        turn_count += 1
        context_ids += [CRO_ID]

        # Snapshot the top-5 logits before generation -- what the model
        # actually considered, not just what it happened to sample.
        probs, ids = top_k_logits(model, context_ids)
        logits_str = " ".join(f"{id_to_token[str(i)]}({p * 100:.0f}%)" for p, i in zip(probs, ids))

        _, context_ids = generate_turn(model, context_ids, id_to_token)
        turn_count += 1

        print(f"TOP 5 LOGITS: {logits_str}\n")

        if turn_count >= MAX_TURNS:
            context_ids = []
            turn_count = 0
            print("(4 turns reached -- context reset)\n")


if __name__ == "__main__":
    main()
