"""nanoGPT-style decoder-only transformer trained on model/corpus_final.json.

The corpus mixes 4-turn and 6-turn dialogues, so CONTEXT_LEN is set to 96 to
comfortably cover the longer 6-turn sequences (worst case: 6 turns * (1 tag
+ 4 tokens) = 30 tokens, well under 96 -- the rest is [PAD]).

Each dialogue is flattened to a single token sequence:
    [HUM] t1 t2 t3 [CRO] t4 t5 ... [HUM] tn [CRO] tn+1
padded to CONTEXT_LEN, with the next-token loss masked to CRO-turn content
tokens only -- the model never gets gradient signal for predicting [HUM]
turns, the [HUM]/[CRO] tags, or [PAD], so it only ever learns to produce
CRO's replies.

Outputs:
  model/tokenizer.json      word/tag -> id mapping
  model/primitive_mind.pt   trained weights + config
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MODEL_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.json"
CORPUS_PATH = MODEL_DIR / "corpus_final.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CHECKPOINT_PATH = MODEL_DIR / "primitive_mind.pt"

SEED = 42

# --- model spec ------------------------------------------------------------
N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128
D_FF = 512
DROPOUT = 0.1
CONTEXT_LEN = 96  # covers the longest dialogue (6 turns) with headroom

# --- training spec ----------------------------------------------------------
LR = 3e-4
LR_MIN = 3e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 32
EPOCHS = 50
PRINT_EVERY = 5

SAMPLE_SITUATIONS = ("danger", "food", "water", "spirits-unknown", "us-vs-them")

PAD_ID, HUM_ID, CRO_ID = 0, 1, 2


# ===========================================================================
# Tokenizer
# ===========================================================================

def build_tokenizer():
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_to_id = {"[PAD]": PAD_ID, "[HUM]": HUM_ID, "[CRO]": CRO_ID}
    next_id = 3
    for category, entries in data.items():
        if category == "_meta":
            continue
        for word in entries:
            if word not in token_to_id:
                token_to_id[word] = next_id
                next_id += 1

    tokenizer = {
        "vocab_size": len(token_to_id),
        "special_tokens": {"[PAD]": PAD_ID, "[HUM]": HUM_ID, "[CRO]": CRO_ID},
        "token_to_id": token_to_id,
        "id_to_token": {str(i): tok for tok, i in token_to_id.items()},
    }

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, indent=2)

    return tokenizer


# ===========================================================================
# Dataset
# ===========================================================================

def build_examples(corpus, token_to_id):
    """Each example: (ids, is_cro_content), both length CONTEXT_LEN.
    is_cro_content[i] marks whether ids[i] is a content token spoken by CRO --
    this drives the next-token loss mask in the training loop below.
    Dialogue turn count varies (4 or 6 in this corpus); only the padding
    amount changes per-example."""
    examples = []
    for dialogue in corpus["dialogues"]:
        ids, is_cro = [], []
        for turn in dialogue["turns"]:
            is_cro_turn = turn["speaker"] == "CRO"
            ids.append(CRO_ID if is_cro_turn else HUM_ID)
            is_cro.append(False)  # the tag itself is never a loss target
            for tok in turn["tokens"]:
                ids.append(token_to_id[tok])
                is_cro.append(is_cro_turn)

        if len(ids) > CONTEXT_LEN:
            raise ValueError(f"Dialogue exceeds CONTEXT_LEN={CONTEXT_LEN}: {len(ids)} tokens")

        pad_len = CONTEXT_LEN - len(ids)
        ids += [PAD_ID] * pad_len
        is_cro += [False] * pad_len
        examples.append((ids, is_cro))
    return examples


class DialogueDataset(Dataset):
    def __init__(self, examples):
        self.ids = torch.tensor([e[0] for e in examples], dtype=torch.long)
        self.is_cro = torch.tensor([e[1] for e in examples], dtype=torch.bool)

    def __len__(self):
        return self.ids.size(0)

    def __getitem__(self, idx):
        return self.ids[idx], self.is_cro[idx]


# ===========================================================================
# Model (nanoGPT-style decoder-only transformer)
# ===========================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, nh, T, hd)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PrimitiveMindGPT(nn.Module):
    def __init__(self, vocab_size, context_len=CONTEXT_LEN, d_model=D_MODEL,
                 n_heads=N_HEADS, d_ff=D_FF, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.context_len = context_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying (nanoGPT-style)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.context_len, f"sequence length {T} exceeds context_len {self.context_len}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ===========================================================================
# Training
# ===========================================================================

def masked_loss(logits, targets, target_mask):
    """Next-token cross-entropy, averaged over only the masked (CRO-content)
    target positions."""
    vocab_size = logits.size(-1)
    per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
    )
    mask = target_mask.reshape(-1).float()
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def train(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR_MIN
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        n_batches = 0
        for ids, is_cro in loader:
            ids, is_cro = ids.to(device), is_cro.to(device)
            inputs, targets = ids[:, :-1], ids[:, 1:]
            target_mask = is_cro[:, 1:]

            logits = model(inputs)
            loss = masked_loss(logits, targets, target_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")


# ===========================================================================
# Sample generation
# ===========================================================================

def sample_generations(model, corpus, tokenizer, device, situations=SAMPLE_SITUATIONS):
    token_to_id = tokenizer["token_to_id"]
    id_to_token = tokenizer["id_to_token"]

    by_situation = {}
    for dialogue in corpus["dialogues"]:
        by_situation.setdefault(dialogue["situation"], []).append(dialogue)

    print("\nSample generations:")
    for situation in situations:
        candidates = by_situation.get(situation)
        if not candidates:
            print(f"  [{situation}]  (no dialogues found for this situation)")
            continue
        dialogue = candidates[0]
        first_hum, first_cro = dialogue["turns"][0], dialogue["turns"][1]

        prompt_ids = [HUM_ID] + [token_to_id[t] for t in first_hum["tokens"]] + [CRO_ID]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        out = model.generate(prompt, max_new_tokens=4, temperature=0.8)[0].tolist()
        generated_ids = out[len(prompt_ids):]
        generated_words = [id_to_token[str(i)] for i in generated_ids]
        # Trim at the first structural token (a new turn tag or padding) --
        # the model was never trained to predict those, so what follows is noise.
        stop_at = next(
            (i for i, w in enumerate(generated_words) if w in ("[HUM]", "[CRO]", "[PAD]")),
            len(generated_words),
        )
        generated_words = generated_words[:stop_at]

        print(f"  [{dialogue['situation']}]")
        print(f"    HUM: {' '.join(first_hum['tokens'])}")
        print(f"    CRO (generated): {' '.join(generated_words)}")
        print(f"    CRO (actual):    {' '.join(first_cro['tokens'])}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer()
    vocab_size = tokenizer["vocab_size"]
    print(f"Tokenizer: {vocab_size} tokens ([PAD]=0, [HUM]=1, [CRO]=2 + {vocab_size - 3} vocab words)")
    print(f"Saved tokenizer to {TOKENIZER_PATH}")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    turn_counts = {len(d["turns"]) for d in corpus["dialogues"]}
    print(f"Corpus turn-length mix: {sorted(turn_counts)}")

    examples = build_examples(corpus, tokenizer["token_to_id"])
    dataset = DialogueDataset(examples)
    print(f"Loaded {len(dataset)} training examples from {CORPUS_PATH.name}")
    print(f"Context length: {CONTEXT_LEN}")
    print(f"Device: {device}")

    model = PrimitiveMindGPT(vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    train(model, dataset, device)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "context_len": CONTEXT_LEN,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
        },
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")

    sample_generations(model, corpus, tokenizer, device)


if __name__ == "__main__":
    main()
