"""FastAPI backend for the Primitive Mind toy transformer.

Loads the trained checkpoint + tokenizer from ../model/ (relative to this
file) and exposes:
  POST /chat        generate a CRO reply to a HUM turn, with full internal
                     debug output (embeddings, attention weights, logits)
  GET  /vocabulary   the tokenizer's vocabulary, grouped by embeddings.json category
  GET  /health       liveness/readiness check

The model architecture (PrimitiveMindGPT) is imported directly from
model/train.py rather than duplicated here, so the checkpoint's state_dict
always matches the class that loads it -- model/train.py is the single
source of truth for the architecture.
"""

import asyncio
import json
import math
import sys
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR.parent / "model"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.json"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
CHECKPOINT_PATH = MODEL_DIR / "primitive_mind.pt"
# Anchored to this file's location (not CWD) so it resolves correctly both
# in the single-container deploy (CMD runs from /app, CWD == WORKDIR) and
# in local dev (uvicorn normally launched from inside backend/, where a
# literal "frontend/index.html" would resolve to a nonexistent path).
FRONTEND_INDEX_PATH = BACKEND_DIR.parent / "frontend" / "index.html"
MODEL_VERSION = "v2"

sys.path.insert(0, str(MODEL_DIR))
from train import PrimitiveMindGPT, HUM_ID, CRO_ID, CONTEXT_LEN  # noqa: E402

RATE_LIMIT_PER_DAY = 10_000
MAX_INPUT_WORDS = 10
TOP_K_LOGITS = 8
RESERVED_INPUT_WORDS = {"[pad]", "[hum]", "[cro]"}

TEMPERATURE = 0.8
REPETITION_PENALTY = 1.5  # logit /= 1.5 for a token already emitted this turn
EOS_TOKENS = ("[PAD]", "[HUM]")  # stop when the top predicted token is one of these
EOS_MAX_LENGTH = 4  # stop once the response reaches this many tokens

# Caps how many /chat requests run inference at once -- excess requests
# queue on the semaphore instead of all firing simultaneously and
# thrashing the same CPU cores. Paired with torch.set_num_threads(1) below
# (set once model is loaded) so each individual forward pass doesn't also
# spawn its own internal BLAS thread pool on top of that contention.
INFERENCE_CONCURRENCY = 4
inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)


# ===========================================================================
# Model loading (once, at startup)
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model = PrimitiveMindGPT(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    torch.set_num_threads(1)  # avoid per-request BLAS thread contention; see INFERENCE_CONCURRENCY above

    embeddings_data = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    vocab_by_category = {}
    word_vectors = {}
    for category, entries in embeddings_data.items():
        if category == "_meta":
            continue
        vocab_by_category[category] = list(entries.keys())
        word_vectors.update(entries)
    # Special tokens ([PAD]/[HUM]/[CRO]) have their own hand-authored vectors
    # too -- include them so the frontend can look up ANY prompt token
    # (tags included) uniformly, e.g. for the debug embeddings table.
    word_vectors.update(embeddings_data["_meta"]["special_tokens"])

    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.vocab_by_category = vocab_by_category
    app.state.word_vectors = word_vectors
    app.state.dimensions = embeddings_data["_meta"]["dimensions"]

    yield


app = FastAPI(title="Primitive Mind Backend", lifespan=lifespan)


# ===========================================================================
# CORS -- allow all origins
# ===========================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Browsers reject `allow_credentials=True` combined with a wildcard
    # origin, so this stays False for "allow all origins" to actually work.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Rate limiting -- 10k requests/day per client, in-memory
# ===========================================================================
# In-memory means the counters live only in this process: they reset on
# restart and are NOT shared across multiple uvicorn worker processes. Fine
# for a single-process dev/demo deployment; a multi-worker deployment would
# need a shared store (e.g. Redis) instead.

_rate_limit_counts = defaultdict(lambda: {"date": None, "count": 0})


def _client_ip(request: Request) -> str:
    """Real client IP behind a reverse proxy (e.g. HF Spaces): trust the
    leftmost entry of X-Forwarded-For over request.client.host, which
    otherwise resolves to the proxy's own IP for every visitor. Safe to
    trust here because the proxy fronts this process entirely -- there is
    no direct path for an external client to reach this server and forge
    the header themselves; the proxy always sets/overwrites it."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = _client_ip(request)
        today = date.today().isoformat()

        record = _rate_limit_counts[client_ip]
        if record["date"] != today:
            record["date"] = today
            record["count"] = 0
        record["count"] += 1

        if record["count"] > RATE_LIMIT_PER_DAY:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: {RATE_LIMIT_PER_DAY} requests/day per client"},
            )
        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ===========================================================================
# Debug-instrumented forward pass
# ===========================================================================
# generate_response() below (used for the actual sampling) calls the model's
# fast, fused scaled_dot_product_attention path, which does not expose
# attention weights. This manual re-implementation is only used to build the /chat
# debug payload -- it runs once over the prompt, not during generation.

@torch.no_grad()
def debug_forward(model, idx):
    """Returns (embeddings, attentions, logits) for one forward pass:
      embeddings: (T, d_model) token+positional embedding per input position
      attentions: list of n_layers arrays, each (n_heads, T, T)
      logits:     (T, vocab_size) output logits per position
    """
    _, T = idx.shape
    device = idx.device

    pos = torch.arange(T, device=device)
    tok_emb = model.tok_emb(idx)[0]  # (T, d_model)
    pos_emb = model.pos_emb(pos)  # (T, d_model)
    x = (tok_emb + pos_emb).unsqueeze(0)  # (1, T, d_model); dropout is a no-op in eval()

    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    attentions = []
    for block in model.blocks:
        h = block.ln1(x)
        attn = block.attn
        q, k, v = attn.qkv(h).view(1, T, 3, attn.n_heads, attn.head_dim).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (1, nh, T, hd)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (1, nh, T, T)
        attentions.append(weights[0].tolist())

        out = (weights @ v).transpose(1, 2).contiguous().view(1, T, -1)
        x = x + attn.proj(out)
        x = x + block.mlp(block.ln2(x))

    x = model.ln_f(x)
    logits = model.head(x)[0]  # (T, vocab_size)
    return (tok_emb + pos_emb).tolist(), attentions, logits


# ===========================================================================
# Response generation
# ===========================================================================
# Deliberately not model.generate() (fast fused-attention path with plain
# temperature sampling) -- this adds a repetition penalty and an explicit
# EOS check on top, mirroring model/inference.py so both surfaces generate
# the same way.

@torch.no_grad()
def generate_response(model, prompt_ids, id_to_token, device):
    """Sample the CRO reply one token at a time. At every step: apply a
    repetition penalty to tokens already emitted this turn (divide their
    logit by REPETITION_PENALTY -- this is what stops degenerate loops like
    "now now now now"), then stop if the model's top predicted token is
    [PAD]/[HUM] or EOS_MAX_LENGTH tokens have been generated; otherwise
    sample the next token at TEMPERATURE."""
    generated_ids = []
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    while True:
        if len(generated_ids) >= EOS_MAX_LENGTH:
            break

        logits = model(idx)[0, -1, :].clone()
        for tid in set(generated_ids):
            logits[tid] = logits[tid] / REPETITION_PENALTY

        top_id = int(torch.argmax(logits).item())
        if id_to_token[str(top_id)] in EOS_TOKENS:
            break

        probs = F.softmax(logits / TEMPERATURE, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated_ids.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

    return [id_to_token[str(i)] for i in generated_ids]


def _run_inference(model, prompt, prompt_ids, id_to_token, device):
    """Both forward passes for one /chat request, bundled so the /chat route
    can offload them to a worker thread as a single unit via
    asyncio.to_thread -- keeps the event loop free to serve other requests
    (e.g. /health) while inference runs, and inference_semaphore in the
    caller caps how many of these run at once."""
    embeddings, attentions, next_logits = debug_forward(model, prompt)
    response_words = generate_response(model, prompt_ids, id_to_token, device)
    return embeddings, attentions, next_logits, response_words


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_INDEX_PATH)


class HistoryTurn(BaseModel):
    speaker: str
    tokens: list[str] = Field(max_length=10)


class ChatRequest(BaseModel):
    message: str
    # Prior turns in the current conversation, oldest first (e.g. the last
    # HUM/CRO/HUM/CRO window the client is displaying). Optional and
    # stateless -- the client resends whatever context it wants included;
    # the server holds nothing between requests. Only the most recent
    # MAX_HISTORY_TURNS are used if more are sent. max_length caps list size
    # at the Pydantic validation layer, before MAX_HISTORY_TURNS slicing --
    # otherwise an arbitrarily large list is fully parsed/allocated first.
    history: list[HistoryTurn] = Field(default=[], max_length=20)


MAX_HISTORY_TURNS = 4


@app.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    model = request.app.state.model
    tokenizer = request.app.state.tokenizer
    device = request.app.state.device
    token_to_id = tokenizer["token_to_id"]
    id_to_token = tokenizer["id_to_token"]

    words = payload.message.strip().lower().split()
    if not words:
        raise HTTPException(status_code=400, detail="message must contain at least one word")
    if len(words) > MAX_INPUT_WORDS:
        raise HTTPException(status_code=400, detail=f"message too long: max {MAX_INPUT_WORDS} words")

    unknown = [w for w in words if w in RESERVED_INPUT_WORDS or w not in token_to_id]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown word(s) not in the {tokenizer['vocab_size'] - 3}-word vocabulary: "
                f"{unknown}. See GET /vocabulary."
            ),
        )

    prompt_ids = []
    for turn in payload.history[-MAX_HISTORY_TURNS:]:
        turn_words = [w.lower() for w in turn.tokens]
        unknown_hist = [w for w in turn_words if w in RESERVED_INPUT_WORDS or w not in token_to_id]
        if unknown_hist:
            raise HTTPException(
                status_code=400,
                detail=f"history contains unknown word(s): {unknown_hist}. See GET /vocabulary.",
            )
        prompt_ids.append(CRO_ID if turn.speaker.upper() == "CRO" else HUM_ID)
        prompt_ids.extend(token_to_id[w] for w in turn_words)

    prompt_ids += [HUM_ID] + [token_to_id[w] for w in words] + [CRO_ID]
    if len(prompt_ids) >= CONTEXT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"context too long: {len(prompt_ids)} tokens, max {CONTEXT_LEN - 1}. Send less history.",
        )
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    async with inference_semaphore:
        embeddings, attentions, next_logits, response_words = await asyncio.to_thread(
            _run_inference, model, prompt, prompt_ids, id_to_token, device
        )

    next_token_probs = F.softmax(next_logits[-1], dim=-1)
    topk = torch.topk(next_token_probs, k=min(TOP_K_LOGITS, next_token_probs.shape[-1]))
    logits_debug = [
        {"token": id_to_token[str(idx.item())], "prob": round(p.item(), 4)}
        for p, idx in zip(topk.values, topk.indices)
    ]

    return {
        "input": payload.message,
        "input_tokens": words,
        "response": " ".join(response_words),
        "response_tokens": response_words,
        "debug": {
            "prompt_tokens": [id_to_token[str(i)] for i in prompt_ids],
            "embeddings": [
                {"token": id_to_token[str(tid)], "vector": [round(v, 4) for v in vec]}
                for tid, vec in zip(prompt_ids, embeddings)
            ],
            "attention": [
                {
                    "layer": layer_idx,
                    "weights": [[[round(v, 4) for v in row] for row in head] for head in layer],
                }
                for layer_idx, layer in enumerate(attentions)
            ],
            "logits": {
                "position": "next token after prompt (first token of the CRO reply)",
                "top_k": logits_debug,
            },
        },
    }


@app.get("/vocabulary")
def vocabulary(request: Request):
    tokenizer = request.app.state.tokenizer
    return {
        "vocab_size": tokenizer["vocab_size"],
        "special_tokens": tokenizer["special_tokens"],
        "dimensions": request.app.state.dimensions,
        "categories": request.app.state.vocab_by_category,
        "token_to_id": tokenizer["token_to_id"],
        # word/tag -> its hand-authored 6-dim embeddings.json vector (not the
        # model's learned 128-dim vector -- this is the human-interpretable
        # one used for the hover status bar and the debug embeddings table).
        "embeddings": request.app.state.word_vectors,
    }


@app.get("/health")
def health(request: Request):
    tokenizer = getattr(request.app.state, "tokenizer", None)
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "model", None) is not None,
        "model_version": MODEL_VERSION,
        "vocab_size": tokenizer["vocab_size"] if tokenizer else None,
        "device": str(getattr(request.app.state, "device", "unknown")),
    }
