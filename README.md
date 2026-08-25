# 🪲 Primitive Mind

> An AI Cro-Magnon that lets you watch every thought.

Primitive Mind is an interpretable toy language model — small 
enough that every step of its reasoning is visible. It speaks 
in a 120-word vocabulary with fixed sentence patterns, no 
tenses, no punctuation. Watch how the model arrives at each 
reply: embeddings, attention weights, and output probabilities 
at every step.

**[Live demo →](https://primitive-mind.com)**

---

## What it is

A nanoGPT-style transformer (817k parameters) trained from 
scratch on a synthetic dialogue corpus built around a 
hand-authored 120-word vocabulary. Every word is positioned 
in a 6-dimensional semantic space (alive, size, motion, 
emotion, abstract, good) — values set by hand so any human 
can read them directly.

Small enough to inspect completely.  
Small enough to be wrong in interesting ways.

---

## How it works

1. Select words from the vocabulary panel to build a message
2. The model runs a full forward pass
3. The debug panel walks you through each step:
   - **Tokenize** — input as token IDs with turn markers
   - **Embeddings** — each token as a 6-dim human-readable vector
   - **Attention** — which tokens the model attends to
   - **Logits** — probability distribution over the vocabulary

---

## Structure

```
primitive-mind/
├── model/
│   ├── embeddings.json       # 120-word vocabulary with 6-dim vectors
│   ├── tokenizer.json        # word → token ID mapping
│   ├── corpus_final.json     # 461 Claude-validated dialogues
│   ├── primitive_mind.pt     # trained model weights (Git LFS)
│   ├── train.py              # nanoGPT training script
│   ├── generate_corpus.py    # synthetic corpus generator
│   ├── validate_corpus.py    # Claude API validation
│   └── inference.py          # CLI for local testing
├── backend/
│   ├── main.py               # FastAPI server
│   └── pyproject.toml        # Poetry dependencies
└── frontend/
    └── index.html            # single-file web app
```

---

## Run locally

### Backend
```bash
cd backend
poetry install
poetry run uvicorn main:app --reload --port 8000
```

### Frontend
Open `frontend/index.html` in a browser  
or serve with any static file server.

---

## Model

| Parameter | Value |
|---|---|
| Architecture | Decoder-only transformer |
| Layers | 4 |
| Attention heads | 4 |
| d_model | 128 |
| Parameters | 817,280 |
| Vocabulary | 120 words + 3 special tokens |
| Training data | 461 dialogues, 4 turns each |
| Training | 50 epochs, AdamW, cosine decay |

---

## Vocabulary

120 words across 9 categories: beings, body states, nature, 
food & hunt, actions, qualities, time & direction, 
social & unknown, dialog.

Each word has 6 hand-authored embedding dimensions:
`alive` · `size` · `motion` · `emotion` · `abstract` · `good`

---

## Why

Most AI explainers are either too abstract (metaphors) or 
too technical (math). Primitive Mind sits in between — a real 
transformer you can debug in your head.

Built as both a public explainer and a personal research tool 
for mechanistic interpretability experiments.

---

## Author

Oleksandr Podoliako  
[primitive-mind.com](https://primitive-mind.com)

---

## License

MIT
