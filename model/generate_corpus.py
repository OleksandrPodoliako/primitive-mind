"""Generate a synthetic dialogue corpus for the Primitive Mind toy language model.

Seed-based approach: 5 hand-written seed dialogues per situation (100 seeds
total), each expanded into 9 additional variations by substituting tokens
with other members of the same synonym group. 5 seeds + 5*9 variations = 50
dialogues per situation, 1000 total. Writes model/corpus_raw.json.

NOTE ON SEED WORDING: several words used in the originally drafted seeds
are not present in embeddings.json's 120-word vocabulary (e.g. "where",
"away", "food", "shelter", "drink", "night", "weak" ...) and none of them
belong to any synonym group, so substitution could never fix them. Each was
replaced with the closest in-vocabulary word so every seed is valid from the
start -- see OOV_REPLACEMENTS below for the exact mapping used.
"""

import json
import random
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.json"
OUTPUT_PATH = MODEL_DIR / "corpus_raw.json"
EXTRA_OUTPUT_PATH = MODEL_DIR / "corpus_extra.json"

SEED = 42
SEEDS_PER_SITUATION = 5
VARIATIONS_PER_SEED = 9
MAX_ATTEMPTS_PER_VARIATION = 20
SPEAKERS = ("HUM", "CRO", "HUM", "CRO")

# Situations that came back under-represented after Claude-based semantic
# validation of corpus_raw.json (validated via model/validate_corpus.py).
# These 4 situations were given 10 extra, simpler/more-causal seeds each
# (15 total) -- see the "Simpler/more-coherent additions" comments in
# SEED_DIALOGUES below. `--extra` regenerates just these 4 at 50 each,
# using only the embeddings-vocabulary/length check (not the Claude API),
# and writes them to EXTRA_OUTPUT_PATH instead of touching corpus_raw.json.
EXTRA_SITUATIONS = ("danger", "fire", "shelter", "us-vs-them")
EXTRA_TARGET_COUNT = 50
EXTRA_SEED = 43

# Words used in an earlier draft of the seeds that turned out not to exist in
# embeddings.json, mapped to the closest in-vocabulary word. Kept here as a
# record of what changed; the SEED_DIALOGUES below already have these applied.
OOV_REPLACEMENTS = {
    "where": "far", "away": "far", "out": "there", "back": "here",
    "watch": "see", "dark": "sun-down", "keep": "stay", "burn": "stay",
    "sweet": "good", "winter": "cold", "night": "sun-down", "long": "much",
    "fair": "good", "care": "help", "happy": "good", "sad": "bad",
    "gone": "dead", "learn": "know", "born": "alive", "remember": "know",
    "bring": "carry", "follow": "come", "sound": "spirit", "thing": "spirit",
    "drink": "take", "food": "meat", "shelter": "cave", "exchange": "share",
    "weak": "tired", "all": "tribe", "many days": "many-days",
}

SYNONYM_GROUPS = {
    "THREATS": ["enemy", "wolf", "snake", "stranger"],
    "PEOPLE": ["person", "tribe", "elder", "man", "woman"],
    "SAFE_PLACE": ["cave", "hill", "forest"],
    "DIRECTION": ["here", "there", "near", "far", "that-way", "this-way"],
    "DANGER_ACT": ["run", "hide", "fight"],
    "FOOD_ITEMS": ["meat", "fruit", "root", "fish-food"],
    "HUNT_ACT": ["hunt", "find", "gather"],
    "WATER_SRC": ["water", "river"],
    "FIRE_SRC": ["fire"],
    "BODY_NEG": ["cold", "hungry", "tired", "hurt", "thirsty"],
    "BODY_POS": ["warm", "full", "healthy", "strong"],
    "QUAL_GOOD": ["good", "safe", "much"],
    "QUAL_BAD": ["bad", "danger", "little"],
    "TIME": ["now", "before", "sun-up", "sun-down"],
    "SIZE": ["big", "small", "many", "one"],
}

WORD_TO_GROUP = {
    word: group for group, words in SYNONYM_GROUPS.items() for word in words
}

SEED_DIALOGUES = {
    "danger": [
        [["enemy", "come", "near"], ["person", "run", "fast"], ["question", "cave", "far"], ["cave", "up", "hill"]],
        [["wolf", "here", "now"], ["person", "hide", "cave"], ["wolf", "go", "far"], ["person", "safe", "now"]],
        [["enemy", "many", "strong"], ["tribe", "fight", "together"], ["enemy", "run", "far"], ["tribe", "safe", "now"]],
        [["snake", "near", "water"], ["person", "see", "danger"], ["person", "go", "fast"], ["snake", "not", "come"]],
        [["stranger", "come", "here"], ["person", "call", "tribe"], ["tribe", "come", "fast"], ["stranger", "go", "far"]],
        # Simpler/more-coherent additions (see EXTRA_SITUATIONS below): each
        # follows statement -> direct response -> follow-up on turn 1 -> resolution.
        [["wolf", "come", "near"], ["person", "run", "fast"], ["wolf", "come", "here"], ["person", "hide", "cave"]],
        [["enemy", "near", "now"], ["tribe", "fight", "enemy"], ["enemy", "run", "far"], ["tribe", "safe", "now"]],
        [["snake", "here", "now"], ["person", "see", "snake"], ["person", "go", "far"], ["person", "safe", "now"]],
        [["question", "wolf", "far"], ["wolf", "far", "there"], ["person", "stay", "here"], ["person", "safe", "now"]],
        [["stranger", "near", "now"], ["person", "hide", "cave"], ["stranger", "go", "far"], ["person", "safe", "now"]],
        [["enemy", "come", "fast"], ["tribe", "hide", "cave"], ["enemy", "go", "far"], ["tribe", "safe", "now"]],
        [["wolf", "near", "water"], ["person", "see", "wolf"], ["person", "run", "far"], ["person", "safe", "now"]],
        [["enemy", "fight", "person"], ["person", "run", "far"], ["enemy", "come", "near"], ["person", "hide", "cave"]],
        [["snake", "come", "fast"], ["person", "run", "far"], ["snake", "not", "come"], ["person", "safe", "now"]],
        [["enemy", "many", "strong"], ["tribe", "fight", "together"], ["enemy", "go", "far"], ["tribe", "safe", "now"]],
    ],
    "food": [
        [["person", "hungry", "now"], ["hunt", "animal", "here"], ["animal", "run", "fast"], ["person", "find", "meat"]],
        [["tribe", "hungry", "much"], ["elder", "know", "path"], ["path", "go", "forest"], ["forest", "fruit", "good"]],
        [["question", "meat", "far"], ["meat", "here", "much"], ["person", "take", "meat"], ["person", "full", "now"]],
        [["child", "hungry", "cold"], ["give", "fruit", "child"], ["child", "eat", "fruit"], ["child", "good", "now"]],
        [["person", "find", "root"], ["root", "good", "eat"], ["person", "gather", "much"], ["tribe", "eat", "together"]],
    ],
    "water": [
        [["person", "thirsty", "now"], ["river", "near", "here"], ["person", "go", "river"], ["person", "take", "good"]],
        [["question", "water", "far"], ["water", "far", "that-way"], ["person", "walk", "far"], ["water", "good", "much"]],
        [["tribe", "thirsty", "much"], ["elder", "know", "water"], ["water", "up", "hill"], ["tribe", "go", "fast"]],
        [["child", "thirsty", "hurt"], ["give", "water", "child"], ["child", "take", "now"], ["child", "healthy", "now"]],
        [["rain", "come", "now"], ["person", "gather", "water"], ["water", "much", "good"], ["tribe", "not", "thirsty"]],
    ],
    "fire": [
        [["person", "cold", "now"], ["make", "fire", "here"], ["fire", "good", "warm"], ["person", "warm", "now"]],
        [["fire", "go", "there"], ["person", "make", "fire"], ["fire", "come", "here"], ["person", "safe", "warm"]],
        [["rain", "come", "strong"], ["fire", "danger", "now"], ["person", "hide", "fire"], ["fire", "stay", "good"]],
        [["child", "cold", "hurt"], ["fire", "here", "good"], ["child", "come", "fire"], ["child", "warm", "now"]],
        [["sun-down", "come", "now"], ["make", "fire", "big"], ["fire", "stay", "danger"], ["tribe", "sleep", "safe"]],
        # Simpler/more-coherent additions (see EXTRA_SITUATIONS below).
        [["tribe", "cold", "now"], ["tribe", "make", "fire"], ["fire", "warm", "good"], ["tribe", "warm", "now"]],
        [["question", "fire", "far"], ["fire", "here", "near"], ["person", "go", "fire"], ["person", "warm", "now"]],
        [["elder", "cold", "now"], ["elder", "make", "fire"], ["fire", "good", "warm"], ["elder", "warm", "now"]],
        [["child", "cold", "hurt"], ["give", "child", "fire"], ["child", "warm", "now"], ["child", "good", "now"]],
        [["fire", "far", "there"], ["person", "go", "fire"], ["fire", "warm", "good"], ["person", "safe", "now"]],
        [["wind", "strong", "cold"], ["tribe", "make", "fire"], ["fire", "good", "warm"], ["tribe", "safe", "now"]],
        [["fire", "here", "small"], ["person", "make", "fire"], ["fire", "good", "big"], ["person", "warm", "now"]],
        [["sun-down", "come", "now"], ["make", "fire", "here"], ["fire", "good", "warm"], ["tribe", "sleep", "safe"]],
        [["snow", "come", "now"], ["tribe", "make", "fire"], ["fire", "warm", "good"], ["tribe", "safe", "now"]],
        [["person", "cold", "much"], ["person", "make", "fire"], ["fire", "warm", "now"], ["person", "good", "now"]],
    ],
    "shelter": [
        [["rain", "come", "strong"], ["person", "go", "cave"], ["cave", "safe", "good"], ["person", "sleep", "cave"]],
        [["wind", "strong", "cold"], ["tribe", "build", "cave"], ["cave", "good", "strong"], ["tribe", "warm", "now"]],
        [["question", "cave", "far"], ["cave", "far", "hill"], ["person", "walk", "far"], ["cave", "safe", "good"]],
        [["sun-down", "come", "cold"], ["person", "find", "cave"], ["cave", "good", "safe"], ["person", "sleep", "now"]],
        [["snow", "come", "now"], ["tribe", "go", "cave"], ["cave", "warm", "safe"], ["tribe", "together", "good"]],
        # Simpler/more-coherent additions (see EXTRA_SITUATIONS below).
        [["wind", "strong", "cold"], ["tribe", "go", "cave"], ["cave", "warm", "safe"], ["tribe", "sleep", "cave"]],
        [["question", "cave", "far"], ["cave", "near", "hill"], ["person", "go", "cave"], ["person", "safe", "now"]],
        [["snow", "come", "now"], ["tribe", "go", "cave"], ["cave", "warm", "safe"], ["tribe", "safe", "now"]],
        [["enemy", "come", "near"], ["person", "go", "cave"], ["cave", "safe", "good"], ["person", "hide", "cave"]],
        [["sun-down", "come", "now"], ["elder", "find", "cave"], ["cave", "good", "safe"], ["elder", "sleep", "now"]],
        [["tribe", "tired", "much"], ["tribe", "go", "cave"], ["cave", "safe", "good"], ["tribe", "sleep", "cave"]],
        [["rain", "come", "now"], ["tribe", "find", "cave"], ["cave", "safe", "good"], ["tribe", "stay", "cave"]],
        [["child", "cold", "hurt"], ["person", "go", "cave"], ["cave", "warm", "safe"], ["child", "sleep", "cave"]],
        [["wolf", "near", "now"], ["tribe", "go", "cave"], ["cave", "safe", "good"], ["tribe", "hide", "cave"]],
        [["rain", "come", "strong"], ["tribe", "go", "cave"], ["cave", "safe", "warm"], ["tribe", "sleep", "cave"]],
    ],
    "sleep": [
        [["person", "tired", "much"], ["person", "sleep", "now"], ["sun", "up", "come"], ["person", "wake", "strong"]],
        [["tribe", "sleep", "safe"], ["person", "stay", "here"], ["sun-down", "go", "far"], ["sun", "up", "come"]],
        [["child", "tired", "hurt"], ["child", "sleep", "now"], ["person", "see", "child"], ["child", "wake", "good"]],
        [["enemy", "near", "now"], ["person", "not", "sleep"], ["person", "see", "sun-down"], ["sun", "up", "safe"]],
        [["person", "wake", "now"], ["sun", "up", "come"], ["person", "strong", "good"], ["tribe", "wake", "together"]],
    ],
    "weather": [
        [["rain", "come", "strong"], ["person", "go", "cave"], ["rain", "go", "far"], ["person", "come", "there"]],
        [["wind", "cold", "strong"], ["tribe", "make", "fire"], ["fire", "good", "warm"], ["tribe", "safe", "now"]],
        [["sun", "strong", "now"], ["person", "find", "water"], ["water", "good", "cold"], ["person", "take", "much"]],
        [["snow", "come", "many"], ["tribe", "go", "cave"], ["snow", "cold", "danger"], ["cave", "safe", "warm"]],
        [["question", "rain", "come"], ["rain", "come", "after"], ["person", "make", "cave"], ["cave", "good"]],
    ],
    "seasons": [
        [["sun", "strong", "now"], ["hunt", "animal", "much"], ["meat", "much", "good"], ["tribe", "full", "now"]],
        [["cold", "come", "now"], ["tribe", "gather", "meat"], ["meat", "much", "good"], ["tribe", "safe", "cold"]],
        [["many-days", "cold"], ["tribe", "stay", "cave"], ["sun", "come", "here"], ["tribe", "go", "there", "good"]],
        [["animal", "come", "here"], ["tribe", "hunt", "now"], ["meat", "much", "good"], ["tribe", "full", "strong"]],
        [["fruit", "come", "now"], ["tribe", "gather", "much"], ["fruit", "good", "much"], ["tribe", "eat", "together"]],
    ],
    "direction": [
        [["question", "cave", "far"], ["cave", "that-way", "far"], ["person", "go", "that-way"], ["cave", "good", "safe"]],
        [["question", "water", "far"], ["water", "this-way", "near"], ["person", "go", "fast"], ["water", "here", "good"]],
        [["enemy", "that-way"], ["person", "go", "this-way"], ["person", "run", "fast"], ["person", "safe", "now"]],
        [["question", "tribe", "far"], ["tribe", "up", "hill"], ["person", "go", "up"], ["tribe", "here", "safe"]],
        [["meat", "that-way", "far"], ["person", "walk", "far"], ["meat", "here", "much"], ["person", "take", "meat"]],
    ],
    "distance": [
        [["enemy", "near", "now"], ["person", "run", "far"], ["enemy", "not", "come"], ["person", "safe", "far"]],
        [["water", "far", "there"], ["person", "thirsty", "much"], ["person", "walk", "far"], ["water", "good", "now"]],
        [["cave", "near", "here"], ["person", "go", "cave"], ["cave", "safe", "good"], ["person", "sleep", "now"]],
        [["tribe", "far", "there"], ["person", "call", "tribe"], ["tribe", "come", "fast"], ["tribe", "here", "now"]],
        [["meat", "near", "here"], ["person", "find", "meat"], ["meat", "good", "much"], ["person", "eat", "now"]],
    ],
    "time-of-day": [
        [["sun", "up", "now"], ["tribe", "hunt", "animal"], ["animal", "near", "forest"], ["tribe", "find", "meat"]],
        [["sun", "down", "come"], ["tribe", "make", "fire"], ["fire", "good", "warm"], ["tribe", "sleep", "safe"]],
        [["sun-down", "come", "now"], ["person", "fear", "sun-down"], ["fire", "here", "good"], ["person", "safe", "now"]],
        [["sun", "up", "strong"], ["person", "go", "water"], ["water", "good", "cold"], ["person", "take", "much"]],
        [["sun-down", "much", "cold"], ["tribe", "together", "fire"], ["fire", "warm", "good"], ["tribe", "sleep", "now"]],
    ],
    "body-states": [
        [["person", "hurt", "bad"], ["person", "stay", "here"], ["tribe", "help", "person"], ["person", "healthy", "now"]],
        [["elder", "sick", "now"], ["tribe", "give", "meat"], ["elder", "eat", "good"], ["elder", "strong", "now"]],
        [["child", "cold", "hurt"], ["give", "child", "fire"], ["child", "warm", "now"], ["child", "good", "healthy"]],
        [["person", "tired", "hurt"], ["person", "sleep", "now"], ["person", "wake", "strong"], ["person", "good", "now"]],
        [["tribe", "hungry", "cold"], ["find", "meat", "fire"], ["eat", "together", "warm"], ["tribe", "good", "now"]],
    ],
    "us-vs-them": [
        [["stranger", "come", "here"], ["person", "see", "stranger"], ["stranger", "go", "far"], ["person", "safe"]],
        [["ours", "tribe", "strong"], ["theirs", "tribe", "tired"], ["ours", "tribe", "safe"], ["theirs", "go", "far"]],
        [["stranger", "take", "meat"], ["tribe", "fight", "stranger"], ["stranger", "run", "far"], ["meat", "ours", "now"]],
        [["theirs", "enemy", "come"], ["tribe", "together", "fight"], ["enemy", "run", "far"], ["tribe", "safe", "now"]],
        [["stranger", "good", "come"], ["tribe", "give", "meat"], ["stranger", "give", "stone"], ["together", "good"]],
        # Simpler/more-coherent additions (see EXTRA_SITUATIONS below).
        [["stranger", "come", "near"], ["person", "call", "tribe"], ["tribe", "come", "fast"], ["stranger", "go", "far"]],
        [["enemy", "take", "meat"], ["tribe", "fight", "enemy"], ["enemy", "run", "far"], ["meat", "ours", "now"]],
        [["man", "see", "stranger"], ["man", "call", "tribe"], ["tribe", "come", "fast"], ["stranger", "go", "far"]],
        [["enemy", "come", "tribe"], ["tribe", "fight", "together"], ["enemy", "run", "far"], ["tribe", "safe", "now"]],
        [["stranger", "take", "stone"], ["person", "see", "stranger"], ["stranger", "go", "far"], ["stone", "ours", "now"]],
        [["enemy", "come", "near"], ["tribe", "fight", "enemy"], ["enemy", "run", "far"], ["tribe", "safe", "now"]],
        [["stranger", "give", "meat"], ["tribe", "give", "stone"], ["stone", "good", "strong"], ["together", "good"]],
        [["enemy", "near", "tribe"], ["person", "call", "tribe"], ["tribe", "come", "fast"], ["enemy", "go", "far"]],
        [["stranger", "come", "tribe"], ["elder", "see", "stranger"], ["stranger", "go", "far"], ["tribe", "safe", "now"]],
        [["enemy", "near", "now"], ["tribe", "hide", "cave"], ["enemy", "go", "far"], ["tribe", "safe", "now"]],
    ],
    "strength": [
        [["enemy", "big", "strong"], ["person", "fear", "much"], ["tribe", "together", "strong"], ["enemy", "run"]],
        [["person", "tired", "hurt"], ["tribe", "help", "person"], ["person", "strong", "now"], ["person", "good"]],
        [["strong-one", "come"], ["tribe", "safe", "good"], ["strong-one", "fight"], ["enemy", "run", "far"]],
        [["weak-one", "hungry"], ["tribe", "give", "meat"], ["weak-one", "strong", "now"], ["tribe", "together"]],
        [["enemy", "strong", "many"], ["person", "run", "fast"], ["tribe", "come", "help"], ["enemy", "go", "far"]],
    ],
    "together-alone": [
        [["person", "alone", "fear"], ["tribe", "come", "here"], ["together", "safe", "good"], ["person", "good", "now"]],
        [["tribe", "together", "strong"], ["enemy", "not", "come"], ["tribe", "safe", "here"], ["together", "good"]],
        [["person", "go", "alone"], ["person", "danger", "now"], ["tribe", "come", "fast"], ["together", "safe"]],
        [["child", "alone", "cold"], ["person", "find", "child"], ["together", "warm", "safe"], ["child", "good"]],
        [["alone", "danger", "much"], ["call", "tribe", "now"], ["tribe", "come", "fast"], ["together", "safe", "good"]],
    ],
    "exchange": [
        [["person", "give", "meat"], ["tribe", "give", "stone"], ["stone", "good", "strong"], ["together", "good"]],
        [["elder", "give", "know"], ["person", "take", "know"], ["know", "good", "help"], ["person", "go", "safe"]],
        [["tribe", "share", "meat"], ["tribe", "eat", "together"], ["meat", "much", "good"], ["tribe", "strong", "now"]],
        [["person", "take", "much"], ["tribe", "not", "give"], ["person", "give", "here"], ["together", "good", "now"]],
        [["give", "water", "take"], ["take", "meat", "give"], ["share", "good", "much"], ["together", "strong"]],
    ],
    "emotions": [
        [["person", "fear", "much"], ["tribe", "come", "here"], ["together", "safe", "now"], ["fear", "go", "far"]],
        [["tribe", "good", "now"], ["hunt", "good", "much"], ["meat", "much", "good"], ["tribe", "good", "strong"]],
        [["person", "bad", "alone"], ["tribe", "come", "person"], ["together", "warm", "good"], ["person", "good"]],
        [["elder", "dead", "now"], ["tribe", "bad", "much"], ["elder", "good", "before"], ["tribe", "know"]],
        [["child", "good", "healthy"], ["tribe", "good", "now"], ["meat", "much", "warm"], ["together", "good"]],
    ],
    "young-old": [
        [["child", "not", "know"], ["elder", "give", "know"], ["child", "know", "good"], ["child", "strong", "now"]],
        [["elder", "hurt", "tired"], ["tribe", "help", "elder"], ["elder", "good", "safe"], ["tribe", "together"]],
        [["young", "person", "strong"], ["elder", "know", "path"], ["together", "go", "safe"], ["tribe", "good"]],
        [["child", "alone", "danger"], ["elder", "find", "child"], ["elder", "carry", "safe"], ["child", "good"]],
        [["old", "elder", "sick"], ["tribe", "stay", "help"], ["elder", "good", "now"], ["tribe", "together", "safe"]],
    ],
    "birth-death": [
        [["new", "person", "come"], ["tribe", "good", "safe"], ["new", "person", "small"], ["tribe", "give", "meat"]],
        [["elder", "dead", "now"], ["tribe", "bad", "much"], ["elder", "good", "before"], ["tribe", "know", "elder"]],
        [["new", "child", "alive"], ["tribe", "together", "good"], ["child", "alive", "good"], ["tribe", "give", "help"]],
        [["person", "hurt", "bad"], ["tribe", "help", "much"], ["person", "alive", "good"], ["tribe", "together"]],
        [["animal", "dead", "here"], ["tribe", "take", "meat"], ["meat", "good", "much"], ["tribe", "eat", "now"]],
    ],
    "spirits-unknown": [
        [["strange", "spirit", "near"], ["person", "fear", "much"], ["person", "call", "tribe"], ["together", "safe"]],
        [["bad", "spirit", "here"], ["tribe", "make", "fire"], ["fire", "stay", "spirit"], ["tribe", "safe", "now"]],
        [["unknown", "spirit", "sun-down"], ["person", "not", "sleep"], ["sun", "up", "come"], ["spirit", "go", "far"]],
        [["spirit", "walk", "sun-down"], ["tribe", "together", "stay"], ["fire", "stay", "strong"], ["spirit", "go"]],
        [["strange", "animal", "come"], ["person", "not", "know"], ["animal", "go", "far"], ["person", "safe", "now"]],
    ],
}


def load_vocab_words(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = set()
    for category, entries in data.items():
        if category == "_meta":
            continue
        words.update(entries.keys())
    return words


def substitute_token(token, rng):
    """Replace `token` with another member of its synonym group, if it has
    one; tokens outside every group (and singleton groups) pass through."""
    group = WORD_TO_GROUP.get(token)
    if not group:
        return token
    options = SYNONYM_GROUPS[group]
    choices = [w for w in options if w != token] or options
    return rng.choice(choices)


def make_variation(seed_turns, rng):
    return [[substitute_token(tok, rng) for tok in turn] for turn in seed_turns]


def to_dialogue(situation, turns):
    return {
        "situation": situation,
        "turns": [{"speaker": s, "tokens": t} for s, t in zip(SPEAKERS, turns)],
    }


def is_valid_dialogue(dialogue, vocab_words):
    for turn in dialogue["turns"]:
        tokens = turn["tokens"]
        if not (2 <= len(tokens) <= 4):
            return False
        if any(tok not in vocab_words for tok in tokens):
            return False
    return True


def generate_extra_dialogues(vocab_words, rng):
    """Regenerate just EXTRA_SITUATIONS at EXTRA_TARGET_COUNT dialogues each,
    using their (now 15-seed) pools. Variations are distributed round-robin
    across all 15 seeds rather than 9-per-seed -- more base seeds and less
    substitution churn per seed is the whole point of the fix (heavy
    substitution was what produced the incoherent dialogues in corpus_raw.json)."""
    dialogues = []
    per_situation_counts = {}
    failed_variations = 0

    for situation in EXTRA_SITUATIONS:
        seeds = SEED_DIALOGUES[situation]
        situation_dialogues = []
        for seed in seeds:
            seed_dialogue = to_dialogue(situation, seed)
            if not is_valid_dialogue(seed_dialogue, vocab_words):
                raise ValueError(f"Seed for '{situation}' failed validation: {seed}")
            situation_dialogues.append(seed_dialogue)

        needed = EXTRA_TARGET_COUNT - len(situation_dialogues)
        if needed < 0:
            raise ValueError(
                f"'{situation}' has {len(seeds)} seeds, more than EXTRA_TARGET_COUNT={EXTRA_TARGET_COUNT}"
            )

        produced = 0
        attempts = 0
        max_attempts = needed * MAX_ATTEMPTS_PER_VARIATION
        while produced < needed and attempts < max_attempts:
            seed = seeds[attempts % len(seeds)]  # round-robin across all 15 seeds
            attempts += 1
            variation = make_variation(seed, rng)
            dialogue = to_dialogue(situation, variation)
            if is_valid_dialogue(dialogue, vocab_words):
                situation_dialogues.append(dialogue)
                produced += 1
            else:
                failed_variations += 1
        while produced < needed:
            # Should not happen with clean seeds/groups; guarantees the
            # exact per-situation count regardless.
            situation_dialogues.append(to_dialogue(situation, seeds[produced % len(seeds)]))
            produced += 1

        dialogues.extend(situation_dialogues)
        per_situation_counts[situation] = len(situation_dialogues)

    return dialogues, per_situation_counts, failed_variations


def main_extra():
    rng = random.Random(EXTRA_SEED)
    vocab_words = load_vocab_words(EMBEDDINGS_PATH)

    dialogues, per_situation_counts, failed_variations = generate_extra_dialogues(vocab_words, rng)

    corpus = {
        "meta": {"total": len(dialogues), "situations": len(EXTRA_SITUATIONS)},
        "dialogues": dialogues,
    }

    with open(EXTRA_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Loaded {len(vocab_words)} vocabulary words from {EMBEDDINGS_PATH.name}")
    print(f"Regenerated {len(dialogues)} dialogues across {len(EXTRA_SITUATIONS)} situations: {', '.join(EXTRA_SITUATIONS)}")
    print(f"Wrote corpus to {EXTRA_OUTPUT_PATH}")
    print()
    print("Per-situation dialogue counts:")
    for situation, count in per_situation_counts.items():
        print(f"  {situation:20s} {count}")
    print()
    print(f"Variation attempts that failed token/length validation and were discarded: {failed_variations}")


def main():
    rng = random.Random(SEED)
    vocab_words = load_vocab_words(EMBEDDINGS_PATH)

    dialogues = []
    per_situation_counts = {}
    failed_variations = 0

    for situation, seeds in SEED_DIALOGUES.items():
        situation_dialogues = []
        for seed in seeds:
            seed_dialogue = to_dialogue(situation, seed)
            if not is_valid_dialogue(seed_dialogue, vocab_words):
                raise ValueError(f"Seed for '{situation}' failed validation: {seed}")
            situation_dialogues.append(seed_dialogue)

            produced = 0
            attempts = 0
            while produced < VARIATIONS_PER_SEED and attempts < MAX_ATTEMPTS_PER_VARIATION * VARIATIONS_PER_SEED:
                attempts += 1
                variation = make_variation(seed, rng)
                dialogue = to_dialogue(situation, variation)
                if is_valid_dialogue(dialogue, vocab_words):
                    situation_dialogues.append(dialogue)
                    produced += 1
                else:
                    failed_variations += 1
            while produced < VARIATIONS_PER_SEED:
                # Should not happen with clean seeds/groups; guarantees the
                # exact per-situation count regardless.
                situation_dialogues.append(to_dialogue(situation, seed))
                produced += 1

        dialogues.extend(situation_dialogues)
        per_situation_counts[situation] = len(situation_dialogues)

    corpus = {
        "meta": {"total": len(dialogues), "situations": len(SEED_DIALOGUES)},
        "dialogues": dialogues,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    print(f"Loaded {len(vocab_words)} vocabulary words from {EMBEDDINGS_PATH.name}")
    print(f"Generated {len(dialogues)} dialogues across {len(SEED_DIALOGUES)} situations")
    print(f"Wrote corpus to {OUTPUT_PATH}")
    print()
    print("Per-situation dialogue counts:")
    for situation, count in per_situation_counts.items():
        print(f"  {situation:20s} {count}")
    print()
    print(f"Variation attempts that failed token/length validation and were discarded: {failed_variations}")


if __name__ == "__main__":
    import sys

    if "--extra" in sys.argv:
        main_extra()
    else:
        main()
