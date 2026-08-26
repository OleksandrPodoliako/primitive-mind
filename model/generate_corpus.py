"""Generate v2 of the synthetic dialogue corpus for the Primitive Mind toy
language model.

Seed-based approach, same idea as corpus_raw.json's generator but bigger and
stricter: 10 hand-written seed dialogues per situation (200 seeds total),
each with 6 turns (3 HUM + 3 CRO, up from 4), expanded into 9 additional
variations by substituting tokens with other members of the same synonym
group. 10 seeds + 10*9 variations = 100 dialogues per situation, 2000 total
across 20 situations. Writes model/corpus_v2_raw.json.

Two hard rules apply to every CRO reply -- both the seed originals and every
generated variation -- enforced at generation time by
enforce_cro_reply_rules() rather than by hand-fixing the seed literal:
  1. no token may appear twice in the same CRO reply (this also covers the
     "'now' at most once" rule, since "now" is just a token like any other)
  2. a CRO reply may not start with "person"

NOTE ON SEED WORDING: the 200 seeds as originally hand-written by the user
used many words outside the 120-word embeddings.json vocabulary (188 distinct
OOV words, 731 of ~3600 tokens -- about 20%). Per explicit user instruction,
every OOV word was mapped to the closest existing vocabulary word before the
seeds below were assembled -- see OOV_REPLACEMENTS for the full mapping
(kept here as a transparency record; the SEED_DIALOGUES below already have
it applied). Two of the user's originally-given mappings ("build" -> "make",
"run" -> "go") were dropped because "build" and "run" are themselves already
valid vocabulary words (actions category) -- applying them would have
incorrectly altered already-valid tokens, so both are left untouched
wherever they occur.
"""

import json
import random
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.json"
OUTPUT_PATH = MODEL_DIR / "corpus_v2_raw.json"

SEED = 42
SEEDS_PER_SITUATION = 10
VARIATIONS_PER_SEED = 9
MAX_ATTEMPTS_PER_VARIATION = 20
SPEAKERS = ("HUM", "CRO", "HUM", "CRO", "HUM", "CRO")

# Full OOV -> vocabulary mapping applied to the raw seeds before they were
# embedded below. Combines: (a) ~30 mappings established earlier for the v1
# corpus, (b) the user's ~40 explicit mappings for this rewrite ("build" and
# "run" dropped -- see module docstring), (c) ~131 further words mapped the
# same way (closest semantic match) to reach 0 unmapped tokens across all
# 200 seeds. Not used at runtime -- documentation only.
OOV_REPLACEMENTS = {
    "where": "far", "away": "far", "out": "there", "back": "here", "watch": "see",
    "dark": "sun-down", "keep": "stay", "burn": "stay", "sweet": "good",
    "winter": "cold", "night": "sun-down", "long": "much", "fair": "good",
    "care": "help", "happy": "good", "sad": "bad", "gone": "dead", "learn": "know",
    "born": "alive", "remember": "know", "bring": "carry", "follow": "come",
    "sound": "spirit", "thing": "spirit", "drink": "take", "food": "meat",
    "shelter": "cave", "exchange": "share", "weak": "tired", "all": "tribe",
    "trade": "give", "memory": "know", "past": "before", "birth": "alive",
    "death": "dead", "celebrate": "good", "bond": "together", "hold": "stay",
    "stand": "stay", "beat": "fight", "withstand": "stay", "lead": "go",
    "guide": "go", "protect": "stay", "heal": "good", "neutral": "good",
    "appease": "give", "offering": "give", "sign": "see", "read": "know",
    "add": "give", "renew": "good", "precious": "good", "ease": "good",
    "mourn": "bad", "feed": "eat", "foundation": "good",
    "grow": "strong", "always": "same", "more": "much", "calm": "safe",
    "rest": "sleep", "better": "good", "well": "good", "best": "good",
    "season": "many-days", "again": "after", "light": "sun", "life": "alive",
    "storm": "rain", "future": "after", "both": "together", "hope": "good",
    "save": "help", "work": "make", "morning": "sun-up", "trust": "safe",
    "win": "strong", "speak": "call", "live": "alive", "teach": "know",
    "warn": "call", "how": "far", "show": "see", "soon": "now", "worth": "good",
    "gift": "give", "need": "take", "miss": "alone", "less": "little",
    "lonely": "alone", "understand": "know", "wood": "tree", "strike": "throw",
    "dry": "safe", "done": "good", "days": "many-days", "when": "now",
    "wait": "stay", "choose": "take", "friend": "tribe", "peace": "safe",
    "who": "elder", "most": "many", "with": "together", "hard": "bad",
    "what": "unknown", "get": "take", "proud": "good", "angry": "bad",
    "feel": "know", "brave": "strong", "answer": "know", "wisdom": "know",
    "present": "now", "respect": "good", "still": "stay", "break": "hurt",
    "last": "stay", "weather": "rain", "kinds": "many", "spring": "warm",
    "enough": "much", "loud": "strong", "which": "that-way", "way": "that-way",
    "early": "sun-up", "home": "cave", "play": "good", "health": "healthy",
    "stop": "stay", "mark": "stone", "border": "far", "thank": "good",
    "deed": "good", "welcome": "give", "stronger": "strong", "later": "after",
    "regroup": "together", "plan": "know", "fall": "dead", "load": "carry",
    "arrive": "come", "someone": "person", "along": "together", "two": "one",
    "safer": "safe", "agree": "good", "dangerous": "danger", "close": "near",
    "lost": "alone", "too": "much", "want": "take", "solve": "help",
    "circle": "together", "day": "now", "anger": "bad", "think": "know",
    "clear": "good", "dance": "good", "time": "now", "battle": "fight",
    "courage": "strong", "turn": "go", "why": "know", "right": "good",
    "wise": "know", "age": "old", "ask": "call", "first": "one",
    "practice": "know", "experience": "know", "listen": "hear", "deep": "much",
    "plant": "tree", "cycle": "after", "around": "near", "pass": "go",
    "strength": "strong", "look": "see", "bright": "warm", "scare": "fear",
    "leave": "go", "seen": "see", "explain": "know", "scary": "fear",
    "hopeful": "good", "knowledge": "know",
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
    "LEADERS": ["elder", "strong-one", "tribe"],
}

# LEADERS is intentionally NOT folded into the general substitution table
# below: "elder"/"tribe" already belong to PEOPLE, and merging LEADERS in
# would let ordinary variation substitution drift across unrelated groups.
# It's used narrowly, by name, inside enforce_cro_reply_rules()'s dedup fix
# instead (preferred fallback pool when the duplicated token is a leader
# word), and as the source for the person-first-swap rule's rarer sibling.
_SUBSTITUTION_GROUPS = {k: v for k, v in SYNONYM_GROUPS.items() if k != "LEADERS"}
WORD_TO_GROUP = {
    word: group for group, words in _SUBSTITUTION_GROUPS.items() for word in words
}
LEADERS = tuple(SYNONYM_GROUPS["LEADERS"])

# Deduped per the task spec (the user's literal list repeated "cave" twice).
NOT_PERSON_START = (
    "tribe", "elder", "water", "fire", "cave", "wolf", "together",
    "stranger", "child", "animal", "snake", "stone", "forest", "river",
)

SEED_DIALOGUES = {
    "danger": [
        [["enemy", "come", "near"], ["tribe", "run", "fast"], ["question", "cave", "far"], ["cave", "up", "hill"], ["wolf", "here", "now"], ["hide", "cave", "safe"]],
        [["wolf", "near", "water"], ["elder", "call", "tribe"], ["tribe", "come", "fast"], ["together", "fight", "strong"], ["enemy", "run", "far"], ["person", "safe", "good"]],
        [["snake", "here", "danger"], ["tribe", "hide", "cave"], ["snake", "go", "far"], ["cave", "safe", "warm"], ["person", "come", "there"], ["fire", "here", "good"]],
        [["enemy", "many", "strong"], ["run", "far", "fast"], ["enemy", "come", "now"], ["water", "hide", "there"], ["person", "safe", "now"], ["tribe", "together", "good"]],
        [["stranger", "come", "near"], ["elder", "see", "stranger"], ["stranger", "go", "far"], ["tribe", "safe", "good"], ["person", "sleep", "now"], ["together", "safe", "warm"]],
        [["danger", "here", "now"], ["fire", "stay", "danger"], ["fire", "stay", "strong"], ["enemy", "fear", "fire"], ["enemy", "go", "far"], ["tribe", "safe", "now"]],
        [["wolf", "hunt", "person"], ["stone", "throw", "wolf"], ["wolf", "run", "far"], ["person", "safe", "good"], ["question", "wolf", "far"], ["wolf", "far", "forest"]],
        [["enemy", "see", "person"], ["hide", "fast", "now"], ["enemy", "walk", "far"], ["together", "come", "there"], ["question", "danger", "far"], ["danger", "far", "far"]],
        [["snake", "near", "cave"], ["elder", "call", "tribe"], ["tribe", "go", "far"], ["snake", "stay", "cave"], ["cave", "safe", "now"], ["tribe", "sleep", "safe"]],
        [["many", "enemy", "come"], ["run", "river", "fast"], ["river", "hide", "person"], ["enemy", "not", "see"], ["enemy", "go", "far"], ["tribe", "come", "here"]],
    ],
    "food": [
        [["person", "hungry", "now"], ["hunt", "animal", "forest"], ["animal", "run", "fast"], ["elder", "find", "path"], ["path", "go", "forest"], ["meat", "here", "much"]],
        [["question", "meat", "far"], ["meat", "river", "far"], ["person", "go", "river"], ["fish-food", "much", "good"], ["tribe", "eat", "together"], ["tribe", "full", "good"]],
        [["child", "hungry", "much"], ["give", "fruit", "child"], ["child", "eat", "fruit"], ["child", "good", "now"], ["much", "fruit", "far"], ["fruit", "forest", "there"]],
        [["tribe", "hungry", "cold"], ["find", "root", "here"], ["root", "good", "eat"], ["gather", "much", "fast"], ["tribe", "eat", "root"], ["tribe", "warm", "full"]],
        [["question", "meat", "far"], ["meat", "hill", "far"], ["person", "walk", "far"], ["meat", "much", "good"], ["tribe", "take", "meat"], ["tribe", "full", "strong"]],
        [["animal", "near", "here"], ["elder", "hunt", "animal"], ["animal", "run", "fast"], ["tribe", "throw", "stone"], ["animal", "dead", "now"], ["meat", "here", "good"]],
        [["person", "find", "fruit"], ["fruit", "good", "good"], ["tribe", "come", "eat"], ["share", "fruit", "together"], ["tribe", "eat", "good"], ["tribe", "full", "good"]],
        [["fish", "here", "much"], ["tribe", "hunt", "fish-food"], ["fish-food", "good"], ["tribe", "eat", "together"], ["question", "much", "far"], ["river", "much", "fish-food"]],
        [["plant-food", "here"], ["gather", "plant-food", "much"], ["plant-food", "good"], ["tribe", "full", "warm"], ["much", "plant-food", "far"], ["forest", "much", "there"]],
        [["cook", "meat", "now"], ["fire", "good", "cook"], ["meat", "cook", "good"], ["tribe", "eat", "together"], ["tribe", "full", "now"], ["tribe", "strong", "good"]],
    ],
    "water": [
        [["person", "thirsty", "now"], ["river", "near", "here"], ["person", "go", "river"], ["water", "cold", "good"], ["tribe", "thirsty", "much"], ["river", "much", "water"]],
        [["question", "water", "far"], ["water", "that-way", "far"], ["person", "walk", "far"], ["water", "river", "good"], ["tribe", "take", "water"], ["tribe", "good", "strong"]],
        [["child", "thirsty", "hurt"], ["give", "water", "child"], ["child", "take", "now"], ["child", "healthy", "good"], ["question", "much", "water"], ["river", "much", "here"]],
        [["water", "far", "far"], ["elder", "know", "path"], ["path", "go", "far"], ["path", "hill", "river"], ["tribe", "find", "water"], ["water", "cold", "good"]],
        [["rain", "come", "now"], ["gather", "water", "much"], ["water", "much", "good"], ["tribe", "not", "thirsty"], ["rain", "go", "far"], ["water", "stay", "good"]],
        [["question", "river", "far"], ["river", "down", "hill"], ["tribe", "go", "hill"], ["river", "here", "cold"], ["tribe", "take", "good"], ["tribe", "strong", "now"]],
        [["water", "bad", "here"], ["elder", "call", "tribe"], ["question", "good", "water"], ["good", "water", "far"], ["tribe", "walk", "far"], ["water", "good", "cold"]],
        [["person", "hurt", "thirsty"], ["water", "help", "person"], ["give", "water", "person"], ["person", "healthy", "now"], ["person", "strong", "now"], ["water", "good", "here"]],
        [["water", "cold", "good"], ["tribe", "take", "much"], ["tribe", "take", "water"], ["tribe", "full", "good"], ["question", "much", "water"], ["river", "much", "water"]],
        [["thirsty", "much", "now"], ["cave", "water", "here"], ["cave", "water", "good"], ["tribe", "take", "cold"], ["tribe", "good", "now"], ["water", "help", "tribe"]],
    ],
    "fire": [
        [["person", "cold", "now"], ["make", "fire", "here"], ["fire", "good", "warm"], ["tribe", "come", "fire"], ["tribe", "warm", "now"], ["fire", "stay", "strong"]],
        [["fire", "go", "there"], ["elder", "make", "fire"], ["fire", "come", "here"], ["tribe", "warm", "safe"], ["question", "tree", "far"], ["tree", "forest", "near"]],
        [["child", "cold", "hurt"], ["fire", "warm", "child"], ["child", "come", "fire"], ["child", "warm", "good"], ["fire", "stay", "good"], ["tribe", "safe", "warm"]],
        [["rain", "hurt", "fire"], ["hide", "fire", "cave"], ["fire", "stay", "good"], ["tribe", "safe", "warm"], ["rain", "go", "far"], ["fire", "strong", "good"]],
        [["sun-down", "come", "cold"], ["build", "fire", "big"], ["fire", "stay", "danger"], ["tribe", "sleep", "safe"], ["sun", "up", "come"], ["fire", "stay", "good"]],
        [["question", "fire", "far"], ["fire", "cave", "here"], ["tribe", "go", "cave"], ["fire", "warm", "tribe"], ["tribe", "safe", "warm"], ["fire", "stay", "good"]],
        [["fire", "danger", "near"], ["tribe", "run", "fast"], ["fire", "go", "that-way"], ["tribe", "safe", "here"], ["question", "fire", "dead"], ["fire", "far", "now"]],
        [["cold", "much", "now"], ["fire", "help", "cold"], ["fire", "warm", "good"], ["tribe", "together", "warm"], ["tribe", "sleep", "now"], ["fire", "see", "tribe"]],
        [["make", "fire", "fast"], ["elder", "make", "fire"], ["fire", "come", "good"], ["tribe", "warm", "safe"], ["tribe", "warm", "now"], ["fire", "stay", "much"]],
        [["stone", "make", "fire"], ["elder", "throw", "stone"], ["fire", "come", "now"], ["tribe", "good", "warm"], ["question", "far", "fire"], ["stone", "throw", "fast"]],
    ],
    "shelter": [
        [["rain", "come", "strong"], ["tribe", "go", "cave"], ["cave", "safe", "warm"], ["tribe", "sleep", "cave"], ["rain", "go", "far"], ["tribe", "come", "there"]],
        [["wind", "cold", "strong"], ["build", "cave", "fast"], ["cave", "good", "strong"], ["tribe", "warm", "safe"], ["question", "cave", "far"], ["cave", "hill", "there"]],
        [["snow", "come", "now"], ["cave", "cave", "tribe"], ["snow", "cold", "much"], ["fire", "cave", "warm"], ["tribe", "safe", "warm"], ["snow", "go", "far"]],
        [["sun-down", "cold", "sun-down"], ["cave", "safe", "here"], ["tribe", "go", "cave"], ["fire", "warm", "cave"], ["tribe", "sleep", "safe"], ["cave", "good", "warm"]],
        [["question", "cave", "far"], ["cave", "hill", "far"], ["tribe", "walk", "hill"], ["cave", "here", "safe"], ["tribe", "go", "cave"], ["cave", "warm", "good"]],
        [["rain", "hurt", "tribe"], ["find", "cave", "fast"], ["cave", "here", "good"], ["tribe", "come", "fast"], ["tribe", "safe", "warm"], ["cave", "help", "tribe"]],
        [["child", "cold", "rain"], ["carry", "child", "cave"], ["child", "warm", "now"], ["cave", "safe", "good"], ["tribe", "tribe", "warm"], ["together", "safe", "cave"]],
        [["build", "cave", "here"], ["elder", "see", "tribe"], ["cave", "strong", "good"], ["tribe", "make", "together"], ["cave", "good", "good"], ["tribe", "safe", "warm"]],
        [["old", "cave", "hurt"], ["build", "cave", "new"], ["tribe", "build", "fast"], ["cave", "strong", "good"], ["tribe", "safe", "now"], ["cave", "stay", "much"]],
        [["danger", "cave", "far"], ["cave", "safe", "near"], ["tribe", "run", "cave"], ["cave", "hide", "tribe"], ["danger", "go", "far"], ["tribe", "safe", "cave"]],
    ],
    "sleep": [
        [["tribe", "tired", "much"], ["sleep", "cave", "safe"], ["person", "see", "sun-down"], ["tribe", "sleep", "good"], ["sun", "up", "come"], ["tribe", "wake", "strong"]],
        [["child", "sleep", "now"], ["person", "see", "child"], ["child", "sleep", "good"], ["child", "wake", "warm"], ["child", "strong", "now"], ["sleep", "help", "child"]],
        [["enemy", "near", "sun-down"], ["person", "not", "sleep"], ["person", "see", "sun-down"], ["fire", "stay", "enemy"], ["sun", "up", "safe"], ["tribe", "sleep", "now"]],
        [["person", "wake", "now"], ["sun", "up", "come"], ["tribe", "wake", "good"], ["tribe", "strong", "eat"], ["question", "meat", "far"], ["meat", "forest", "near"]],
        [["sleep", "here", "safe"], ["fire", "see", "tribe"], ["tribe", "sleep", "good"], ["elder", "see", "sun-down"], ["sun", "up", "come"], ["elder", "wake", "tribe"]],
        [["tired", "much", "now"], ["sleep", "here", "good"], ["person", "sleep", "now"], ["tribe", "see", "safe"], ["person", "wake", "strong"], ["good", "sleep", "help"]],
        [["sun-down", "much", "cold"], ["fire", "warm", "tribe"], ["tribe", "sleep", "fire"], ["fire", "stay", "much"], ["sun", "up", "warm"], ["tribe", "wake", "good"]],
        [["question", "sleep", "far"], ["sleep", "cave", "safe"], ["tribe", "go", "cave"], ["tribe", "sleep", "warm"], ["tribe", "wake", "good"], ["cave", "safe", "sleep"]],
        [["hurt", "person", "sleep"], ["sleep", "help", "hurt"], ["person", "wake", "good"], ["tribe", "help", "person"], ["person", "strong", "now"], ["sleep", "good", "good"]],
        [["elder", "tired", "now"], ["elder", "sleep", "here"], ["tribe", "help", "elder"], ["elder", "sleep", "good"], ["elder", "wake", "strong"], ["tribe", "good", "together"]],
    ],
    "weather": [
        [["rain", "come", "strong"], ["tribe", "go", "cave"], ["rain", "much", "cold"], ["fire", "warm", "cave"], ["rain", "go", "far"], ["tribe", "come", "there"]],
        [["wind", "cold", "strong"], ["tribe", "make", "fire"], ["fire", "good", "warm"], ["wind", "not", "hurt", "tribe"], ["wind", "go", "far"], ["tribe", "safe", "warm"]],
        [["sun", "strong", "now"], ["find", "water", "fast"], ["water", "cold", "good"], ["tribe", "take", "much"], ["tribe", "good", "now"], ["sun", "warm", "good"]],
        [["snow", "come", "much"], ["cave", "cave", "tribe"], ["snow", "cold", "danger"], ["fire", "warm", "cave"], ["snow", "go", "far"], ["tribe", "come", "there", "good"]],
        [["question", "rain", "come"], ["rain", "come", "after"], ["tribe", "make", "cave"], ["cave", "good", "strong"], ["rain", "come", "now"], ["tribe", "safe", "cave"]],
        [["rain", "danger", "here"], ["tribe", "hide", "cave"], ["rain", "much", "strong"], ["cave", "safe", "tribe"], ["rain", "go", "far"], ["tribe", "come", "there", "safe"]],
        [["cold", "much", "now"], ["fire", "help", "tribe"], ["fire", "warm", "good"], ["tribe", "safe", "warm"], ["question", "cold", "go"], ["cold", "go", "sun", "up"]],
        [["sky", "sun-down", "now"], ["rain", "come", "now"], ["tribe", "find", "cave"], ["cave", "here", "good"], ["rain", "come", "now"], ["tribe", "safe", "safe"]],
        [["sun", "go", "far"], ["cold", "come", "now"], ["tribe", "make", "fire"], ["fire", "warm", "tribe"], ["tribe", "safe", "warm"], ["sun", "come", "here"]],
        [["question", "rain", "come"], ["sun", "come", "after"], ["tribe", "go", "hunt"], ["hunt", "good", "sun"], ["tribe", "find", "meat"], ["meat", "much", "good"]],
    ],
    "seasons": [
        [["sun", "strong", "now"], ["hunt", "animal", "much"], ["meat", "much", "good"], ["tribe", "gather", "tribe"], ["tribe", "full", "strong"], ["good", "many-days", "now"]],
        [["cold", "come", "now"], ["tribe", "gather", "meat"], ["meat", "much", "good"], ["tribe", "safe", "cold"], ["question", "cold", "much"], ["cold", "many", "many-days"]],
        [["many", "many-days", "cold"], ["tribe", "stay", "cave"], ["meat", "little", "now"], ["elder", "find", "root"], ["root", "good", "eat"], ["tribe", "not", "hungry"]],
        [["animal", "come", "here"], ["tribe", "hunt", "now"], ["meat", "much", "good"], ["tribe", "full", "strong"], ["question", "now", "hunt"], ["hunt", "sun", "up"]],
        [["fruit", "come", "now"], ["tribe", "gather", "fruit"], ["fruit", "much", "good"], ["tribe", "eat", "together"], ["tribe", "full", "good"], ["good", "many-days", "here"]],
        [["rain", "much", "now"], ["plant-food", "strong", "good"], ["plant-food", "much"], ["tribe", "gather", "tribe"], ["tribe", "full", "warm"], ["rain", "good", "many-days"]],
        [["cold", "go", "far"], ["sun", "come", "here", "warm"], ["tribe", "go", "there"], ["hunt", "animal", "now"], ["meat", "here", "good"], ["tribe", "strong", "after"]],
        [["question", "meat", "many-days"], ["meat", "much", "sun"], ["tribe", "hunt", "gather"], ["meat", "tribe", "many"], ["tribe", "full", "safe"], ["good", "many-days", "here"]],
        [["snow", "go", "far"], ["warm", "come", "warm"], ["animal", "come", "here"], ["tribe", "hunt", "much"], ["meat", "much", "after"], ["tribe", "full", "strong"]],
        [["much", "cold", "before"], ["warm", "come", "after"], ["tribe", "stay", "warm"], ["sun", "come", "here"], ["tribe", "hunt", "now"], ["meat", "much", "good"]],
    ],
    "direction": [
        [["question", "cave", "far"], ["cave", "that-way", "far"], ["person", "go", "that-way"], ["cave", "hill", "there"], ["person", "find", "cave"], ["cave", "safe", "good"]],
        [["question", "water", "far"], ["water", "this-way", "near"], ["tribe", "go", "fast"], ["water", "river", "here"], ["tribe", "take", "good"], ["water", "cold", "good"]],
        [["enemy", "that-way"], ["tribe", "go", "this-way"], ["tribe", "run", "fast"], ["enemy", "not", "come"], ["tribe", "safe", "now"], ["go", "far", "good"]],
        [["question", "tribe", "far"], ["tribe", "up", "hill"], ["person", "go", "up"], ["tribe", "here", "safe"], ["person", "find", "tribe"], ["tribe", "good", "safe"]],
        [["meat", "that-way", "far"], ["tribe", "walk", "far"], ["tribe", "find", "meat"], ["meat", "much", "good"], ["tribe", "eat", "good"], ["come", "here", "here"]],
        [["question", "path", "far"], ["path", "down", "river"], ["tribe", "go", "down"], ["path", "good", "safe"], ["tribe", "find", "river"], ["river", "water", "good"]],
        [["cave", "this-way"], ["tribe", "go", "fast"], ["cave", "near", "here"], ["tribe", "safe", "warm"], ["question", "far", "far"], ["cave", "near", "now"]],
        [["sun", "up", "that-way"], ["go", "that-way", "sun-up"], ["tribe", "walk", "sun-up"], ["find", "meat", "there"], ["meat", "good", "much"], ["tribe", "eat", "good"]],
        [["question", "enemy", "far"], ["enemy", "far", "that-way"], ["tribe", "safe", "here"], ["enemy", "not", "come"], ["tribe", "sleep", "safe"], ["enemy", "far", "good"]],
        [["go", "up", "hill"], ["hill", "see", "far", "land"], ["tribe", "see", "land"], ["land", "good", "there"], ["tribe", "go", "land"], ["land", "much", "meat"]],
    ],
    "distance": [
        [["enemy", "near", "now"], ["tribe", "run", "far"], ["enemy", "not", "come"], ["tribe", "safe", "far"], ["question", "far", "far"], ["far", "much", "safe"]],
        [["water", "far", "far"], ["elder", "know", "path"], ["tribe", "walk", "much"], ["water", "near", "now"], ["tribe", "take", "good"], ["good", "walk", "far"]],
        [["cave", "near", "here"], ["tribe", "go", "cave"], ["cave", "safe", "warm"], ["tribe", "safe", "near"], ["tribe", "good", "now"], ["near", "cave", "good"]],
        [["meat", "far", "far"], ["tribe", "walk", "far"], ["tribe", "tired", "much"], ["meat", "good", "far"], ["tribe", "find", "meat"], ["meat", "much", "good"]],
        [["question", "far", "near"], ["cave", "near", "hill"], ["water", "far", "river"], ["river", "that-way", "far"], ["tribe", "take", "cave"], ["cave", "near", "good"]],
        [["tribe", "far", "far"], ["call", "tribe", "strong"], ["tribe", "hear", "call"], ["tribe", "come", "fast"], ["tribe", "here", "now"], ["together", "safe", "good"]],
        [["danger", "near", "now"], ["run", "far", "fast"], ["run", "that-way", "that-way"], ["run", "that-way", "far"], ["tribe", "safe", "far"], ["danger", "not", "come"]],
        [["cave", "far", "cold"], ["walk", "fast", "warm"], ["cave", "near", "now"], ["tribe", "go", "cave"], ["tribe", "warm", "safe"], ["far", "good", "walk"]],
        [["question", "near", "far"], ["enemy", "far", "good"], ["water", "near", "good"], ["cave", "near", "good"], ["tribe", "take", "cave"], ["cave", "near", "safe"]],
        [["hunt", "near", "here"], ["animal", "near", "forest"], ["tribe", "hunt", "near"], ["meat", "here", "good"], ["tribe", "full", "good"], ["hunt", "near", "good"]],
    ],
    "time-of-day": [
        [["sun", "up", "now"], ["tribe", "hunt", "animal"], ["animal", "near", "forest"], ["tribe", "find", "meat"], ["meat", "good", "much"], ["tribe", "eat", "strong"]],
        [["sun", "down", "come"], ["tribe", "make", "fire"], ["fire", "good", "warm"], ["tribe", "sleep", "safe"], ["sun-down", "much", "cold"], ["fire", "stay", "much"]],
        [["sun-down", "come", "now"], ["fire", "stay", "sun-down"], ["tribe", "fear", "sun-down"], ["fire", "help", "fear"], ["tribe", "safe", "fire"], ["sun-down", "not", "hurt"]],
        [["sun", "up", "strong"], ["go", "water", "now"], ["water", "cold", "good"], ["tribe", "take", "much"], ["tribe", "strong", "now"], ["good", "sun", "up"]],
        [["sun-down", "see", "now"], ["elder", "see", "tribe"], ["tribe", "sleep", "safe"], ["elder", "see", "far"], ["sun", "up", "come"], ["elder", "wake", "tribe"]],
        [["question", "now", "hunt"], ["hunt", "sun", "up"], ["tribe", "wake", "sun-up"], ["animal", "near", "sun-up"], ["tribe", "hunt", "good"], ["sun-up", "hunt", "good"]],
        [["sun", "go", "down"], ["tribe", "come", "cave"], ["tribe", "eat", "together"], ["fire", "warm", "tribe"], ["tribe", "sleep", "good"], ["sun", "up", "come"]],
        [["much", "sun-down", "cold"], ["fire", "stay", "tribe", "sun-down"], ["tribe", "warm", "safe"], ["fire", "see", "tribe"], ["sun", "up", "warm"], ["tribe", "wake", "good"]],
        [["question", "sun", "up"], ["sun", "up", "now"], ["tribe", "stay", "warm"], ["sun", "come", "fast"], ["sun", "up", "now"], ["tribe", "wake", "good"]],
        [["hunt", "before", "sun-down"], ["tribe", "go", "now"], ["tribe", "find", "animal"], ["meat", "here", "good"], ["tribe", "eat", "sun-down"], ["fire", "sun", "eat"]],
    ],
    "body-states": [
        [["person", "hurt", "bad"], ["elder", "help", "person"], ["elder", "give", "root"], ["root", "help", "hurt"], ["person", "good", "now"], ["sleep", "good", "good"]],
        [["elder", "sick", "now"], ["tribe", "give", "meat"], ["elder", "eat", "good"], ["elder", "strong", "now"], ["elder", "good", "good"], ["tribe", "help", "elder"]],
        [["child", "cold", "hurt"], ["fire", "warm", "child"], ["child", "warm", "good"], ["child", "healthy", "now"], ["child", "strong", "good"], ["child", "good", "now"]],
        [["person", "tired", "hurt"], ["person", "sleep", "now"], ["person", "wake", "good"], ["sleep", "good", "person"], ["person", "strong", "now"], ["sleep", "help", "much"]],
        [["tribe", "hungry", "cold"], ["find", "meat", "fire"], ["eat", "together", "warm"], ["tribe", "good", "now"], ["tribe", "strong", "good"], ["meat", "fire", "help"]],
        [["person", "fear", "much"], ["tribe", "come", "person"], ["together", "safe", "now"], ["fear", "go", "far"], ["person", "good", "now"], ["tribe", "help", "fear"]],
        [["person", "thirsty", "hurt"], ["water", "help", "person"], ["person", "take", "much"], ["person", "healthy", "good"], ["person", "strong", "now"], ["water", "good", "good"]],
        [["elder", "pain", "now"], ["give", "elder", "sleep"], ["elder", "sleep", "good"], ["pain", "go", "far"], ["elder", "good", "now"], ["sleep", "good", "elder"]],
        [["strong", "person", "go"], ["tribe", "come", "strong"], ["tribe", "safe", "strong"], ["strong", "go", "good"], ["tribe", "good", "now"], ["strong", "help", "tribe"]],
        [["person", "healthy", "good"], ["healthy", "help", "tribe"], ["tribe", "tribe", "good"], ["tribe", "strong", "together"], ["together", "good", "now"], ["healthy", "good", "give"]],
    ],
    "us-vs-them": [
        [["stranger", "come", "here"], ["elder", "see", "stranger"], ["stranger", "good", "bad"], ["see", "before", "safe"], ["stranger", "give", "meat"], ["stranger", "good", "now"]],
        [["ours", "tribe", "strong"], ["theirs", "tribe", "tired"], ["theirs", "tribe", "go"], ["ours", "tribe", "safe"], ["tribe", "safe", "good"], ["strong", "tribe", "strong"]],
        [["stranger", "take", "meat"], ["tribe", "stay", "stranger"], ["stranger", "give", "here"], ["meat", "ours", "good"], ["stranger", "go", "far"], ["tribe", "safe", "good"]],
        [["theirs", "enemy", "come"], ["tribe", "together", "fight"], ["enemy", "run", "far"], ["tribe", "safe", "good"], ["together", "strong", "good"], ["ours", "tribe", "strong"]],
        [["stranger", "good", "come"], ["tribe", "give", "meat"], ["stranger", "give", "stone"], ["share", "good", "good"], ["together", "good", "now"], ["stranger", "tribe", "now"]],
        [["question", "ours", "theirs"], ["stone", "see", "ours"], ["theirs", "stay", "far"], ["ours", "land", "here"], ["far", "good", "now"], ["safe", "good", "tribe"]],
        [["theirs", "tribe", "hunt", "here"], ["elder", "call", "theirs"], ["theirs", "go", "far"], ["land", "ours", "good"], ["tribe", "safe", "land"], ["call", "good", "fight"]],
        [["stranger", "child", "alone"], ["tribe", "help", "child"], ["child", "safe", "now"], ["help", "good", "same"], ["stranger", "good", "tribe"], ["good", "good", "good"]],
        [["ours", "tribe", "strong"], ["much", "person", "come"], ["tribe", "give", "tribe"], ["together", "tribe", "strong"], ["big", "tribe", "good"], ["together", "safe", "tribe"]],
        [["theirs", "tribe", "take"], ["share", "meat", "theirs"], ["theirs", "tribe", "good"], ["give", "help", "good"], ["together", "tribe", "good"], ["share", "make", "strong"]],
    ],
    "strength": [
        [["enemy", "big", "strong"], ["tribe", "together", "fight"], ["together", "strong", "good"], ["enemy", "run", "far"], ["tribe", "strong", "good"], ["together", "fight", "strong"]],
        [["person", "tired", "hurt"], ["tribe", "help", "person"], ["person", "strong", "now"], ["help", "make", "strong"], ["person", "go", "now"], ["strong", "go", "tribe"]],
        [["strong-one", "come"], ["tribe", "safe", "good"], ["strong-one", "fight"], ["enemy", "run", "far"], ["tribe", "safe", "now"], ["strong-one", "stay"]],
        [["weak-one", "hungry"], ["tribe", "give", "meat"], ["weak-one", "strong", "now"], ["meat", "help", "tired"], ["together", "tribe", "strong"], ["tribe", "help", "tribe"]],
        [["enemy", "strong", "many"], ["tribe", "run", "fast"], ["tribe", "safe", "far"], ["alive", "fight", "after"], ["tribe", "together", "now"], ["together", "know", "good"]],
        [["question", "elder", "strong"], ["elder", "many", "strong"], ["elder", "go", "tribe"], ["elder", "know", "much"], ["tribe", "come", "elder"], ["elder", "go", "good"]],
        [["child", "strong", "strong"], ["know", "child", "hunt"], ["child", "know", "fast"], ["child", "strong", "now"], ["child", "hunt", "good"], ["tribe", "strong", "after"]],
        [["animal", "strong", "big"], ["tribe", "hunt", "together"], ["together", "hunt", "animal"], ["animal", "dead", "down"], ["meat", "much", "good"], ["together", "hunt", "good"]],
        [["rain", "strong", "come"], ["tribe", "stay", "together"], ["rain", "go", "far"], ["tribe", "stay", "strong"], ["tribe", "safe", "good"], ["together", "stay", "tribe"]],
        [["person", "carry", "much"], ["strong", "carry", "help"], ["together", "carry", "tribe"], ["share", "carry", "good"], ["tribe", "come", "good"], ["together", "strong", "same"]],
    ],
    "together-alone": [
        [["person", "alone", "fear"], ["tribe", "come", "person"], ["together", "safe", "good"], ["alone", "not", "good"], ["person", "good", "now"], ["together", "same", "good"]],
        [["tribe", "together", "strong"], ["enemy", "not", "come"], ["tribe", "safe", "here"], ["together", "stay", "safe"], ["together", "same", "good"], ["tribe", "strong", "together"]],
        [["person", "go", "alone"], ["elder", "call", "person"], ["alone", "danger", "much"], ["take", "person", "together"], ["one", "go", "together"], ["safe", "together", "good"]],
        [["child", "alone", "cold"], ["tribe", "find", "child"], ["child", "together", "tribe"], ["child", "safe", "warm"], ["child", "good", "now"], ["alone", "not", "good", "child"]],
        [["alone", "hunt", "bad"], ["hunt", "together", "good"], ["tribe", "hunt", "together"], ["meat", "much", "much"], ["tribe", "eat", "good"], ["together", "hunt", "good"]],
        [["question", "alone", "together"], ["together", "same", "good"], ["tribe", "good", "good"], ["alone", "danger", "same"], ["tribe", "stay", "near"], ["together", "tribe", "strong"]],
        [["elder", "alone", "old"], ["tribe", "stay", "elder"], ["elder", "not", "alone"], ["tribe", "help", "elder"], ["elder", "good", "warm"], ["help", "together", "good"]],
        [["stranger", "alone", "alone"], ["tribe", "help", "stranger"], ["stranger", "together", "tribe"], ["stranger", "safe", "now"], ["stranger", "good", "now"], ["help", "same", "good"]],
        [["together", "make", "fast"], ["alone", "make", "slow"], ["tribe", "build", "cave"], ["cave", "good", "fast"], ["cave", "good", "strong"], ["together", "build", "good"]],
        [["person", "alone", "tribe"], ["tribe", "alone", "person"], ["person", "come", "here"], ["tribe", "good", "now"], ["together", "good", "after"], ["alone", "much", "bad"]],
    ],
    "exchange": [
        [["person", "give", "meat"], ["tribe", "give", "stone"], ["stone", "good", "strong"], ["good", "share", "good"], ["together", "good", "now"], ["share", "make", "together"]],
        [["elder", "give", "know"], ["young", "take", "know"], ["know", "help", "much"], ["elder", "give", "good"], ["young", "know", "good"], ["know", "give", "good"]],
        [["tribe", "share", "meat"], ["tribe", "eat", "together"], ["meat", "much", "good"], ["share", "make", "much"], ["tribe", "strong", "good"], ["share", "tribe", "strong"]],
        [["person", "take", "much"], ["tribe", "not", "give", "much"], ["person", "give", "here"], ["good", "share", "good"], ["together", "good", "now"], ["good", "same", "good"]],
        [["give", "water", "take"], ["take", "meat", "give"], ["share", "good", "good"], ["together", "tribe", "good"], ["give", "good", "same"], ["share", "together", "tribe"]],
        [["question", "give", "unknown"], ["give", "meat", "stone"], ["stone", "good", "strong"], ["meat", "meat", "good"], ["give", "good", "good"], ["together", "good", "now"]],
        [["stranger", "take", "give"], ["tribe", "see", "unknown"], ["stranger", "give", "stone"], ["tribe", "give", "fruit"], ["give", "good", "good"], ["stranger", "tribe", "now"]],
        [["tribe", "take", "water"], ["share", "meat", "water"], ["good", "give", "good"], ["together", "tribe", "good"], ["tribe", "take", "good"], ["give", "help", "take"]],
        [["give", "help", "take"], ["help", "come", "here"], ["tribe", "help", "tribe"], ["tribe", "help", "tribe"], ["together", "good", "tribe"], ["give", "take", "good"]],
        [["person", "give", "much"], ["tribe", "know", "give"], ["give", "build", "safe"], ["safe", "make", "tribe"], ["tribe", "strong", "safe"], ["give", "good", "good"]],
    ],
    "emotions": [
        [["person", "fear", "much"], ["tribe", "come", "help"], ["together", "fear", "little"], ["fear", "go", "far"], ["person", "good", "now"], ["together", "fight", "fear"]],
        [["tribe", "good", "now"], ["meat", "much", "good"], ["tribe", "eat", "together"], ["good", "tribe", "good"], ["good", "now", "now"], ["tribe", "good", "good"]],
        [["person", "bad", "alone"], ["tribe", "come", "person"], ["together", "warm", "good"], ["bad", "go", "far"], ["person", "good", "now"], ["together", "good", "bad"]],
        [["elder", "dead", "now"], ["tribe", "bad", "much"], ["elder", "good", "before"], ["tribe", "know", "elder"], ["know", "stay", "good"], ["elder", "alive", "know"]],
        [["child", "good", "healthy"], ["tribe", "good", "good"], ["child", "strong", "strong"], ["tribe", "good", "child"], ["after", "good", "tribe"], ["child", "good", "tribe"]],
        [["person", "bad", "now"], ["elder", "safe", "person"], ["person", "safe", "down"], ["bad", "not", "help"], ["person", "know", "good"], ["safe", "good", "good"]],
        [["tribe", "good", "hunt"], ["eat", "good", "together"], ["tribe", "good", "good"], ["good", "tribe", "together"], ["good", "know", "make"], ["good", "now", "good"]],
        [["person", "alone", "cold"], ["tribe", "warm", "person"], ["person", "know", "good"], ["together", "not", "alone"], ["tribe", "together", "good"], ["alone", "go", "far"]],
        [["fear", "before", "fight"], ["strong", "come", "together"], ["tribe", "stay", "strong"], ["fear", "go", "strong"], ["tribe", "strong", "good"], ["strong", "fight", "fear"]],
        [["question", "know", "bad"], ["alone", "good", "before"], ["know", "good", "help"], ["before", "make", "strong"], ["good", "know", "stay"], ["before", "go", "now"]],
    ],
    "young-old": [
        [["child", "not", "know"], ["elder", "know", "child"], ["child", "know", "good"], ["know", "strong", "slow"], ["child", "know", "much"], ["know", "same", "good"]],
        [["elder", "tired", "tired"], ["tribe", "help", "elder"], ["elder", "sleep", "good"], ["tribe", "carry", "elder"], ["elder", "safe", "good"], ["help", "elder", "good"]],
        [["young", "person", "strong"], ["elder", "know", "path"], ["together", "go", "safe"], ["young", "strong", "old", "know"], ["tribe", "good", "together"], ["old", "make", "together"]],
        [["child", "alone", "danger"], ["elder", "find", "child"], ["elder", "carry", "child"], ["child", "safe", "now"], ["child", "know", "safe"], ["elder", "stay", "young"]],
        [["old", "elder", "sick"], ["tribe", "stay", "help"], ["elder", "good", "now"], ["tribe", "good", "elder"], ["elder", "strong", "after"], ["tribe", "help", "elder"]],
        [["question", "elder", "know"], ["elder", "know", "much"], ["tribe", "call", "elder"], ["elder", "give", "know"], ["tribe", "know", "good"], ["elder", "know", "good"]],
        [["young", "hunt", "one"], ["elder", "see", "young"], ["young", "hunt", "good"], ["elder", "good", "young"], ["young", "know", "hunt"], ["know", "make", "good"]],
        [["elder", "know", "before"], ["before", "know", "now"], ["before", "help", "now"], ["elder", "know", "good"], ["tribe", "know", "before"], ["before", "go", "after"]],
        [["child", "strong", "fast"], ["elder", "see", "child", "strong"], ["child", "strong", "now"], ["elder", "good", "good"], ["child", "help", "tribe"], ["strong", "up", "good"]],
        [["question", "elder", "go"], ["elder", "go", "tribe"], ["elder", "go", "good"], ["know", "go", "good"], ["tribe", "come", "elder"], ["elder", "path", "good"]],
    ],
    "birth-death": [
        [["new", "person", "come"], ["tribe", "good", "good"], ["new", "person", "small"], ["tribe", "give", "help"], ["new", "person", "strong"], ["tribe", "after", "good"]],
        [["elder", "dead", "now"], ["tribe", "bad", "much"], ["elder", "good", "before"], ["tribe", "know", "elder"], ["elder", "alive", "know"], ["know", "stay", "alive"]],
        [["new", "child", "alive"], ["tribe", "good", "good"], ["child", "alive", "good"], ["tribe", "give", "help"], ["child", "strong", "strong"], ["tribe", "good", "child"]],
        [["person", "hurt", "bad"], ["tribe", "help", "much"], ["person", "alive", "good"], ["tribe", "help", "person"], ["person", "good", "now"], ["alive", "good", "give"]],
        [["animal", "dead", "here"], ["tribe", "take", "meat"], ["meat", "good", "much"], ["animal", "give", "meat"], ["tribe", "eat", "good"], ["animal", "dead", "eat"]],
        [["question", "alive", "dead"], ["elder", "know", "know"], ["elder", "call", "slow"], ["alive", "good", "good"], ["tribe", "hear", "good"], ["elder", "know", "much"]],
        [["new", "many-days", "alive"], ["alive", "come", "after"], ["animal", "tree", "strong"], ["alive", "after", "good"], ["tribe", "good", "alive"], ["new", "alive", "good"]],
        [["person", "near", "dead"], ["tribe", "gather", "near"], ["person", "know", "tribe"], ["together", "good", "dead"], ["person", "go", "safe"], ["tribe", "bad", "together"]],
        [["alive", "carry", "good"], ["new", "alive", "good"], ["tribe", "strong", "after"], ["new", "person", "give", "strong"], ["after", "see", "good"], ["alive", "good", "tribe"]],
        [["know", "dead", "good"], ["dead", "alive", "know"], ["know", "stay", "strong"], ["before", "eat", "now"], ["tribe", "know", "tribe"], ["know", "tribe", "together"]],
    ],
    "spirits-unknown": [
        [["strange", "spirit", "near"], ["tribe", "stay", "together"], ["fire", "stay", "strange"], ["strange", "fear", "fire"], ["strange", "go", "far"], ["fire", "stay", "tribe"]],
        [["bad", "spirit", "here"], ["elder", "know", "spirit"], ["elder", "call", "spirit"], ["spirit", "go", "far"], ["tribe", "safe", "now"], ["elder", "safe", "spirit"]],
        [["unknown", "spirit", "sun-down"], ["fire", "stay", "warm"], ["spirit", "go", "far"], ["fire", "fear", "unknown"], ["tribe", "safe", "now"], ["fire", "fight", "unknown"]],
        [["spirit", "walk", "sun-down"], ["tribe", "together", "stay"], ["fire", "stay", "strong"], ["spirit", "fear", "sun"], ["spirit", "go", "far"], ["sun", "fight", "spirit"]],
        [["strange", "animal", "come"], ["elder", "see", "animal"], ["animal", "go", "far"], ["elder", "know", "animal"], ["tribe", "safe", "good"], ["elder", "safe", "tribe"]],
        [["question", "spirit", "good"], ["spirit", "good", "see"], ["tribe", "good", "spirit"], ["spirit", "go", "tribe"], ["tribe", "safe", "now"], ["good", "carry", "safe"]],
        [["unknown", "spirit", "see"], ["elder", "know", "spirit"], ["tribe", "know", "now"], ["unknown", "little", "fear"], ["tribe", "safe", "good"], ["know", "fight", "unknown"]],
        [["spirit", "bad", "now"], ["tribe", "give", "give"], ["give", "safe", "spirit"], ["spirit", "good", "now"], ["tribe", "safe", "good"], ["give", "give", "spirit"]],
        [["strange", "sun", "sky"], ["elder", "know", "sun"], ["sun", "good", "see"], ["elder", "know", "see"], ["tribe", "good", "good"], ["see", "carry", "good"]],
        [["fear", "unknown", "many"], ["know", "fight", "fear"], ["know", "unknown", "good"], ["know", "safe", "fear"], ["tribe", "little", "fear"], ["know", "same", "good"]],
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


def enforce_cro_reply_rules(tokens, vocab_words, rng, strip_person=False):
    """Apply the hard CRO-reply rules to a single CRO turn's tokens:
      1. no token may appear twice (this also covers "'now' at most once",
         since "now" is just a token like any other)
      2. the reply may not start with "person"
      3. if `strip_person` (the reply directly answers a "question" HUM
         turn), "person" may not appear anywhere in the reply -- per the
         task spec's question->answer format ("direct factual answer
         without 'person'"), not just the leading-token rule
    Runs on every CRO turn at generation time -- seed originals and
    variations alike -- rather than hand-fixing violations in the seed
    literal, so it automatically also covers duplicates/person-starts
    introduced by synonym substitution."""
    out = list(tokens)
    seen = set()
    for i, tok in enumerate(out):
        if tok in seen:
            if tok in LEADERS:
                candidates = [w for w in LEADERS if w not in seen]
            else:
                group = WORD_TO_GROUP.get(tok)
                candidates = [w for w in SYNONYM_GROUPS[group] if w not in seen] if group else []
            if not candidates:
                candidates = [w for w in vocab_words if w not in seen and w != "question"]
            out[i] = rng.choice(sorted(candidates))
            seen.add(out[i])
        else:
            seen.add(tok)

    if strip_person:
        positions = [i for i, tok in enumerate(out) if tok == "person"]
    else:
        positions = [0] if out and out[0] == "person" else []
    for i in positions:
        others = set(out) - {out[i]}
        candidates = [w for w in NOT_PERSON_START if w not in others]
        out[i] = rng.choice(candidates or NOT_PERSON_START)
    return out


def to_dialogue(situation, turns, vocab_words, rng):
    fixed_turns = []
    for speaker, tokens in zip(SPEAKERS, turns):
        if speaker == "CRO":
            is_answer_to_question = (
                bool(fixed_turns)
                and fixed_turns[-1]["speaker"] == "HUM"
                and fixed_turns[-1]["tokens"][0] == "question"
            )
            tokens = enforce_cro_reply_rules(tokens, vocab_words, rng, strip_person=is_answer_to_question)
        fixed_turns.append({"speaker": speaker, "tokens": tokens})
    return {"situation": situation, "turns": fixed_turns}


def is_valid_dialogue(dialogue, vocab_words):
    for turn in dialogue["turns"]:
        tokens = turn["tokens"]
        if not (2 <= len(tokens) <= 4):
            return False
        if any(tok not in vocab_words for tok in tokens):
            return False
    return True


def compute_stats(dialogues):
    total = len(dialogues)
    cro_turns = 0
    cro_non_person_start = 0
    question_dialogues = 0
    question_answers_with_person = 0

    for dialogue in dialogues:
        turns = dialogue["turns"]
        has_question = False
        for i, turn in enumerate(turns):
            if turn["speaker"] == "CRO":
                cro_turns += 1
                if turn["tokens"][0] != "person":
                    cro_non_person_start += 1
            if turn["speaker"] == "HUM" and turn["tokens"][0] == "question":
                has_question = True
                if i + 1 < len(turns) and turns[i + 1]["speaker"] == "CRO":
                    if "person" in turns[i + 1]["tokens"]:
                        question_answers_with_person += 1
        if has_question:
            question_dialogues += 1

    return {
        "cro_turns": cro_turns,
        "cro_non_person_start_pct": 100 * cro_non_person_start / cro_turns if cro_turns else 0.0,
        "question_dialogues": question_dialogues,
        "question_dialogues_pct": 100 * question_dialogues / total if total else 0.0,
        "question_answers_with_person": question_answers_with_person,
    }


def main():
    rng = random.Random(SEED)
    vocab_words = load_vocab_words(EMBEDDINGS_PATH)

    dialogues = []
    per_situation_counts = {}
    failed_variations = 0

    for situation, seeds in SEED_DIALOGUES.items():
        situation_dialogues = []
        for seed in seeds:
            seed_dialogue = to_dialogue(situation, seed, vocab_words, rng)
            if not is_valid_dialogue(seed_dialogue, vocab_words):
                raise ValueError(f"Seed for '{situation}' failed validation: {seed}")
            situation_dialogues.append(seed_dialogue)

            produced = 0
            attempts = 0
            while produced < VARIATIONS_PER_SEED and attempts < MAX_ATTEMPTS_PER_VARIATION * VARIATIONS_PER_SEED:
                attempts += 1
                variation = make_variation(seed, rng)
                dialogue = to_dialogue(situation, variation, vocab_words, rng)
                if is_valid_dialogue(dialogue, vocab_words):
                    situation_dialogues.append(dialogue)
                    produced += 1
                else:
                    failed_variations += 1
            while produced < VARIATIONS_PER_SEED:
                # Should not happen with clean seeds/groups; guarantees the
                # exact per-situation count regardless.
                situation_dialogues.append(to_dialogue(situation, seed, vocab_words, rng))
                produced += 1

        dialogues.extend(situation_dialogues)
        per_situation_counts[situation] = len(situation_dialogues)

    stats = compute_stats(dialogues)

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
    print()
    print("Rule-compliance stats:")
    print(f"  CRO replies starting with a non-'person' token: {stats['cro_non_person_start_pct']:.1f}% of {stats['cro_turns']} CRO turns (target >= 30%)")
    print(f"  Dialogues containing a question->answer pair:   {stats['question_dialogues_pct']:.1f}% of {len(dialogues)} dialogues (target >= 20%)")
    print(f"  Question-answer CRO replies containing 'person': {stats['question_answers_with_person']} (target 0)")


if __name__ == "__main__":
    main()
