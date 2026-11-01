"""Build the prompt set for activation capture: a fixed, mixed sample of public prompts.

Writes data/prompts.csv with one row per prompt. output_text / output_token_ids start empty
and are filled in later by generating with the model; once the token ids are saved, every
token's activations can be reproduced exactly by teacher-forced replay (see tracer.py).

Sources (fetched through the Hugging Face datasets-server API, no extra dependencies):
    dolly   databricks/databricks-dolly-15k   everyday instructions, QA, summarising   CC BY-SA 3.0
    oasst   OpenAssistant/oasst1              real first-turn chat prompts, many langs  Apache 2.0
    gsm8k   openai/gsm8k                      grade-school maths word problems          MIT
    mbpp    google-research-datasets/mbpp     short Python coding tasks                 CC BY 4.0

    .venv/bin/python build_prompts.py            # 100 prompts, 25 per source
    .venv/bin/python build_prompts.py --n 400    # same mix, bigger
"""

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co"
OUT = Path(__file__).parent / "data" / "prompts.csv"
FIELDS = ["id", "source", "category", "lang", "prompt", "output_text", "output_token_ids"]
MAX_CHARS = 1500  # keep prompts short enough that the dataset stays about the model, not the context


def get(endpoint, **params):
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    for attempt in range(6):  # the filter endpoint answers 500 intermittently; retrying works
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == 5:
                raise
            time.sleep(1 + attempt)


def sample(rng, dataset, config, split, k, where=None):
    """k random rows of a split (optionally filtered), fetched as a few random pages."""
    endpoint, extra = ("filter", {"where": where}) if where else ("rows", {})
    total = get(endpoint, dataset=dataset, config=config, split=split, offset=0, length=1, **extra)["num_rows_total"]
    rows, seen = [], set()
    while len(rows) < k * 4 and len(seen) * 100 < total:
        offset = rng.randrange(0, max(1, total - 100))
        if offset // 100 in seen:
            continue
        seen.add(offset // 100)
        page = get(endpoint, dataset=dataset, config=config, split=split, offset=offset, length=100, **extra)
        rows += [r["row"] for r in page["rows"]]
    rng.shuffle(rows)
    return rows


def dolly(rng, k):
    out = []
    for r in sample(rng, "databricks/databricks-dolly-15k", "default", "train", k):
        prompt = r["instruction"].strip()
        if r["context"].strip():
            prompt += "\n\n" + r["context"].strip()
        if len(prompt) <= MAX_CHARS:
            out.append(dict(category=r["category"], lang="en", prompt=prompt))
    return out[:k]


def oasst(rng, k):
    # Root messages only (a user's opening prompt), about half English and half other languages.
    base = "\"parent_id\" IS NULL AND \"role\"='prompter'"
    out = []
    for where, n in [(base + " AND \"lang\"='en'", k - k // 2), (base + " AND \"lang\"<>'en'", k // 2)]:
        rows = [r for r in sample(rng, "OpenAssistant/oasst1", "default", "train", n, where)
                if not r["deleted"] and len(r["text"].strip()) <= MAX_CHARS]
        out += [dict(category="chat", lang=r["lang"], prompt=r["text"].strip()) for r in rows[:n]]
    return out


def gsm8k(rng, k):
    rows = sample(rng, "openai/gsm8k", "main", "train", k)
    return [dict(category="math", lang="en", prompt=r["question"].strip()) for r in rows[:k]]


def mbpp(rng, k):
    rows = sample(rng, "google-research-datasets/mbpp", "full", "train", k)
    return [dict(category="code", lang="en",
                 prompt=f"{r['text'].strip()} Your code should pass this test:\n{r['test_list'][0]}")
            for r in rows[:k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="total prompts, split evenly across sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sources = {"dolly": dolly, "oasst": oasst, "gsm8k": gsm8k, "mbpp": mbpp}
    per = [args.n // len(sources) + (i < args.n % len(sources)) for i in range(len(sources))]
    rows = []
    for (name, fn), k in zip(sources.items(), per):
        got = fn(rng, k)
        print(f"{name:6} {len(got)}/{k}")
        rows += [dict(source=name, **r) for r in got]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow(dict(id=i, output_text="", output_token_ids="", **r))
    print(f"wrote {len(rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
