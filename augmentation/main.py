from __future__ import annotations

# Allow running this file directly (python augmentation/main.py) by ensuring
# the project root is on sys.path so that `import augmentation.*` works.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _root = str(_Path(__file__).resolve().parents[1])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

"""
Augmentation CLI for calendar-event extraction dataset.

Pipeline overview (per raw example):
1) Canonicalize schema
   - Raw uses top-level key "output". We keep records as {"event_text", "output"}.
   - The "output" dict includes exactly 8 keys: action, date, time, attendees, location, duration,
     recurrence, notes. Missing values are set to null (JSON), not empty strings.

2) Target normalization
   - date normalized to DD/MM/YYYY
   - time normalized to 12-hour "H:MM AM/PM" (no leading zero on hour)

3) Keep the original pair (event_text, output)

4) One-per-row augmentation (style diversity without duplication)
   - Build entity pools from the dataset (attendees, locations).
   - Candidate A: entity-swap using the pools (only if attendees/location exist). We update both
     event_text (conservative whole-word replacements) and output accordingly.
   - Candidate B: JSON→text rendering using multiple templates. Output stays identical; text varies in style.
   - We randomly pick exactly ONE of the available candidates per row (if any is different from the original).

5) Optional Faker synthesis (--synth N)
   - Generate N fully new (event_text, output) pairs using Faker. The generator:
     diversified actions, time/date variants (incl. relative forms and ranges), recurrence phrases,
     real-world sparsity (nulls for some fields), and light linguistic noise in event_text while
     keeping output clean. Location sampling is weighted (platforms, rooms, cafes, parks, etc.).

6) Dedup, shuffle, sanitize, and split
   - Dedupe by (event_text, output) signature.
   - Deterministic shuffle with --seed.
   - Sanitize: enforce unified key name ("output") and ensure nulls instead of empty strings.
   - Write consolidated file to data/processed/augmented.jsonl.
   - Create deterministic 75/15/10 splits under data/splits/{train,eval,test}.jsonl.
   - Drop near-duplicates across splits (train > eval > test precedence) to reduce leakage.

Counting logic (upper bounds before dedup):
- Let N = number of raw rows (e.g., N = 792)
- Originals: N
- Augmented: up to N (1 per raw row, chosen between entity-swap or JSON→text render)
- Faker synth: exactly --synth
- Total before dedup: ≤ N + N + --synth (dedup may reduce this)

Usage:
  python augmentation/main.py --synth 1000            # add 1000 Faker examples
  python augmentation/main.py --root /path/to/proj --synth 0 --seed 42

Outputs:
  - data/processed/augmented.jsonl
  - data/splits/{train.jsonl, eval.jsonl, test.jsonl}
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List

from augmentation.utils import read_jsonl, write_jsonl, canonicalize_output, dedupe_text_output, TARGET_KEYS
from augmentation.entity_pools import build_entity_pools
from augmentation.augmentors import swap_entities_with_pools, ensure_output_schema_row, drop_split_leakage
from augmentation.renderers import render_from_json
from augmentation.faker_synth import generate_faker_events


def normalize_date(date_str: str | None) -> str | None:
    if date_str is None or str(date_str).strip() == "":
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(str(date_str), dayfirst=True, fuzzy=True)
        return dt.strftime("%d/%m/%Y")
    except Exception:
        return None


def normalize_time(time_str: str | None) -> str | None:
    if time_str is None or str(time_str).strip() == "":
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(str(time_str), fuzzy=True)
        out = dt.strftime("%I:%M %p")
        return out.lstrip("0")
    except Exception:
        return None


def ensure_schema(o: Dict) -> Dict:
    return {k: o.get(k, None) for k in TARGET_KEYS}


def process(args: argparse.Namespace) -> int:
    project_root = Path(args.root).resolve() if args.root else Path.cwd()
    raw_path = project_root / "data/raw/event_text_mapping.jsonl"
    if not raw_path.exists():
        print(f"Raw dataset not found at {raw_path}")
        return 1

    rows_raw = read_jsonl(raw_path)
    rows = [canonicalize_output(r) for r in rows_raw]

    # Normalize targets and enforce schema
    for r in rows:
        o = ensure_schema(r["output"])  # enforce keys with None
        o["date"] = normalize_date(o.get("date"))
        o["time"] = normalize_time(o.get("time"))
        r["output"] = o

    # Build entity pools from data
    att_pool, loc_pool = build_entity_pools(rows)
    print(f"Entity pools → attendees: {len(att_pool)}, locations: {len(loc_pool)}")

    # Augmentation: exactly ONE new variant per row for style diversity
    pairs: List[Dict] = []
    for r in rows:
        text = r["event_text"].strip()
        o = ensure_schema(r["output"])  # enforce before append
        pairs.append({"event_text": text, "output": o})  # keep original

        # candidates: entity-swap (if applicable) and a JSON→text render
        candidates: List[Dict] = []
        nt, no = swap_entities_with_pools(text, o, att_pool, loc_pool)
        if nt and nt.strip() != text:
            candidates.append({"event_text": nt.strip(), "output": ensure_schema(no)})
        rendered = render_from_json(o)
        if rendered and rendered.strip() != text:
            candidates.append({"event_text": rendered.strip(), "output": o})

        if candidates:
            # Pick exactly one augmented variant to ensure 1-per-row augmentation
            choice = random.choice(candidates)
            pairs.append(choice)

    # Synthetic events via Faker (optional)
    synth = generate_faker_events(num_examples= args.synth if hasattr(args, 'synth') else 0)
    for s in synth:
        pairs.append({"event_text": s["event_text"], "output": ensure_schema(s.get("output") or s.get("json") or {})})

    # Deduplicate
    unique = dedupe_text_output(pairs)

    # Shuffle deterministically before writing augmented.jsonl
    rng_aug = random.Random(args.seed + 1 if hasattr(args, "seed") else 2025)
    shuffled_unique = unique[:]
    rng_aug.shuffle(shuffled_unique)

    # Sanity: enforce unified key name and nulls for missing (no empty strings)
    sanitized = [ensure_output_schema_row(p) for p in shuffled_unique]

    # Write under processed
    out_dir = project_root / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "augmented.jsonl"
    write_jsonl(out_file, sanitized)
    print(f"Wrote {len(sanitized)} examples → {out_file}")

    # Create deterministic train/eval/test splits (75/15/10)
    splits_dir = project_root / "data/splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng_split = random.Random(args.seed + 2 if hasattr(args, "seed") else 4242)
    split_order = sanitized[:]
    rng_split.shuffle(split_order)

    n_total = len(split_order)
    n_train = int(n_total * 0.75)
    n_eval = int(n_total * 0.15)
    n_test = n_total - n_train - n_eval

    train_set = split_order[:n_train]
    eval_set = split_order[n_train:n_train + n_eval]
    test_set = split_order[n_train + n_eval:]

    # Remove near-dupes across splits to avoid leakage
    train_set, eval_set, test_set = drop_split_leakage(train_set, eval_set, test_set)

    write_jsonl(splits_dir / "train.jsonl", train_set)
    write_jsonl(splits_dir / "eval.jsonl", eval_set)
    write_jsonl(splits_dir / "test.jsonl", test_set)
    print(
        f"Splits → train: {len(train_set)}, eval: {len(eval_set)}, test: {len(test_set)} → {splits_dir}"
    )
    return 0


def cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Augment and preprocess calendar-event dataset")
    parser.add_argument("--root", type=str, default="", help="Project root (defaults to CWD)")
    parser.add_argument("--synth", type=int, default=1000, help="Number of Faker-generated synthetic events to add")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic augmentation & splits")
    args = parser.parse_args(argv)
    random.seed(args.seed)
    return process(args)


if __name__ == "__main__":
    import sys
    raise SystemExit(cli(sys.argv[1:]))


