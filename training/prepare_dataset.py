"""
Prepare fine-tuning dataset for SecBERT IOC NER.

Sources:
  - naorm/malware-text-db-cyner  (~5k spans)
  - naorm/dnrti-cyner            (~14k spans)

Both are span-level annotations on pre-tokenized sentences.
This script:
  1. Merges both sources
  2. Groups spans by sentence
  3. Converts span-level → BIO token-level labels
  4. Tokenizes with SecBERT tokenizer + aligns labels to subwords
  5. Saves train/val/test splits as HuggingFace DatasetDict

Label scheme:
  B-IOC / I-IOC      ← CyNER 'Indicator'
  B-CVE / I-CVE      ← CyNER 'Vulnerability'
  B-MALWARE / I-MALWARE ← CyNER 'Malware'
  O                  ← everything else

Usage:
  python training/prepare_dataset.py [--output-dir data/cyner_ner]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "jackaduma/SecBERT"
LABEL_NAMES = ["O", "B-IOC", "I-IOC", "B-CVE", "I-CVE", "B-MALWARE", "I-MALWARE"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}

TYPE_MAP = {
    "Indicator": "IOC",
    "Vulnerability": "CVE",
    "Malware": "MALWARE",
    # Organization, System → O (not IOCs)
}

VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42


# ─── Span → BIO conversion ────────────────────────────────────────────────────

def _find_span_in_words(words: list[str], entity_words: list[str]) -> int | None:
    """
    Find starting index of entity_words as a contiguous sublist of words.
    Returns the start index or None if not found.
    """
    n, m = len(words), len(entity_words)
    for i in range(n - m + 1):
        if words[i : i + m] == entity_words:
            return i
    # Fallback: case-insensitive
    words_lower = [w.lower() for w in words]
    entity_lower = [w.lower() for w in entity_words]
    for i in range(n - m + 1):
        if words_lower[i : i + m] == entity_lower:
            return i
    return None


def sentence_to_bio(sentence: str, spans: list[dict]) -> tuple[list[str], list[str]]:
    """
    Convert a pre-tokenized sentence + list of span dicts to BIO labels.

    Args:
        sentence: space-separated words (already tokenized)
        spans: list of {"text": str, "type": str} dicts

    Returns:
        (words, bio_labels) — parallel lists
    """
    words = sentence.split()
    labels = ["O"] * len(words)

    # Sort by entity length descending so longer spans take priority
    spans_sorted = sorted(spans, key=lambda s: len(s["text"].split()), reverse=True)

    for span in spans_sorted:
        mapped = TYPE_MAP.get(span["type"])
        if mapped is None:
            continue
        entity_words = span["text"].split()
        start = _find_span_in_words(words, entity_words)
        if start is None:
            continue
        # Only label if not already labeled (longer spans win)
        if all(labels[start + i] == "O" for i in range(len(entity_words))):
            labels[start] = f"B-{mapped}"
            for i in range(1, len(entity_words)):
                labels[start + i] = f"I-{mapped}"

    return words, labels


# ─── Dataset loading + grouping ───────────────────────────────────────────────

def load_and_group(dataset_names: list[str]) -> dict[str, list[dict]]:
    """
    Load span datasets, group by sentence.
    Handles different column schemas across CyNER datasets:
      - malware-text-db-cyner: has 'Original Sentence ID', 'Fixed Text'
      - dnrti-cyner: only has 'Original Sentence', 'Text'
    Returns: {sentence_key: {"sentence": str, "spans": [...]}}
    """
    from datasets import load_dataset

    grouped: dict[str, dict] = {}

    for ds_name in dataset_names:
        log.info("Loading %s ...", ds_name)
        ds = load_dataset(ds_name, split="train")
        cols = ds.column_names

        for idx, row in enumerate(ds):
            # Build a stable key: prefer sentence ID if available, else hash sentence text
            if "Original Sentence ID" in cols:
                key = f"{ds_name}::{row['Original Sentence ID']}"
            else:
                key = f"{ds_name}::{hash(row['Original Sentence'])}"

            if key not in grouped:
                grouped[key] = {
                    "sentence": row["Original Sentence"],
                    "spans": [],
                }

            # Use Fixed Text if available (cleaner), else fall back to Text
            entity_text = row.get("Fixed Text") or row["Text"]
            grouped[key]["spans"].append({
                "text": entity_text.strip(),
                "type": row["Type"],
            })

    log.info("Loaded %d unique sentences", len(grouped))
    return grouped


# ─── Tokenizer alignment ──────────────────────────────────────────────────────

def tokenize_and_align(
    examples: dict,
    tokenizer,
    max_length: int = 512,
) -> dict:
    """
    HuggingFace map-compatible function.
    Takes batched examples with 'words' and 'word_labels' (string labels).
    Returns tokenized inputs with integer 'labels'.
    """
    tokenized = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    all_labels = []
    for i, word_labels in enumerate(examples["word_labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned: list[int] = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)  # [CLS], [SEP]
            elif word_id != prev_word_id:
                aligned.append(LABEL2ID[word_labels[word_id]])
            else:
                aligned.append(-100)  # subword continuation
            prev_word_id = word_id
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load and group spans
    grouped = load_and_group([
        "naorm/malware-text-db-cyner",
        "naorm/dnrti-cyner",
    ])

    # 2. Convert to BIO
    records: list[dict] = []
    skipped = 0
    for data in grouped.values():
        words, labels = sentence_to_bio(data["sentence"], data["spans"])
        if len(words) == 0:
            skipped += 1
            continue
        records.append({"words": words, "word_labels": labels})

    log.info("Converted %d sentences (%d skipped)", len(records), skipped)

    # Stats
    label_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for lbl in r["word_labels"]:
            label_counts[lbl] += 1
    log.info("Label distribution: %s", dict(sorted(label_counts.items())))

    # 3. Build HuggingFace Dataset and split
    full_ds = Dataset.from_list(records)
    full_ds = full_ds.shuffle(seed=SEED)

    n = len(full_ds)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_test - n_val

    splits = DatasetDict({
        "train": full_ds.select(range(n_train)),
        "validation": full_ds.select(range(n_train, n_train + n_val)),
        "test": full_ds.select(range(n_train + n_val, n)),
    })
    log.info("Split sizes — train: %d, val: %d, test: %d", n_train, n_val, n_test)

    # 4. Tokenize
    log.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_splits = splits.map(
        lambda ex: tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=["words", "word_labels"],
        desc="Tokenizing",
    )

    # 5. Save
    tokenized_splits.save_to_disk(str(output_path))

    # Also save label names alongside
    import json
    (output_path / "label_names.json").write_text(json.dumps(LABEL_NAMES, indent=2))

    log.info("Saved dataset to %s", output_path)
    log.info("Label names: %s", LABEL_NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CyNER dataset for SecBERT fine-tuning")
    parser.add_argument("--output-dir", default="data/cyner_ner", help="Where to save the processed dataset")
    args = parser.parse_args()
    main(args.output_dir)
