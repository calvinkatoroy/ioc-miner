"""
Fine-tune SecBERT on CyNER IOC NER dataset.

Expects dataset prepared by prepare_dataset.py at --data-dir.

Outputs:
  - Fine-tuned model saved to --output-dir (default: models/secbert-ioc-ner)
  - Training logs to --output-dir/logs/
  - Evaluation results printed and saved to --output-dir/eval_results.json

Usage:
  python training/fine_tune_secbert.py
  python training/fine_tune_secbert.py --data-dir data/cyner_ner --epochs 5 --batch-size 16
  python training/fine_tune_secbert.py --fp16   # on CUDA GPU
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def make_compute_metrics(label_names: list[str]):
    """Returns a compute_metrics function using seqeval."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq, pred_seq_str = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_seq.append(label_names[l])
                pred_seq_str.append(label_names[p])
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_str)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    return compute_metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset + label names
    log.info("Loading dataset from %s", data_dir)
    dataset = load_from_disk(str(data_dir))
    label_names: list[str] = json.loads((data_dir / "label_names.json").read_text())
    id2label = {i: l for i, l in enumerate(label_names)}
    label2id = {l: i for i, l in enumerate(label_names)}
    num_labels = len(label_names)

    log.info("Labels (%d): %s", num_labels, label_names)
    log.info("Train: %d | Val: %d | Test: %d",
             len(dataset["train"]), len(dataset["validation"]), len(dataset["test"]))

    # 2. Load model + tokenizer
    log.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # 3. Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=50,
        fp16=args.fp16,
        report_to="none",  # disable wandb/tensorboard unless user sets up
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(label_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 5. Train
    log.info("Starting training...")
    trainer.train()

    # 6. Evaluate on test set
    log.info("Evaluating on test set...")
    test_results = trainer.evaluate(dataset["test"])
    log.info("Test results: %s", test_results)

    # Save eval results
    (output_dir / "eval_results.json").write_text(json.dumps(test_results, indent=2))

    # 7. Save best model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Model saved to %s", output_dir)

    # Save label names alongside the model
    (output_dir / "label_names.json").write_text(json.dumps(label_names, indent=2))

    # Print final report
    print("\n" + "=" * 60)
    print(f"  F1:        {test_results.get('eval_f1', 0):.4f}")
    print(f"  Precision: {test_results.get('eval_precision', 0):.4f}")
    print(f"  Recall:    {test_results.get('eval_recall', 0):.4f}")
    print(f"  Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SecBERT for IOC NER")
    parser.add_argument("--model", default="jackaduma/SecBERT", help="Base model ID or path")
    parser.add_argument("--data-dir", default="data/cyner_ner", help="Prepared dataset directory")
    parser.add_argument("--output-dir", default="models/secbert-ioc-ner", help="Save fine-tuned model here")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 (requires CUDA)")
    args = parser.parse_args()
    main(args)
