#!/usr/bin/env python3
"""
Train the intent classifier (map, path, mechanics, action, question, goal experts).

Run from project root. Saves the model to intent_model/oracle_intent_model by default.
Requires optional deps: pip install setfit datasets
Training examples are defined in oracle.intent (INTENT_TRAIN_EXAMPLES and derived lists).
The oracle uses this model to route player questions to the right expert.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from oracle.intent import Intent, INTENT_TRAIN_TEXTS, INTENT_TRAIN_LABELS


def load_data_from_file(path: Path) -> tuple[list[str], list[str]]:
    """Load texts and labels from a JSON file: {"texts": [...], "labels": [...]}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = data["texts"]
    labels = data["labels"]
    if len(texts) != len(labels):
        raise ValueError(f"texts and labels length mismatch: {len(texts)} vs {len(labels)}")
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the intent classifier for the CrafText Oracle (map, path, mechanics, action, question, goal experts).",
    )
    parser.add_argument(
        "--base-model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Base SetFit model to fine-tune (default: paraphrase-multilingual-mpnet-base-v2)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Directory to save the trained model (default: intent_model/oracle_intent_model)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional JSON file with keys 'texts' and 'labels' for custom training data",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="SetFit training iterations (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip sanity check after training",
    )
    args = parser.parse_args()

    save_path = args.save_path
    if save_path is None:
        save_path = str(PROJECT_ROOT / "intent_model" / "oracle_intent_model")
    else:
        save_path = str(Path(save_path).resolve())

    if args.data is not None:
        data_path = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
        if not data_path.exists():
            print(f"Error: data file not found: {data_path}", file=sys.stderr)
            sys.exit(1)
        train_texts, train_labels = load_data_from_file(data_path)
        print(f"Loaded {len(train_texts)} examples from {data_path}")
    else:
        train_texts = INTENT_TRAIN_TEXTS
        train_labels = INTENT_TRAIN_LABELS
        print(f"Using training data from oracle.intent ({len(train_texts)} examples)")

    print(f"Base model: {args.base_model}")
    print(f"Save path:  {save_path}")
    print(f"Iterations: {args.num_iterations}, batch size: {args.batch_size}")
    print("Training...")

    intent = Intent(model_name=args.base_model, use_model=True)
    intent.train(
        train_texts=train_texts,
        train_labels=train_labels,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        save_path=save_path,
    )

    print("Training finished. Model saved to:", save_path)

    if not args.no_check:
        print("\nSanity check (loaded model):")
        loaded = Intent(model_name=save_path, use_model=True)
        for q in [
            "Where is the nearest diamond?",
            "In which direction should I move to find stone?",
            "How can I navigate to coordinates [45, 54]?",
            "goal: go to the point [20, 3]. In which direction should I go now?",
            "What do I need to gather coal?",
            "What do I need to perform PLACE_TABLE?",
            "What do I need to perform MAKE_WOOD_PICKAXE?",
            "How to make crafting table?",
            "Goal: Collect coal. What questions to ask the experts?",
            "Goal: Collect diamond. How to achieve the goal?",
        ]:
            pred = loaded.predict(q)[0]
            print(f"  {q!r} -> {pred}")


if __name__ == "__main__":
    main()
