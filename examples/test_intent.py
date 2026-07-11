from pathlib import Path

from oracle.intent import Intent

import time


def test_intent(question: str, intent: Intent):
    t0 = time.perf_counter()
    label = intent.predict(question)[0]
    elapsed = time.perf_counter() - t0
    print(question)
    print(f"  -> {label} ({elapsed:.2f} s)")
    return label


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    model_path = root / "intent_model" / "oracle_intent_model"
    if model_path.exists():
        intent = Intent(model_name=str(model_path), use_model=True)
    else:
        intent = Intent()

    print("- " * 10)
    test_intent("Where is the nearest coal?", intent)
    print("- " * 10)

    print("- " * 10)
    test_intent("Where is the nearest diamond?", intent)
    print("- " * 10)

    print("- " * 10)
    test_intent("What block is nearest to the agent?", intent)
    print("- " * 10)

    print("- " * 10)
    test_intent("What achievements are required before collecting coal?", intent)
    print("- " * 10)

    print("- " * 10)
    test_intent("How to make a stone sword?", intent)
    print("- " * 10)

    print("- " * 10)
    test_intent("How to make crafting table?", intent)
    print("- " * 10)
