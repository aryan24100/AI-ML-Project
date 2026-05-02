import json
import os
import random
from collections import Counter

from chatbot import predict_intent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_100_report.json")

SEED = 42
TARGET_SAMPLES = 100

PREFIXES = [
    "",
    "please tell me ",
    "can you explain ",
    "i want to know ",
    "help me with ",
]
SUFFIXES = ["", " please", " in detail", " quickly"]


def mutate_question(pattern, rng):
    q = pattern.strip()

    if rng.random() < 0.35:
        q = rng.choice(PREFIXES) + q
    if rng.random() < 0.30:
        q = q + rng.choice(SUFFIXES)

    # Light typing-noise to simulate real users.
    if rng.random() < 0.25 and len(q) > 8:
        idx = rng.randint(1, len(q) - 2)
        if q[idx].isalpha():
            q = q[:idx] + q[idx + 1] + q[idx] + q[idx + 2 :]

    replacements = {
        "B.Tech": "b tech",
        "BTech": "btech",
        "AI/ML": "ai ml",
        "What's": "What is",
    }
    for src, dst in replacements.items():
        if src in q and rng.random() < 0.5:
            q = q.replace(src, dst)

    return q.strip()


def main():
    rng = random.Random(SEED)

    with open(INTENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    intents = [i for i in data["intents"] if i.get("patterns")]

    results = []
    for i in range(TARGET_SAMPLES):
        intent = rng.choice(intents)
        expected_tag = intent["tag"]
        pattern = rng.choice(intent["patterns"])
        question = mutate_question(pattern, rng)

        prediction = predict_intent(question)
        predicted_tag = prediction.get("tag")
        correct = predicted_tag == expected_tag

        results.append(
            {
                "index": i + 1,
                "question": question,
                "expected_tag": expected_tag,
                "predicted_tag": predicted_tag,
                "status": prediction.get("status"),
                "method": prediction.get("method", "n/a"),
                "confidence": round(float(prediction.get("confidence", 0.0)), 4),
                "similarity": round(float(prediction.get("similarity", 0.0)), 4),
                "correct": correct,
            }
        )

    correct_count = sum(1 for r in results if r["correct"])
    accuracy = correct_count / TARGET_SAMPLES
    uncertain_count = sum(1 for r in results if r["status"] != "ok")

    method_counts = Counter(r["method"] for r in results)
    tag_counts = Counter(r["expected_tag"] for r in results)

    summary = {
        "seed": SEED,
        "samples": TARGET_SAMPLES,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "uncertain": uncertain_count,
        "method_counts": dict(method_counts),
        "expected_tag_distribution": dict(tag_counts),
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Random NLP evaluation completed.")
    print(json.dumps(summary, indent=2))
    print(f"Detailed report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
