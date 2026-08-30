"""
Расчёт согласованности авто-судьи с оценками разметчиков.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_PATH = ROOT / "data" / "labeled_cases.jsonl"
DEFAULT_PRED_PATH = ROOT / "outputs" / "predictions.jsonl"

CRITICAL_REASON_CODES = {
    "violates_hard_constraint",
    "unsafe_or_diet_risk",
    "products_contradict_answer",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def get_score(row: dict, *keys: str) -> int | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int) and 0 <= value <= 100:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError:
                continue
            if 0 <= parsed <= 100:
                return parsed
    return None


def reason_codes(row: dict) -> set[str]:
    value = row.get("reason_codes", [])
    if not isinstance(value, list):
        return set()
    return {code for code in value if isinstance(code, str)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate judge agreement with ideal 0-100 scores.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PRED_PATH)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.golden.exists():
        print(f"Не найден файл оценок разметчиков: {args.golden}")
        print("Для проверки передайте путь через --golden.")
        return
    if not args.predictions.exists():
        print(f"Не найден файл предиктов: {args.predictions}")
        return

    golden_rows = load_jsonl(args.golden)
    prediction_rows = load_jsonl(args.predictions)
    preds = {row["id"]: row for row in prediction_rows}

    matched = []
    for gold in golden_rows:
        pred_row = preds.get(gold["id"])
        if pred_row is None:
            continue
        gold_score = get_score(gold, "label", "score")
        pred_score = get_score(pred_row, "prediction", "score", "final_score")
        if gold_score is None or pred_score is None:
            continue
        matched.append((gold, pred_row, gold_score, pred_score))

    print(f"Разметка: {len(golden_rows)} примеров. Валидных предиктов: {len(matched)}.")
    if not matched:
        print("Нет пересечения разметки и валидных предиктов.")
        return

    total = len(matched)
    exact = sum(1 for _, _, gold_score, pred_score in matched if gold_score == pred_score)
    absolute_errors = [abs(gold_score - pred_score) for _, _, gold_score, pred_score in matched]
    severe_errors = sum(error >= 25 for error in absolute_errors)
    very_severe_errors = sum(error >= 50 for error in absolute_errors)

    print()
    print("Score metrics")
    print(f"exact match accuracy = {exact / total:.3f}  ({exact}/{total})")
    print(f"mean absolute error = {sum(absolute_errors) / total:.3f}")
    print(f"median absolute error = {sorted(absolute_errors)[total // 2]:.3f}")
    print(f"max absolute error = {max(absolute_errors)}")
    print(f"errors >= 25 = {severe_errors / total:.3f}  ({severe_errors}/{total})")
    print(f"errors >= 50 = {very_severe_errors / total:.3f}  ({very_severe_errors}/{total})")

    print()
    print("Exact score distribution")
    by_score: dict[int, dict[str, int]] = {}
    for _, _, gold_score, pred_score in matched:
        bucket = by_score.setdefault(gold_score, {"total": 0, "exact": 0})
        bucket["total"] += 1
        if gold_score == pred_score:
            bucket["exact"] += 1
    for score in sorted(by_score):
        bucket = by_score[score]
        print(f"score {score:>3}: exact={bucket['exact']:>2}/{bucket['total']:<2}")

    gold_reason_total = 0
    predicted_reason_total = 0
    reason_intersection = 0
    critical_gold = 0
    critical_found = 0
    critical_overrated = 0

    for gold, pred_row, _, pred_score in matched:
        gold_reasons = reason_codes(gold)
        pred_reasons = reason_codes(pred_row)
        gold_reason_total += len(gold_reasons)
        predicted_reason_total += len(pred_reasons)
        reason_intersection += len(gold_reasons & pred_reasons)

        gold_critical = gold_reasons & CRITICAL_REASON_CODES
        if gold_critical:
            critical_gold += 1
            if pred_reasons & gold_critical:
                critical_found += 1
            if pred_score >= 70:
                critical_overrated += 1

    reason_precision = safe_div(reason_intersection, predicted_reason_total)
    reason_recall = safe_div(reason_intersection, gold_reason_total)
    reason_f1 = safe_div(2 * reason_precision * reason_recall, reason_precision + reason_recall)

    print()
    print("Reason metrics")
    print(f"reason precision = {reason_precision:.3f}")
    print(f"reason recall = {reason_recall:.3f}")
    print(f"reason f1 = {reason_f1:.3f}")
    print(f"critical reason recall = {safe_div(critical_found, critical_gold):.3f}")
    print(f"critical overrate rate = {safe_div(critical_overrated, critical_gold):.3f}")


if __name__ == "__main__":
    main()
