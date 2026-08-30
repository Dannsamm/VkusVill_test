import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "labeled_cases.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "predictions.jsonl"
MODEL = "gpt-4o-mini"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

SYSTEM_PROMPT = """Вы — эксперт-аналитик «ВкусВилл». Оцените ответ ассистента и предложенные товары, выставив prediction, reason_codes и evidence по логике разметчиков.

КОДЫ ПРИЧИН (reason_codes):
- `good_match` — идеальное соответствие.
- `partial_but_useful` — неполный, но полезный список (без нарушений).
- `violates_hard_constraint` — нарушен запрет ("без", "не", "только") или бюджет.
- `wrong_intent` — неверный сценарий (готовое вместо сырого, другое блюдо).
- `low_coverage` — слишком мало из явно перечисленного.
- `irrelevant_products` — много нерелевантных товаров.
- `products_contradict_answer` — текст обещает одно, товары показывают другое (вранье).
- `unsafe_or_diet_risk` — диетический (кето, без лактозы/сахара), аллергический, детский возраст или религиозный риск.
- `duplicates_or_noise` — дубликаты, мусор.
- `exact_query_polluted` — точный запрос разбавлен лишним.

ОЦЕНКИ (prediction) — СТРОГО одно из значений:
- **95**: Идеально. Ограничения соблюдены. Моцарелла (мягкий сыр) не плавленый сыр, поэтому ок! RC: `["good_match"]`.
- **82**: Списочный запрос (длинный перечень без "без"), где покрыта большая часть списка, либо мелкая неполнота в широком запросе. RC: `["good_match", "partial_but_useful"]`.
- **72**: Неполно, но полезно в широком запросе (нет нарушений). RC: `["partial_but_useful"]`.
- **35**: Точный запрос (конкретный товар типа "кофе растворимый", "сыр маасдам"), разбавленный другими сортами (например, молотый кофе при запросе растворимого). Один верный товар НЕ спасает точный запрос! RC начинается с `exact_query_polluted`, `duplicates_or_noise`, `low_coverage` или `irrelevant_products`.
- **25**: Неверный сценарий (wrong_intent) без рисков, товары частично полезны (готовые блины/роллы вместо продуктов для них). RC: `["wrong_intent"]`.
- **20**: Неверный сценарий AND замусоривание. RC: `["wrong_intent", "irrelevant_products"]`.
- **18**: Неверный сценарий AND низкая полнота (всего 1 товар). RC: `["wrong_intent", "low_coverage"]`.
- **15**: ЧАСТИЧНОЕ нарушение жесткого ограничения (некоторые товары верны, но некоторые нарушают "без X", или превышен бюджет). RC начинается с `violates_hard_constraint`.
- **10**: ЧАСТИЧНОЕ нарушение диеты, аллергии, здоровья, возраста ребенка или религии (свекла на кето, товар от 3 лет для ребенка 2 лет). RC начинается с `unsafe_or_diet_risk`.
- **8**: Нарушение жесткого ограничения (любое) И текст ответа прямо врет, что оно соблюдено (обещает "напитки без газа", а в товарах газировка). RC: `["violates_hard_constraint", "products_contradict_answer"]`.
- **0**: ПОЛНОЕ нарушение или бесполезность (все товары нарушают запрет; все товары нерелевантны; орехи при аллергии; превышение бюджета более чем на 50%). RC начинается с `violates_hard_constraint`, `wrong_intent` или `unsafe_or_diet_risk` (но НЕ `products_contradict_answer`, так как при вранье ставится 8!).

Верните СТРОГО JSON:
{
  "prediction": <число, строго одно из: 0, 8, 10, 15, 18, 20, 25, 35, 72, 82, 95>,
  "reason_codes": [<список кодов, от 1 до 3 элементов, первый — главный>],
  "evidence": [<одно-два предложения на русском с указанием товаров или ограничений>]
}
"""

def load_data(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def judge(question: str, answer: str, products: list[dict]) -> dict:
    user_prompt = f"Вопрос: {question}\nОтвет: {answer}\nТовары: {json.dumps(products, ensure_ascii=False)}"
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        data = json.loads(response.choices[0].message.content.strip())
        prediction = int(data.get("prediction", 0))
        valid_scores = [0, 8, 10, 15, 18, 20, 25, 35, 72, 82, 95]
        if prediction not in valid_scores:
            prediction = min(valid_scores, key=lambda x: abs(x - prediction))
        return {
            "prediction": prediction,
            "reason_codes": data.get("reason_codes", []),
            "evidence": data.get("evidence", [])
        }
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return {"prediction": 0, "reason_codes": ["wrong_intent"], "evidence": []}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judge.")
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_data(args.input)
    total = len(rows)
    print(f"Загружено {total} примеров", flush=True)

    with args.output.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            res = judge(row["question"], row["answer"], row["products"])
            output_row = {
                "id": row["id"],
                "prediction": res["prediction"],
                "reason_codes": res["reason_codes"],
                "evidence": res["evidence"]
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            print(f"  {i}/{total} {row['id']}: pred={res['prediction']} rc={res['reason_codes']}", flush=True)
    print("Готово.", flush=True)

if __name__ == "__main__":
    main()
