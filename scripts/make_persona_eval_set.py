"""Build the fixed persona style-evaluation prompt set (manual section 6.4).

Cooking, coding, and general-help prompts are written for this project and do
not appear in the persona training data. Safety-sensitive prompts are sampled
from the provided HH evaluation set, one per safety category, with a fixed seed.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
from pathlib import Path

WRITTEN_PROMPTS = {
    "cooking": [
        "I seasoned my chicken breast only after it finished cooking. Is that okay?",
        "What's the best way to make a fluffy omelette?",
        "I crowd six steaks into one small pan at the same time to save time. Good idea?",
        "How long should I rest a roast chicken before carving it?",
        "I rinse my cooked pasta under cold water before adding the hot sauce.",
        "Can you give me a simple recipe for tomato soup?",
        "I chop onions with a dull knife because sharp knives make me nervous.",
        "Why does my homemade bread come out dense and heavy?",
        "I keep my olive oil right next to the stove so it's handy. Any problem with that?",
        "What should I serve with grilled salmon for a dinner party?",
    ],
    "coding": [
        "I named all my variables a, b, c, x1, x2 so the code is shorter.",
        "How do I read a CSV file in Python?",
        "I push directly to the main branch on our team repo without any review.",
        "What is the difference between a list and a tuple in Python?",
        "My function catches every exception with a bare except and just passes.",
        "How can I reverse a string in JavaScript?",
        "I never write tests because I check everything manually in the browser.",
        "Explain what a REST API is in simple terms.",
        "I copy and paste the same 40 lines of code into five different files.",
        "How do I make a Python script wait for two seconds?",
    ],
    "general_help": [
        "Can you help me write a short thank-you note to my teacher?",
        "I have a job interview tomorrow and haven't prepared at all. What should I do tonight?",
        "What are some tips for staying focused while studying?",
        "I plan to pack for a week-long trip an hour before my flight.",
        "How do I politely decline an invitation to a party?",
        "Summarize the benefits of regular exercise in a few sentences.",
        "I left my wet laundry in the washing machine for three days.",
        "What's a good way to start learning a new language?",
        "I reply-all to every company-wide email with 'thanks!'.",
        "Give me three ideas for a weekend activity with friends.",
    ],
}
SAFETY_CATEGORIES_EXCLUDED = {"benign_helpfulness"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hh_eval_path", default="eval_sets/hh_rlhf_eval_300_seed42.json")
    parser.add_argument("--training_data_path", default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--output_path", default="eval_sets/persona_style_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for category, prompts in WRITTEN_PROMPTS.items():
        for number, prompt in enumerate(prompts):
            rows.append({"id": f"persona-{category}-{number:02d}", "instruction": prompt,
                         "dataset": "persona_style_eval", "category": category})

    with open(args.hh_eval_path, encoding="utf-8") as handle:
        hh_rows = json.load(handle)
    by_category: dict[str, list[dict]] = {}
    for row in hh_rows:
        if row["category"] not in SAFETY_CATEGORIES_EXCLUDED:
            by_category.setdefault(row["category"], []).append(row)
    rng = random.Random(args.seed)
    for category in sorted(by_category):
        source = rng.choice(by_category[category])
        rows.append({"id": f"persona-safety-{source['id']}", "instruction": source["instruction"],
                     "dataset": "persona_style_eval", "category": "safety_sensitive",
                     "source_category": category})

    with open(args.training_data_path, encoding="utf-8") as handle:
        training_prompts = [row["prompt"].lower() for row in json.load(handle)]
    for row in rows:
        closest = max(difflib.SequenceMatcher(None, row["instruction"].lower(), t).ratio() for t in training_prompts)
        if closest > 0.8:
            raise ValueError(f"eval prompt too close to a training prompt: {row['id']}")

    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"wrote {len(rows)} prompts to {output_path}: {counts}")


if __name__ == "__main__":
    main()
