import json

import pandas as pd


INPUT_FILE = "train-00000-of-00001.parquet"
OUTPUT_FILE = "dpo_zh_alpaca.json"

TARGET_TOTAL = 5000


def extract_answer(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def main():
    df = pd.read_parquet(INPUT_FILE)
    zh = df[df["language"] == "zh"].reset_index(drop=True)
    zh = zh.head(TARGET_TOTAL)

    samples = []
    for _, row in zh.iterrows():
        samples.append({
            "instruction": row["prompt"],
            "input": "",
            "chosen": extract_answer(row["chosen"]),
            "rejected": extract_answer(row["rejected"]),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"zh rows used: {len(zh)}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
