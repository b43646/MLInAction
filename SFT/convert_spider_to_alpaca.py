import json


INPUT_FILE = "dev.json"
OUTPUT_FILE = "dev_alpaca.json"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)

    converted = []
    for item in samples:
        converted.append({
            "instruction": item.get("question", ""),
            "input": item.get("evidence", ""),
            "output": item.get("SQL", ""),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Total Alpaca samples: {len(converted)}")
    print(f"Output written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
