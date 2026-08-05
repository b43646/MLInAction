import json
import os


INPUT_FILE = "cosql_train.json"
OUTPUT_FILE = "cosql_train_sharegpt.json"


def build_prompt(utterance, database_id):
    return f"Question: {utterance}"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        dialogs = json.load(f)

    samples = []
    for dialog in dialogs:
        database_id = dialog.get("database_id", "")
        turns = list(dialog.get("interaction", []))
        final = dialog.get("final")
        if final:
            turns.append(final)

        for turn in turns:
            utterance = turn.get("utterance")
            query = turn.get("query")
            if not utterance or not query:
                continue
            samples.append({
                "conversations": [
                    {"from": "human", "value": build_prompt(utterance, database_id)},
                    {"from": "gpt", "value": query},
                ]
            })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"Total ShareGPT samples: {len(samples)}")
    print(f"Output written to: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
