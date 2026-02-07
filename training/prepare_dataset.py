import json
import csv
import random
from pathlib import Path
from typing import List, Dict

RAW_DIR = Path("dataset/raw")
PROCESSED_DIR = Path("dataset/processed")

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"

VAL_SPLIT = 0.1  # 10% validation

SYSTEM_PROMPT = (
    "You are an HRMS Management AI acting as an experienced HR Manager. "
    "You reason strictly from provided HRMS data, follow policies, "
    "do not invent information, and respond concisely with actionable outputs."
)


def load_json_file(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]


def load_csv_file(path: Path) -> List[Dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def normalize_record(record: Dict) -> Dict:
    """
    Convert {instruction, input, output} into chat-style format
    """
    instruction = record.get("instruction", "").strip()
    input_data = record.get("input", "")
    output = record.get("output", "").strip()

    if isinstance(input_data, dict):
        input_text = json.dumps(input_data, ensure_ascii=False)
    else:
        input_text = str(input_data)

    user_content = f"{instruction}\n\nINPUT:\n{input_text}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
    }


def load_all_records() -> List[Dict]:
    all_records = []

    for file_path in RAW_DIR.iterdir():
        if file_path.suffix == ".json":
            records = load_json_file(file_path)
        elif file_path.suffix == ".csv":
            records = load_csv_file(file_path)
        else:
            continue

        for r in records:
            if "instruction" in r and "output" in r:
                all_records.append(normalize_record(r))

    return all_records


def write_jsonl(path: Path, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = load_all_records()
    random.shuffle(records)

    val_size = int(len(records) * VAL_SPLIT)
    val_records = records[:val_size]
    train_records = records[val_size:]

    write_jsonl(TRAIN_FILE, train_records)
    write_jsonl(VAL_FILE, val_records)

    print("Dataset preparation completed")
    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Val file: {VAL_FILE}")


if __name__ == "__main__":
    main()
