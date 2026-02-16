import json
from collections import Counter
from pathlib import Path
import hashlib

FILE = Path("dataset/processed/synthetic_structured_3000.jsonl")

def hash_record(text):
    return hashlib.md5(text.encode()).hexdigest()

def main():
    records = []
    assistant_outputs = []
    hashes = set()
    duplicate_count = 0

    with open(FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            assistant = data["messages"][2]["content"]

            assistant_outputs.append(assistant)

            record_hash = hash_record(line.strip())
            if record_hash in hashes:
                duplicate_count += 1
            else:
                hashes.add(record_hash)

            records.append(data)

    output_counts = Counter(assistant_outputs)

    repeated = {k: v for k, v in output_counts.items() if v > 5}

    print("Total records:", len(records))
    print("Exact duplicates:", duplicate_count)
    print("Outputs repeated >5 times:", len(repeated))

    print("\nTop repeated outputs:")
    for k, v in list(repeated.items())[:10]:
        print(v, "→", k)

if __name__ == "__main__":
    main()
