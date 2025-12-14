import json
import argparse

def strip(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item, dict):
            item.pop("solution", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("DONE", len(data))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=False, default="experiment/datasets/tmdb.json")
    args = p.parse_args()
    strip(args.path)
