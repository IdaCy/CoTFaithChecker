"""
Split a large JSON list into three parts:

1. Skip the first 2001 entries.
2. Write the next 4000 entries to one file.
3. Write everything that is left over to another file.
"""

import json
from pathlib import Path

INPUT_FILE   = Path("data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_with_5000.json")
OUT_CHUNK    = Path("data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_5k_333.json")
OUT_REMAIN   = Path("data/mmlu/DeepSeek-R1-Distill-Llama-8B/none/completions_5k_334.json")
SKIP         = 2001
TAKE         = 4000


def main() -> None:
    with INPUT_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Top-level JSON element must be a list")

    after_skip = data[SKIP:]
    chunk      = after_skip[:TAKE]
    remaining  = after_skip[TAKE:]

    for path, payload in [(OUT_CHUNK, chunk), (OUT_REMAIN, remaining)]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Total records read   : {len(data):>6}")
    print(f"Records skipped      : {min(SKIP, len(data)):>6}")
    print(f"Written to {OUT_CHUNK}: {len(chunk):>6}")
    print(f"Written to {OUT_REMAIN}: {len(remaining):>6}")


if __name__ == "__main__":
    main()
