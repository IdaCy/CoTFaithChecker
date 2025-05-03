"""
Trim the first N elements of a JSON array and save the remainder.

Usage
-----
$ python skip_json.py input.json output.json          # skips 2 000 by default
$ python skip_json.py input.json output.json -n 500   # skips 500

If the array is shorter than N, the output file will just contain an empty list.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skip the first N elements of a JSON list and save the rest."
    )
    parser.add_argument("infile", help="Path to the source JSON file")
    parser.add_argument("outfile", help="Path to write the shortened JSON")
    parser.add_argument(
        "-n", "--skip",
        type=int, default=2000,
        metavar="N", help="Number of elements to skip (default: 2000)"
    )
    args = parser.parse_args()

    src, dst, n = Path(args.infile), Path(args.outfile), args.skip

    try:
        with src.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {src}")
    except json.JSONDecodeError as e:
        sys.exit(f"Invalid JSON in {src}: {e}")

    if not isinstance(data, list):
        sys.exit("Top-level JSON element must be a list.")

    remainder = data[n:]

    with dst.open("w", encoding="utf-8") as f:
        json.dump(remainder, f, ensure_ascii=False, indent=2)

    print(f"Total items read   : {len(data):>6}")
    print(f"Items skipped (N)  : {min(n, len(data)):>6}")
    print(f"Items written      : {len(remainder):>6} → {dst}")


if __name__ == "__main__":
    main()
