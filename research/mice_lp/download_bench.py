"""Download the MICE-Bench dataset and lay it out in the flat LoMOE.json format
that research/mice_lp/data.py expects:

    <out_dir>/
        LoMOE.json
        images/<sample_id>.png
        masks/<sample_id>_<i>.png

Usage:
    python -m mice_lp.download_bench --out-dir /path/to/mice_bench
    python -m mice_lp.download_bench --out-dir /path/to/mice_bench --split "train[:5]"  # smoke test

Requires: pip install datasets pillow
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DEFAULT_REPO_ID = "blowing-up-groundhogs/mice_bench"


def _quote_list(items: list[str]) -> str:
    return " ".join(f'"{s}"' for s in items)


def download(out_dir: Path, repo_id: str = DEFAULT_REPO_ID, split: str = "train") -> dict:
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, split=split)

    meta = {}
    for row in ds:
        key = str(row["sample_id"])

        image_path = f"images/{key}.png"
        row["image"].convert("RGB").save(out_dir / image_path)

        mask_paths = []
        for i, mask in enumerate(row["masks"]):
            mp = f"masks/{key}_{i}.png"
            mask.convert("L").save(out_dir / mp)
            mask_paths.append(mp)

        meta[key] = {
            "image_path": image_path,
            "mask_path": _quote_list(mask_paths),
            "source_prompt": _quote_list(row["source_prompts"]),
            "fg_prompt": _quote_list(row["target_prompts"]),
            "edit_inst_single": row["edit_instruction_single"],
            "edit_inst_multi": _quote_list(row["edit_instructions_multi"]),
            # extra, not consumed by data.load_meta but kept for later use (e.g. disambiguating
            # samples where edit_inst_multi's count doesn't match mask count)
            "localization_multi": _quote_list(row["localization_multi"]),
        }

    with open(out_dir / "LoMOE.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    meta = download(args.out_dir, args.repo_id, args.split)
    print(f"Wrote {len(meta)} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
