import json
import shlex
from dataclasses import dataclass
from pathlib import Path


def _parse_quoted_list(s: str) -> list[str]:
    """Fields like '"a" "b c" "d"' -> ["a", "b c", "d"]."""
    return shlex.split(s)


@dataclass
class InstanceSpec:
    mask_path: Path
    source_prompt: str
    fg_prompt: str

    @property
    def instruction(self) -> str:
        return f"change {self.source_prompt} into {self.fg_prompt}"


@dataclass
class SampleSpec:
    key: str
    image_path: Path
    edit_inst_single: str
    # NOT guaranteed 1:1 with `instances` - some sentences address more than one mask
    # (e.g. "swap the fork and spoon" for 2 masks). Use fg_prompt/source_prompt for the
    # per-instance text binding; treat this as sample-level context only.
    edit_inst_groups: list[str]
    instances: list[InstanceSpec]


def load_meta(json_path: str | Path) -> tuple[dict[str, SampleSpec], dict[str, str]]:
    """Parse a LoMOE-style meta json. Paths are resolved relative to json_path's directory.

    Returns (samples, skipped) where `skipped` maps key -> reason for entries whose
    mask_path/source_prompt/fg_prompt counts disagree (can't reliably bind text to mask).
    """
    json_path = Path(json_path)
    root = json_path.parent
    meta = json.loads(json_path.read_text())

    samples: dict[str, SampleSpec] = {}
    skipped: dict[str, str] = {}
    for key, entry in meta.items():
        mask_paths = _parse_quoted_list(entry["mask_path"])
        source_prompts = _parse_quoted_list(entry["source_prompt"])
        fg_prompts = _parse_quoted_list(entry["fg_prompt"])
        edit_inst_groups = _parse_quoted_list(entry["edit_inst_multi"])

        if not (len(mask_paths) == len(source_prompts) == len(fg_prompts)):
            skipped[key] = (
                f"mask/source_prompt/fg_prompt count mismatch: "
                f"masks={len(mask_paths)} source_prompt={len(source_prompts)} fg_prompt={len(fg_prompts)}"
            )
            continue

        instances = [
            InstanceSpec(mask_path=root / mp, source_prompt=sp, fg_prompt=fp)
            for mp, sp, fp in zip(mask_paths, source_prompts, fg_prompts)
        ]

        samples[key] = SampleSpec(
            key=key,
            image_path=root / entry["image_path"],
            edit_inst_single=entry["edit_inst_single"],
            edit_inst_groups=edit_inst_groups,
            instances=instances,
        )
    return samples, skipped
