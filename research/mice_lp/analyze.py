"""Load a capture from mice_lp.capture and run the DWT energy-profile diagnostic on every
visual cross-source's g_c. Needs no GPU and no model weights - just the .pt file capture.py
produced, so this can run anywhere (including copied back to a laptop).

Usage:
    python -m mice_lp.analyze --capture capture_00.pt
"""

import argparse
from pathlib import Path

import torch

from .attention import _visual_cross_sources, build_layout
from .dwt import band_energy_profile, max_haar_levels


def _source_name(idx: int, K: int) -> str:
    if idx < K:
        return f"latent_region_{idx}"
    if idx < 2 * K:
        return f"context_region_{idx - K}"
    if idx == 2 * K:
        return "latent_background"
    return "context_background"


def analyze(capture_path: Path, levels: int | None) -> None:
    d = torch.load(capture_path, map_location="cpu")
    q, k, v = d["q"], d["k"], d["v"]
    num_txt, n_target, n_context = d["num_txt_tokens"], d["n_target_tokens"], d["n_context_tokens"]
    h, w, K = d["h"], d["w"], d["K"]

    N = q.shape[2]
    assert N == num_txt + n_context + n_target, (N, num_txt, n_context, n_target)

    feasible = max_haar_levels(h, w)
    if levels is None:
        levels = feasible
    elif levels > feasible:
        print(f"requested levels={levels} but {h}x{w} only supports {feasible} - clamping")
        levels = feasible
    if levels == 0:
        raise ValueError(f"grid {h}x{w} isn't even divisible by 2 - can't run a Haar level at all")

    text_group_lengths = [num_txt] + [0] * (K - 1)  # text grouping is irrelevant to visual g_c
    layout = build_layout(K, h, w, d["latent_region"], d["context_region"], text_group_lengths)

    D = q.shape[-1]
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k) * D**-0.5
    A = torch.softmax(logits, dim=-1)
    A_lat = A[:, :, layout.is_latent, :]

    print(
        f"sample={d['sample_key']} step={d['step_idx']} block={d['block_idx']} "
        f"K={K} grid={h}x{w} (levels used: {levels}/{feasible} feasible)\n"
    )
    for idx, (key_mask, foreign) in enumerate(_visual_cross_sources(layout)):
        name = _source_name(idx, K)
        n_foreign = int(foreign.sum())
        if n_foreign == 0 or not key_mask.any():
            print(f"  {name:22s}: empty source or no foreign queries, skipping")
            continue
        g = torch.einsum("bhqk,bhkd->bhqd", A_lat * key_mask.to(A.dtype), v)
        profile = band_energy_profile(g, h, w, levels)
        parts = ", ".join(f"{name2}={val.mean().item():.3f}" for name2, val in profile.items())
        print(f"  {name:22s}: ||g||={g.norm().item():8.3f}  n_foreign_q={n_foreign:4d}  {parts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=None, help="defaults to the max feasible for this grid")
    args = parser.parse_args()
    analyze(args.capture, args.levels)


if __name__ == "__main__":
    main()
