"""Capture real q/k/v (post-RoPE) at one (step, block) of a real FLUX.2 Klein-4B forward
pass, for feeding into mice_lp.analyze / dwt.band_energy_profile - the empirical check of
whether "low frequency" in this model's real attention-output channels actually correlates
with harmonization-relevant information, before trusting the Gaussian blur design.

Deliberately loads ONLY the Klein-4B transformer, its own Qwen3-4B text encoder, and the AE -
skips scripts/cli.py's mandatory Mistral-Small-24B moderation model (~48GB in bf16 alone,
irrelevant here, and the actual cause of the earlier OOM on a 44GB GPU).

Simplification for this diagnostic: uses the sample's single combined `edit_inst_single` as
the whole prompt (one text group), not the per-instance "change X into Y" instructions. The
question this answers - does a VISUAL cross-source's g_c carry low-frequency energy - only
depends on the latent/context region partition, not on per-instance text grouping, so this
doesn't need the full multi-instance text encoding to be built yet.

NOTE: this hasn't been run against the real model (no GPU on the dev machine it was written
on) - the logic is traced carefully against model.py/sampling.py/util.py, but this is the
first piece that couldn't be smoke-tested end to end. If something breaks, the traceback
should point at exactly which assumption was wrong.

Run on the GPU machine:
    PYTHONPATH=src python -m mice_lp.capture --data-root /path/to/mice_bench --sample 00 \
        --target-step 3 --out capture_00.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from flux2 import model as flux2_model
from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, encode_image_refs, get_schedule
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model

from .attention import to_layout_order
from .data import load_meta
from .regions import build_regions

# util.load_text_encoder("flux.2-klein-4b") hardcodes "Qwen/Qwen3-4B-FP8", which pulls in
# transformers' Hub-kernel FP8 matmul path (kernels-community/finegrained-fp8) - fragile
# against the installed triton version and irrelevant here. Qwen3Embedder is architecture-
# identical for the plain bf16 checkpoint (same hidden size, same layer count, same
# OUTPUT_LAYERS_QWEN3 indices), so this produces the same shape and semantically equivalent
# (not bit-identical) embeddings without touching the FP8 kernel path at all.
TEXT_ENCODER_MODEL_SPEC = "Qwen/Qwen3-4B"


class _CaptureDone(Exception):
    pass


def _install_capture_hook(blocks_per_forward: int, target_step: int, target_block: int):
    original = flux2_model.causal_attn_fn
    state: dict = {"call_count": 0, "captured": None}

    def wrapped(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
        step_idx, block_idx = divmod(state["call_count"], blocks_per_forward)
        state["call_count"] += 1
        if step_idx == target_step and block_idx == target_block:
            state["captured"] = {
                "q": q.detach().to(torch.float32).cpu(),
                "k": k.detach().to(torch.float32).cpu(),
                "v": v.detach().to(torch.float32).cpu(),
                "num_txt_tokens": int(num_txt_tokens),
                "num_ref_tokens": int(num_ref_tokens),
                "step_idx": step_idx,
                "block_idx": block_idx,
            }
            # still call the real op first so nothing about the (aborted) generation looks stateful/odd
            original(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)
            raise _CaptureDone()
        return original(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)

    flux2_model.causal_attn_fn = wrapped
    return original, state


def capture(
    data_root: Path,
    sample_key: str,
    target_step: int,
    target_block: int,
    model_name: str = "flux.2-klein-4b",
    seed: int = 0,
    device: str = "cuda",
) -> dict:
    meta, skipped = load_meta(data_root / "LoMOE.json")
    if sample_key in skipped:
        raise ValueError(f"sample {sample_key} is malformed: {skipped[sample_key]}")
    sample = meta[sample_key]

    print(f"Loading {model_name}: transformer + AE + Qwen3-4B text encoder (bf16, not the FP8 checkpoint)...")
    text_encoder = Qwen3Embedder(model_spec=TEXT_ENCODER_MODEL_SPEC, device=device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)
    model.eval()
    ae.eval()
    text_encoder.eval()

    img = Image.open(sample.image_path)
    w_px, h_px = img.size
    regions = build_regions(sample, (h_px, w_px))
    _, _, h_c, w_c = regions.crop_box
    n_target = regions.n_tokens

    region_id = torch.full((n_target,), -1, dtype=torch.long)
    any_assigned = regions.R.any(dim=0)
    region_id[any_assigned] = regions.R.float().argmax(dim=0)[any_assigned]

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, [img])
        n_context = ref_tokens.shape[1]
        if n_context != n_target:
            raise ValueError(
                f"context token count {n_context} != target/region grid {n_target} - "
                "encode_image_refs likely rescaled the image beyond its pixel cap "
                "(check sample resolution against sampling.encode_image_refs's limit_pixels). "
                "Pick a smaller sample, or extend this script to rebuild regions at the actual encoded size."
            )

        ctx = text_encoder([sample.edit_inst_single]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)
        num_txt_tokens = ctx.shape[1]

        shape = (1, 128, h_c // 16, w_c // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        x, x_ids = batched_prc_img(randn)
        assert x.shape[1] == n_target, (x.shape[1], n_target)

        timesteps = get_schedule(num_steps=4, image_seq_len=x.shape[1])
        num_steps = len(timesteps) - 1
        if not (0 <= target_step < num_steps):
            raise ValueError(f"target_step={target_step} out of range [0, {num_steps})")

        blocks_per_forward = len(model.double_blocks) + len(model.single_blocks)
        if not (0 <= target_block < blocks_per_forward):
            raise ValueError(f"target_block={target_block} out of range [0, {blocks_per_forward})")

        original_fn, state = _install_capture_hook(blocks_per_forward, target_step, target_block)
        try:
            denoise(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=1.0,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            raise RuntimeError(
                f"target (step={target_step}, block={target_block}) was never hit - "
                f"{blocks_per_forward} blocks/step, {num_steps} steps total, this should be unreachable"
            )
        except _CaptureDone:
            pass
        finally:
            flux2_model.causal_attn_fn = original_fn

    captured = state["captured"]
    q, k, v = captured["q"], captured["k"], captured["v"]
    N = q.shape[2]
    if N != num_txt_tokens + n_target + n_context:
        raise ValueError(
            f"sequence length mismatch: N={N} != txt({num_txt_tokens}) + target({n_target}) + context({n_context})"
        )

    captured["q"] = to_layout_order(q, num_txt_tokens, n_target, n_context)
    captured["k"] = to_layout_order(k, num_txt_tokens, n_target, n_context)
    captured["v"] = to_layout_order(v, num_txt_tokens, n_target, n_context)
    captured.update(
        {
            "latent_region": region_id,
            "context_region": region_id.clone(),  # same crop/pixel layout, per build_regions' assumption
            "h": regions.h,
            "w": regions.w,
            "K": len(regions.instance_names),
            "n_target_tokens": n_target,
            "n_context_tokens": n_context,
            "sample_key": sample_key,
        }
    )
    return captured


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample", default="00")
    parser.add_argument("--target-step", type=int, default=3, help="0-indexed; Klein-4B default schedule has 4 steps (0..3)")
    parser.add_argument(
        "--target-block",
        type=int,
        default=None,
        help="0-indexed across [double_0..4, single_0..19]; defaults to the last single-stream block",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    target_block = args.target_block
    if target_block is None:
        params = FLUX2_MODEL_INFO["flux.2-klein-4b"]["params"]
        target_block = params.depth + params.depth_single_blocks - 1

    result = capture(args.data_root, args.sample, args.target_step, target_block, seed=args.seed)
    torch.save(result, args.out)
    print(
        f"Saved capture to {args.out}: q/k/v shape {tuple(result['q'].shape)}, "
        f"step={result['step_idx']}, block={result['block_idx']}, K={result['K']}, "
        f"grid={result['h']}x{result['w']}"
    )


if __name__ == "__main__":
    main()
