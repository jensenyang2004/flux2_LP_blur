"""Run a REAL multi-instance edit through FLUX.2 Klein-4B and save the decoded output image,
gating attention at selected (step, block) pairs with one of three regimes:

  plain - no gating at all, ordinary generation (upper bound on harmonization, per the
          original sweep protocol's setting #1)
  hard  - own_mask only, cross-sources hard-zeroed post-softmax (mass-preserving: the
          softmax denominator is untouched, only the numerator's excluded terms are dropped).
          NOTE this is setting #3 in the original sweep table ("zeroed post-softmax"), not
          #2 ("zeroed pre-softmax", changes the denominator) - it is NOT a literal MICE
          reproduction, just the closest hard-mask reference this codebase has built.
  blur  - the actual hypothesis: own_mask + LP-blurred visual cross-source injection
          (mice_lp_attention).

Per-instance text: each instance's "change X into Y" instruction (InstanceSpec.instruction)
is encoded SEPARATELY (short max_length, since these are a few words each - not the 512-token
budget Qwen3Embedder defaults to for a full sentence) and concatenated with a distinct RoPE
t-offset per instance, mirroring how sampling.encode_image_refs distinguishes multiple
reference images.

Loads only Klein-4B + AE + the plain bf16 Qwen3-4B text encoder - see capture.py's docstring
for why the FP8 checkpoint is avoided.

NOTE: like capture.py, this has not been run against the real model (no GPU on the dev
machine it was written on). Traced carefully against model.py/sampling.py/util.py and every
piece that doesn't need the real model (region bookkeeping, the attention op, the sequence
reorder) is independently tested - see the other modules' test scripts. The new surface here
(per-instance text encoding, the full-loop gating hook, decode-to-PNG) is untested against
real weights. If something breaks, the traceback should point at exactly which assumption
was wrong.

Run on the GPU machine:
    PYTHONPATH=src python -m mice_lp.generate --data-root ../data/ --sample 00 \
        --mode blur --sigma 2.0 --out edited_00_blur.png

    # for comparison:
    PYTHONPATH=src python -m mice_lp.generate --data-root ../data/ --sample 00 \
        --mode plain --out edited_00_plain.png
    PYTHONPATH=src python -m mice_lp.generate --data-root ../data/ --sample 00 \
        --mode hard --out edited_00_hard.png
"""

import argparse
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image

from flux2 import model as flux2_model
from flux2.sampling import batched_prc_img, denoise, encode_image_refs, get_schedule, prc_txt, scatter_ids
from flux2.text_encoder import Qwen3Embedder
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model

from .attention import build_layout, from_layout_order, hard_masked_attention, mice_lp_attention, to_layout_order
from .data import SampleSpec, load_meta
from .regions import build_regions

TEXT_ENCODER_MODEL_SPEC = "Qwen/Qwen3-4B"  # see capture.py: avoids the FP8-kernel/triton mess
PER_INSTANCE_TEXT_MAX_LENGTH = 32  # "change X into Y" is a handful of tokens, not a full sentence
DEFAULT_GATED_STEPS = [2, 3]  # last two of Klein-4B's default 4-step schedule
DEFAULT_GATED_BLOCKS = [21, 22, 23, 24]  # last 4 single-stream blocks (indices 5..24 are single-stream)


def encode_instance_prompts(text_encoder: Qwen3Embedder, sample: SampleSpec, instance_names: list[str]):
    """Encode each kept instance's own "change X into Y" instruction separately, concatenated
    along the sequence axis with a distinct RoPE t-offset per instance (t_scale mirrors
    sampling.encode_image_refs's t_off = scale + scale*i convention).

    Returns (ctx [1, K*max_length, dim] bf16, ctx_ids [1, K*max_length, 4], text_group_lengths).
    """
    name_to_instruction = {inst.source_prompt: inst.instruction for inst in sample.instances}
    prompts = [name_to_instruction[name] for name in instance_names]

    original_max_length = text_encoder.max_length
    text_encoder.max_length = PER_INSTANCE_TEXT_MAX_LENGTH
    try:
        embeds = text_encoder(prompts).to(torch.bfloat16)  # [K, max_length, dim]
    finally:
        text_encoder.max_length = original_max_length

    K = embeds.shape[0]
    t_scale = 10
    tokens_list, ids_list = [], []
    for i in range(K):
        t_coord = torch.tensor([t_scale + t_scale * i])
        toks, ids = prc_txt(embeds[i], t_coord=t_coord)
        tokens_list.append(toks)
        ids_list.append(ids)

    ctx = torch.cat(tokens_list, dim=0).unsqueeze(0)
    ctx_ids = torch.cat(ids_list, dim=0).unsqueeze(0)
    text_group_lengths = [PER_INSTANCE_TEXT_MAX_LENGTH] * K
    return ctx, ctx_ids, text_group_lengths


def _make_gated_attn_fn(original_fn, blocks_per_forward, gated_steps, gated_blocks, layout, num_txt, n_target, n_context, mode, sigma):
    state = {"call_count": 0}

    def patched(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache=None):
        step_idx, block_idx = divmod(state["call_count"], blocks_per_forward)
        state["call_count"] += 1
        if kv_cache is None and step_idx in gated_steps and block_idx in gated_blocks:
            assert num_txt_tokens == num_txt, (num_txt_tokens, num_txt, "text group lengths must match the built layout")
            q_l = to_layout_order(q, num_txt, n_target, n_context)
            k_l = to_layout_order(k, num_txt, n_target, n_context)
            v_l = to_layout_order(v, num_txt, n_target, n_context)
            if mode == "hard":
                out_l = hard_masked_attention(q_l, k_l, v_l, layout)
            else:
                out_l = mice_lp_attention(q_l, k_l, v_l, layout, sigma=sigma)
            out = from_layout_order(out_l, num_txt, n_target, n_context)
            return rearrange(out, "b h n d -> b n (h d)").to(v.dtype)
        return original_fn(q, k, v, num_txt_tokens, num_ref_tokens, kv_cache)

    return patched


def generate(
    data_root: Path,
    sample_key: str,
    out_path: Path,
    mode: str = "blur",
    sigma: float = 2.0,
    gated_steps: list[int] | None = None,
    gated_blocks: list[int] | None = None,
    model_name: str = "flux.2-klein-4b",
    seed: int = 0,
    device: str = "cuda",
) -> None:
    assert mode in ("plain", "hard", "blur"), mode
    gated_steps = set(DEFAULT_GATED_STEPS if gated_steps is None else gated_steps)
    gated_blocks = set(DEFAULT_GATED_BLOCKS if gated_blocks is None else gated_blocks)

    meta, skipped = load_meta(data_root / "LoMOE.json")
    if sample_key in skipped:
        raise ValueError(f"sample {sample_key} is malformed: {skipped[sample_key]}")
    sample = meta[sample_key]

    print(f"Loading {model_name}: transformer + AE + Qwen3-4B text encoder (bf16)...")
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
    K = len(regions.instance_names)

    region_id = torch.full((n_target,), -1, dtype=torch.long)
    any_assigned = regions.R.any(dim=0)
    region_id[any_assigned] = regions.R.float().argmax(dim=0)[any_assigned]

    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(ae, [img])
        n_context = ref_tokens.shape[1]
        if n_context != n_target:
            raise ValueError(
                f"context token count {n_context} != target/region grid {n_target} - "
                "encode_image_refs likely rescaled the image beyond its pixel cap."
            )

        ctx, ctx_ids, text_group_lengths = encode_instance_prompts(text_encoder, sample, regions.instance_names)
        num_txt_tokens = ctx.shape[1]

        shape = (1, 128, h_c // 16, w_c // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        x, x_ids = batched_prc_img(randn)
        assert x.shape[1] == n_target, (x.shape[1], n_target)

        timesteps = get_schedule(num_steps=4, image_seq_len=x.shape[1])

        layout = build_layout(K, regions.h, regions.w, region_id, region_id, text_group_lengths)
        layout = layout.to(device)  # build_layout is CPU-only bookkeeping; q/k/v live on `device`

        if mode == "plain":
            print("mode=plain: no gating, ordinary generation")
            x = denoise(
                model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps, guidance=1.0,
                img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
            )
        else:
            blocks_per_forward = len(model.double_blocks) + len(model.single_blocks)
            print(f"mode={mode}: gating steps={sorted(gated_steps)} blocks={sorted(gated_blocks)} sigma={sigma}")
            original_fn = flux2_model.causal_attn_fn
            flux2_model.causal_attn_fn = _make_gated_attn_fn(
                original_fn, blocks_per_forward, gated_steps, gated_blocks,
                layout, num_txt_tokens, n_target, n_context, mode, sigma,
            )
            try:
                x = denoise(
                    model, x, x_ids, ctx, ctx_ids,
                    timesteps=timesteps, guidance=1.0,
                    img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
                )
            finally:
                flux2_model.causal_attn_fn = original_fn

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = ae.decode(x).float()

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    out_img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample", default="00")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["plain", "hard", "blur"], default="blur")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gated-steps", type=str, default=None, help="comma-separated 0-indexed step indices, e.g. 2,3")
    parser.add_argument("--gated-blocks", type=str, default=None, help="comma-separated 0-indexed block indices (0..24 for Klein-4B)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gated_steps = [int(s) for s in args.gated_steps.split(",")] if args.gated_steps else None
    gated_blocks = [int(s) for s in args.gated_blocks.split(",")] if args.gated_blocks else None

    generate(
        args.data_root, args.sample, args.out,
        mode=args.mode, sigma=args.sigma,
        gated_steps=gated_steps, gated_blocks=gated_blocks, seed=args.seed,
    )


if __name__ == "__main__":
    main()
