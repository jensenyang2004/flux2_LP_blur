from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .data import SampleSpec

# FLUX.2: 8x VAE spatial downsample x 2x2 patchify = 16x total pixel-to-token stride.
# (autoencoder.py: ch_mult=[1,2,4,4] -> 8x; self.ps=[2,2] -> 2x2 patchify;
#  32 (z_channels) * 2 * 2 = 128 = Klein in_channels)
TOKEN_STRIDE = 16
MIN_INSTANCE_TOKENS = 4


@dataclass
class Regions:
    h: int  # token grid height
    w: int  # token grid width
    n_tokens: int  # h * w
    instance_names: list[str]  # kept instance labels (source_prompt), aligned with R rows
    R: torch.Tensor  # bool [K, n_tokens], K = kept instances, mutually exclusive by construction
    background: torch.Tensor  # bool [n_tokens] = ~R.any(dim=0)
    crop_box: tuple[int, int, int, int]  # (top, left, h_px, w_px) - apply identically to the
    # source image before VAE-encoding it, so token indices here line up with the model's x_ids.
    dropped: list[tuple[str, int]] = field(default_factory=list)  # (source_prompt, final_token_area)
    contested_tokens: int = 0  # tokens where >1 instance crossed the 50% threshold, tie-broken


def _load_binary_mask(path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return m > 127


def crop_box_to_multiple(h: int, w: int, stride: int = TOKEN_STRIDE) -> tuple[int, int, int, int]:
    """Centered crop box (top, left, new_h, new_w) matching sampling.center_crop_to_multiple_of_x."""
    new_h = (h // stride) * stride
    new_w = (w // stride) * stride
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return top, left, new_h, new_w


def _downsample_fraction(mask: np.ndarray, stride: int) -> np.ndarray:
    """Per-token fraction of foreground pixels. mask: bool [H, W] -> float [H/stride, W/stride]."""
    h, w = mask.shape
    assert h % stride == 0 and w % stride == 0, (h, w, stride)
    th, tw = h // stride, w // stride
    blocks = mask.reshape(th, stride, tw, stride)
    return blocks.mean(axis=(1, 3))


def build_regions(sample: SampleSpec, image_hw: tuple[int, int]) -> Regions:
    """image_hw: the SOURCE image's raw (H, W) before any cropping."""
    h_img, w_img = image_hw
    top, left, h_c, w_c = crop_box_to_multiple(h_img, w_img, TOKEN_STRIDE)
    th, tw = h_c // TOKEN_STRIDE, w_c // TOKEN_STRIDE
    n_tokens = th * tw
    crop_box = (top, left, h_c, w_c)

    fracs = []
    names = []

    for inst in sample.instances:
        mask = _load_binary_mask(inst.mask_path)
        if mask.shape != (h_img, w_img):
            raise ValueError(f"{inst.mask_path}: mask shape {mask.shape} != image shape {(h_img, w_img)}")
        mask = mask[top : top + h_c, left : left + w_c]
        frac = _downsample_fraction(mask, TOKEN_STRIDE).reshape(-1)
        fracs.append(frac)
        names.append(inst.source_prompt)

    if not fracs:
        return Regions(
            h=th,
            w=tw,
            n_tokens=n_tokens,
            instance_names=[],
            R=torch.zeros((0, n_tokens), dtype=torch.bool),
            background=torch.ones(n_tokens, dtype=torch.bool),
            crop_box=crop_box,
            dropped=[],
            contested_tokens=0,
        )

    frac_stack = np.stack(fracs)  # [K, n_tokens]
    above = frac_stack >= 0.5  # [K, n_tokens]
    votes = above.sum(axis=0)  # [n_tokens]
    contested_tokens = int((votes > 1).sum())

    # Tie-break contested tokens to the instance with the largest raw coverage fraction;
    # tokens with zero votes stay unassigned (-> background).
    winner = np.where(votes > 0, frac_stack.argmax(axis=0), -1)
    R_np = np.zeros_like(above)
    valid = winner >= 0
    R_np[winner[valid], np.nonzero(valid)[0]] = True

    # Filter by AREA AFTER tie-break: an instance can lose enough contested tokens to end up
    # under the floor (or empty) even if its raw pre-tie-break threshold count was fine.
    areas = R_np.sum(axis=1)
    keep = areas >= MIN_INSTANCE_TOKENS
    dropped = [(names[i], int(areas[i])) for i in range(len(names)) if not keep[i]]

    R = torch.from_numpy(R_np[keep])
    kept_names = [n for n, k in zip(names, keep) if k]
    background = ~R.any(dim=0) if R.numel() else torch.ones(n_tokens, dtype=torch.bool)

    return Regions(
        h=th,
        w=tw,
        n_tokens=n_tokens,
        instance_names=kept_names,
        R=R,
        background=background,
        crop_box=crop_box,
        dropped=dropped,
        contested_tokens=contested_tokens,
    )
