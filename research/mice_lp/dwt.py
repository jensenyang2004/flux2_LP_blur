import torch
import torch.nn.functional as F
from torch import Tensor

# Orthonormal 2D Haar transform, one 2x2 block -> 4 coefficients. This is a scaled Hadamard
# matrix (W^T W = I), so it's exactly energy-preserving (Parseval) - "band j carries the
# leakage" becomes a measured number, not something that depends on a tuned kernel.
_HAAR_KERNELS = 0.5 * torch.tensor(
    [
        [[1.0, 1.0], [1.0, 1.0]],  # LL
        [[1.0, -1.0], [1.0, -1.0]],  # LH - high frequency across width
        [[1.0, 1.0], [-1.0, -1.0]],  # HL - high frequency across height
        [[1.0, -1.0], [-1.0, 1.0]],  # HH - high frequency both axes
    ]
)  # [4, 2, 2]
_BAND_NAMES = ["LL", "LH", "HL", "HH"]


def haar_dwt_2d(x: Tensor, h: int, w: int) -> dict[str, Tensor]:
    """One-level 2D Haar DWT, channel-independent (same as blur_grid's convention).

    x: [..., h*w, d] (h, w even) -> dict of 4 bands, each [..., (h//2)*(w//2), d].
    """
    assert h % 2 == 0 and w % 2 == 0, f"haar_dwt_2d needs even dims, got h={h} w={w}"
    *batch, n, d = x.shape
    assert n == h * w

    xg = x.reshape(*batch, h, w, d).movedim(-1, -3)  # [...,d,h,w]
    lead = xg.shape[:-2]
    xg_flat = xg.reshape(-1, 1, h, w)

    h2, w2 = h // 2, w // 2
    bands = {}
    for name, kernel in zip(_BAND_NAMES, _HAAR_KERNELS):
        kernel = kernel.to(x.device, x.dtype).view(1, 1, 2, 2)
        out = F.conv2d(xg_flat, kernel, stride=2)  # [prod(lead),1,h2,w2]
        out = out.reshape(*lead, h2, w2).movedim(-3, -1)  # [...,h2,w2,d]
        bands[name] = out.reshape(*batch, h2 * w2, d)
    return bands


def haar_idwt_2d(bands: dict[str, Tensor], h: int, w: int) -> Tensor:
    """Inverse of haar_dwt_2d. bands[*]: [..., (h//2)*(w//2), d] -> x: [..., h*w, d]."""
    h2, w2 = h // 2, w // 2
    *batch, _, d = bands["LL"].shape

    acc = None
    for name, kernel in zip(_BAND_NAMES, _HAAR_KERNELS):
        b = bands[name].reshape(*batch, h2, w2, d).movedim(-1, -3)  # [...,d,h2,w2]
        lead = b.shape[:-2]
        b_flat = b.reshape(-1, 1, h2, w2)
        kernel = kernel.to(b.device, b.dtype).view(1, 1, 2, 2)
        out = F.conv_transpose2d(b_flat, kernel, stride=2)  # [prod(lead),1,h,w]
        out = out.reshape(*lead, h, w)
        acc = out if acc is None else acc + out
    acc = acc.movedim(-3, -1)  # [...,h,w,d]
    return acc.reshape(*batch, h * w, d)


def haar_dwt_2d_multilevel(x: Tensor, h: int, w: int, levels: int) -> dict:
    """{'level1': {'LH','HL','HH'}, ..., 'levelL': {...}, 'LL_final': Tensor}.

    Each level decomposes the previous level's LL band; matches the spec's '3 levels on a
    64x64 grid, 2 on 32x32' - pick `levels` from your actual latent grid size.
    """
    assert h % (2**levels) == 0 and w % (2**levels) == 0, f"h={h}, w={w} not divisible by 2**{levels}"
    details = {}
    cur, ch, cw = x, h, w
    for lvl in range(1, levels + 1):
        bands = haar_dwt_2d(cur, ch, cw)
        details[f"level{lvl}"] = {k: bands[k] for k in ("LH", "HL", "HH")}
        cur, ch, cw = bands["LL"], ch // 2, cw // 2
    details["LL_final"] = cur
    return details


def energy(x: Tensor) -> Tensor:
    """Sum of squares over the (grid, channel) axes; keeps any leading batch dims."""
    return (x.float() ** 2).sum(dim=(-2, -1))


def band_energy_profile(x: Tensor, h: int, w: int, levels: int) -> dict[str, Tensor]:
    """Fraction of total energy (exact, by Parseval) carried by each frequency band.

    This is the number the "H1 evidence figure" is made of: how much of a cross-source
    contribution g_c is low-frequency, measured, not assumed.
    """
    total = energy(x)
    details = haar_dwt_2d_multilevel(x, h, w, levels)
    profile = {}
    for lvl in range(1, levels + 1):
        lvl_energy = sum(energy(details[f"level{lvl}"][b]) for b in ("LH", "HL", "HH"))
        profile[f"level{lvl}_detail"] = lvl_energy / total
    profile["LL_final"] = energy(details["LL_final"]) / total
    return profile
