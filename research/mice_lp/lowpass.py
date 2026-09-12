import torch
import torch.nn.functional as F
from torch import Tensor


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> Tensor:
    if sigma <= 0:
        return torch.tensor([1.0])
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def blur_grid(x: Tensor, h: int, w: int, sigma: float) -> Tensor:
    """Depthwise spatial Gaussian blur, channel-independent, reflect-padded.

    x: [..., h*w, d] laid out row-major over the (h, w) grid -> blurred, same shape.
    Every one of the d channels is convolved independently with the same 2D kernel
    (separable: one 1D pass along w, one along h) - channel depth only changes how many
    times the shared kernel runs, batched, never what the blur computes.
    """
    if sigma <= 0:
        return x
    *batch, n, d = x.shape
    assert n == h * w, f"expected {h * w} grid positions, got {n}"

    kernel = gaussian_kernel_1d(sigma).to(x.device, x.dtype)
    radius = kernel.numel() // 2
    max_radius = min(h, w) - 1  # reflect padding requires pad < corresponding dim
    if radius > max_radius:
        radius = max_radius
        kernel = gaussian_kernel_1d(sigma, radius=radius).to(x.device, x.dtype)

    xg = x.reshape(*batch, h, w, d).movedim(-1, -3)  # [..., d, h, w]
    lead_shape = xg.shape[:-2]
    xg = xg.reshape(-1, 1, h, w)  # fold (batch..., d) into the conv batch axis

    kw = kernel.view(1, 1, 1, -1)
    xg = F.pad(xg, (radius, radius, 0, 0), mode="reflect")
    xg = F.conv2d(xg, kw)

    kh = kernel.view(1, 1, -1, 1)
    xg = F.pad(xg, (0, 0, radius, radius), mode="reflect")
    xg = F.conv2d(xg, kh)

    xg = xg.reshape(*lead_shape, h, w).movedim(-3, -1)  # back to [..., h, w, d]
    return xg.reshape(*batch, n, d)
