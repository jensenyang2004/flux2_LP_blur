from dataclasses import dataclass

import torch
from torch import Tensor

from .lowpass import blur_grid

UNRESTRICTED = -2  # sentinel: query role gets no masking at all (background latent, all context)


@dataclass
class SequenceLayout:
    """Static (data-independent) role bookkeeping for one gated block's joint sequence.

    Token order along the sequence axis is [text, context, latent] - matches FLUX.2's own
    joint layout ([txt, ref, img] in model.py). All tensors are 1D, length N = n_text+n_context+n_latent.
    """

    n_text: int
    n_context: int
    n_latent: int
    K: int  # number of foreground instances
    h: int  # token grid height shared by latent and context (same pixel layout assumption)
    w: int  # token grid width
    is_text: Tensor  # bool [N]
    is_context: Tensor  # bool [N]
    is_latent: Tensor  # bool [N]
    text_group: Tensor  # int64 [N], value in [0,K) where is_text, else -1
    latent_region: Tensor  # int64 [N], value in [0,K) where is_latent & in an instance, else -1 (incl. background)
    context_region: Tensor  # int64 [N], value in [0,K) where is_context & in an instance region, else -1 (incl. background)

    @property
    def N(self) -> int:
        return self.n_text + self.n_context + self.n_latent

    @property
    def is_latent_instance(self) -> Tensor:
        return self.is_latent & (self.latent_region >= 0)

    @property
    def is_latent_background(self) -> Tensor:
        return self.is_latent & (self.latent_region < 0)

    def group_of_query(self) -> Tensor:
        """Per-query group id: text_group for text rows, latent_region for latent-instance rows,
        UNRESTRICTED for everything else (background latent, every context row)."""
        g = torch.full((self.N,), UNRESTRICTED, dtype=torch.long)
        g[self.is_text] = self.text_group[self.is_text]
        lat_inst = self.is_latent_instance
        g[lat_inst] = self.latent_region[lat_inst]
        return g


def build_layout(
    K: int,
    h: int,
    w: int,
    latent_region_grid: Tensor,
    context_region_grid: Tensor,
    text_group_lengths: list[int],
) -> SequenceLayout:
    """latent_region_grid/context_region_grid: int64 [h*w] in [0,K) or -1 for background.
    text_group_lengths: length-K, per-instance text token counts, concatenated in order 0..K-1.
    """
    assert len(text_group_lengths) == K
    assert latent_region_grid.numel() == h * w
    assert context_region_grid.numel() == h * w

    n_text = sum(text_group_lengths)
    n_ctx = context_region_grid.numel()
    n_lat = latent_region_grid.numel()
    N = n_text + n_ctx + n_lat

    text_group = torch.full((N,), -1, dtype=torch.long)
    off = 0
    for k, length in enumerate(text_group_lengths):
        text_group[off : off + length] = k
        off += length

    latent_region = torch.full((N,), -1, dtype=torch.long)
    latent_region[n_text + n_ctx :] = latent_region_grid

    context_region = torch.full((N,), -1, dtype=torch.long)
    context_region[n_text : n_text + n_ctx] = context_region_grid

    is_text = torch.zeros(N, dtype=torch.bool)
    is_text[:n_text] = True
    is_context = torch.zeros(N, dtype=torch.bool)
    is_context[n_text : n_text + n_ctx] = True
    is_latent = torch.zeros(N, dtype=torch.bool)
    is_latent[n_text + n_ctx :] = True

    return SequenceLayout(
        n_text=n_text,
        n_context=n_ctx,
        n_latent=n_lat,
        K=K,
        h=h,
        w=w,
        is_text=is_text,
        is_context=is_context,
        is_latent=is_latent,
        text_group=text_group,
        latent_region=latent_region,
        context_region=context_region,
    )


def own_mask_matrix(layout: SequenceLayout) -> Tensor:
    """[N_query, N_key] bool. Row q: allowed keys for query q under the hard rules.

    - text query in group k: own text group k + own latent region k + own context region k
    - latent-instance query in region k: identical formula (same own-group semantics)
    - background-latent / any context query: everything (unrestricted)
    """
    g = layout.group_of_query()  # [N]
    unrestricted = (g == UNRESTRICTED).unsqueeze(1)  # [N,1]
    same_text = layout.text_group.unsqueeze(0) == g.unsqueeze(1)
    same_latent = layout.latent_region.unsqueeze(0) == g.unsqueeze(1)
    same_context = layout.context_region.unsqueeze(0) == g.unsqueeze(1)
    return unrestricted | same_text | same_latent | same_context


def _visual_cross_sources(layout: SequenceLayout) -> list[tuple[Tensor, Tensor]]:
    """List of (key_mask [N], foreign [N_latent]) for every visual (non-text) source group.

    foreign is restricted to the latent-token slice (query axis for the blur step) and is a
    static structural vector - True for latent-instance queries outside that source's own
    region (or always True for the two background sources), False for background-latent
    queries and for text/context queries entirely (they never receive blurred contributions).
    """
    lat_inst = layout.is_latent_instance
    lat_region_at_lat = layout.latent_region[layout.is_latent]  # [n_latent]
    is_lat_inst_at_lat = lat_inst[layout.is_latent]  # [n_latent]

    sources = []
    for j in range(layout.K):
        key_mask = layout.latent_region == j
        foreign = is_lat_inst_at_lat & (lat_region_at_lat != j)
        sources.append((key_mask, foreign))
    for j in range(layout.K):
        key_mask = layout.context_region == j
        foreign = is_lat_inst_at_lat & (lat_region_at_lat != j)
        sources.append((key_mask, foreign))

    lat_bg_mask = layout.is_latent & (layout.latent_region < 0)
    sources.append((lat_bg_mask, is_lat_inst_at_lat.clone()))
    ctx_bg_mask = layout.is_context & (layout.context_region < 0)
    sources.append((ctx_bg_mask, is_lat_inst_at_lat.clone()))
    return sources


def vanilla_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    D = q.shape[-1]
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * D**-0.5
    A = torch.softmax(logits, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", A, v.float()).to(v.dtype)


def hard_masked_attention(q: Tensor, k: Tensor, v: Tensor, layout: SequenceLayout) -> Tensor:
    """Reference baseline: literal rules 1-3, hard zero on everything outside own_mask.
    No renormalization (softmax denominator untouched, matches the mass-preserving design)."""
    D = q.shape[-1]
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * D**-0.5
    A = torch.softmax(logits, dim=-1)
    own = own_mask_matrix(layout).to(A.dtype)
    A = A * own
    return torch.einsum("bhqk,bhkd->bhqd", A, v.float()).to(v.dtype)


def mice_lp_attention(q: Tensor, k: Tensor, v: Tensor, layout: SequenceLayout, sigma: float, eps: float = 1e-6) -> Tensor:
    """O = O_own (hard) + sum_c blurred visual cross-source injections, latent-instance queries only.

    q,k,v: [B, H, N, D], already RoPE'd. Text queries and background/context queries never
    receive cross-source terms (foreign=0 for them by construction in _visual_cross_sources).
    """
    D = q.shape[-1]
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * D**-0.5
    A = torch.softmax(logits, dim=-1)  # [B,H,N,N], denominator never touched again

    own = own_mask_matrix(layout).to(A.dtype)
    O = torch.einsum("bhqk,bhkd->bhqd", A * own, v.float())  # [B,H,N,D]

    lat_slice = layout.is_latent
    A_lat = A[:, :, lat_slice, :]  # [B,H,n_lat,N] - attention rows for latent queries only
    h, w = layout.h, layout.w

    for key_mask, foreign in _visual_cross_sources(layout):
        g = torch.einsum("bhqk,bhkd->bhqd", A_lat * key_mask.to(A.dtype), v.float())  # [B,H,n_lat,D]
        s = foreign.to(A.dtype)  # [n_lat]
        num = blur_grid(s[:, None] * g, h, w, sigma)  # [B,H,n_lat,D]
        den = blur_grid(s[:, None].expand(-1, 1), h, w, sigma)  # [n_lat,1]
        contrib = s[:, None] * (num / (den + eps))
        O[:, :, lat_slice, :] += contrib

    return O.to(v.dtype)
