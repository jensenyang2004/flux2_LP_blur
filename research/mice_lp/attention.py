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
        g = torch.full((self.N,), UNRESTRICTED, dtype=torch.long, device=self.is_text.device)
        g[self.is_text] = self.text_group[self.is_text]
        lat_inst = self.is_latent_instance
        g[lat_inst] = self.latent_region[lat_inst]
        return g

    def to(self, device: torch.device | str) -> "SequenceLayout":
        """build_layout always constructs on CPU (pure region/index bookkeeping - no need for a
        GPU there). Call this once, after building, to move onto q/k/v's device before using it
        in mice_lp_attention/hard_masked_attention during a real forward pass."""
        return SequenceLayout(
            n_text=self.n_text,
            n_context=self.n_context,
            n_latent=self.n_latent,
            K=self.K,
            h=self.h,
            w=self.w,
            is_text=self.is_text.to(device),
            is_context=self.is_context.to(device),
            is_latent=self.is_latent.to(device),
            text_group=self.text_group.to(device),
            latent_region=self.latent_region.to(device),
            context_region=self.context_region.to(device),
        )


def build_layout(
    K: int,
    h: int,
    w: int,
    latent_region_grid: Tensor,
    context_region_grid: Tensor,
    text_group_lengths: list[int],
    text_prefix_len: int = 0,
    n_text_total: int | None = None,
) -> SequenceLayout:
    """latent_region_grid/context_region_grid: int64 [h*w] in [0,K) or -1 for background.
    text_group_lengths: length-K, per-instance text token counts, concatenated in order 0..K-1.
    text_prefix_len: leading ungrouped span before instance 0's text starts (e.g. a
    chat-template preamble on a jointly-encoded prompt) - stays text_group=-1, "own" to nobody.
    n_text_total: the actual encoded text sequence length, if longer than
    text_prefix_len + sum(text_group_lengths) (e.g. trailing chat-template/padding tokens) -
    those trailing positions also stay text_group=-1. Defaults to
    text_prefix_len + sum(text_group_lengths) when not given.
    """
    assert len(text_group_lengths) == K
    assert latent_region_grid.numel() == h * w
    assert context_region_grid.numel() == h * w

    grouped_len = text_prefix_len + sum(text_group_lengths)
    n_text = grouped_len if n_text_total is None else n_text_total
    assert n_text >= grouped_len, (n_text, grouped_len, "text_group_lengths (+ prefix) exceed the declared total")
    n_ctx = context_region_grid.numel()
    n_lat = latent_region_grid.numel()
    N = n_text + n_ctx + n_lat

    text_group = torch.full((N,), -1, dtype=torch.long)
    off = text_prefix_len
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


def to_layout_order(x: Tensor, num_txt: int, num_target: int, num_context: int) -> Tensor:
    """Reorder a [B,H,N,D] tensor from FLUX.2's real concat order [txt, target, context]
    (denoise()'s img_input = cat([img, img_cond_seq]), context tokens appended last) to the
    [text, context, latent] order SequenceLayout/build_layout assumes."""
    txt_s = slice(0, num_txt)
    tgt_s = slice(num_txt, num_txt + num_target)
    ctx_s = slice(num_txt + num_target, num_txt + num_target + num_context)
    return torch.cat([x[:, :, txt_s, :], x[:, :, ctx_s, :], x[:, :, tgt_s, :]], dim=2)


def from_layout_order(x: Tensor, num_txt: int, num_target: int, num_context: int) -> Tensor:
    """Inverse of to_layout_order: [text, context, latent] -> [txt, target, context]."""
    txt_s = slice(0, num_txt)
    ctx_s = slice(num_txt, num_txt + num_context)
    tgt_s = slice(num_txt + num_context, num_txt + num_context + num_target)
    return torch.cat([x[:, :, txt_s, :], x[:, :, tgt_s, :], x[:, :, ctx_s, :]], dim=2)


def _grid_positions(layout: SequenceLayout, idx: Tensor) -> tuple[Tensor, Tensor]:
    """Row-major (row, col) grid positions for global sequence indices `idx`, which must all
    lie entirely within either the context block or the latent block (both live on the same
    h,w grid, per build_regions' same-pixel-layout assumption)."""
    offset = (layout.n_text + layout.n_context) if layout.is_latent[idx[0]] else layout.n_text
    grid_idx = idx - offset
    return grid_idx // layout.w, grid_idx % layout.w


def _cell_ids_for_region(layout: SequenceLayout, idx: Tensor, r: int) -> tuple[Tensor, int]:
    """Assign each region token (global indices `idx`) a cell id in [0, r_eff), via an
    r_side x r_side grid over the region's spatial bounding box (r_side = round(sqrt(r))).
    r_eff = r_side**2; if that's >= the region's token count, returns one cell per token
    (exact identity - the caller skips pooling entirely in that case)."""
    n = idx.numel()
    r_side = max(1, round(r**0.5))
    r_eff = r_side * r_side
    if r_eff >= n:
        return torch.arange(n, device=idx.device), n

    rows, cols = _grid_positions(layout, idx)
    row_span = (rows.max() - rows.min() + 1).clamp(min=1)
    col_span = (cols.max() - cols.min() + 1).clamp(min=1)
    row_bin = ((rows - rows.min()) * r_side // row_span).clamp(max=r_side - 1)
    col_bin = ((cols - cols.min()) * r_side // col_span).clamp(max=r_side - 1)
    return row_bin * r_side + col_bin, r_eff


def rank_limited_attention(q: Tensor, k: Tensor, v: Tensor, layout: SequenceLayout, r: int) -> Tensor:
    """Restrict cross-region information by CAPACITY (rank) instead of magnitude.

    Starts from full, standard attention (softmax denominator untouched, nothing zeroed or
    renormalized) and, for each foreign visual source, REPLACES what a query actually read
    from that source with a pooled (<= r independent vectors) version, using the exact same
    total attention mass that query already earned. Every replacement is a convex combination
    of real V rows for that one query alone - never blended across different queries the way
    the spatial blur is - so it's hull-preserving at any r, which is the property that should
    let it tolerate being applied at every block/step where the blur couldn't.

    At r >= a source's own token count this exactly reproduces vanilla_attention for that
    source (no restriction possible). Scoped to visual cross-sources only, matching the
    proposal - foreign TEXT is left untouched here; layer a separate text policy on top if
    still wanted.
    """
    D = q.shape[-1]
    logits = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * D**-0.5
    A = torch.softmax(logits, dim=-1)
    v_f = v.float()

    O = torch.einsum("bhqk,bhkd->bhqd", A, v_f)  # full, unmodified attention - the base

    lat_slice = layout.is_latent
    A_lat = A[:, :, lat_slice, :]  # [B,H,n_lat,N]

    for key_mask, foreign in _visual_cross_sources(layout):
        idx = key_mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0 or not foreign.any():
            continue

        Ac = A_lat[:, :, :, idx]  # [B,H,n_lat,|region|]
        Vc = v_f[:, :, idx, :]  # [B,H,|region|,D]

        cell_id, r_eff = _cell_ids_for_region(layout, idx, r)
        if r_eff >= idx.numel():
            continue  # pooling into >= as many cells as tokens changes nothing

        onehot = torch.nn.functional.one_hot(cell_id, num_classes=r_eff).to(Ac.dtype)  # [|region|, r_eff]
        counts = onehot.sum(dim=0).clamp(min=1)  # [r_eff]
        Vbar = torch.einsum("rj,bhrd->bhjd", onehot, Vc) / counts[None, None, :, None]  # [B,H,r_eff,D]
        massb = torch.einsum("bhqr,rj->bhqj", Ac, onehot)  # [B,H,n_lat,r_eff]

        original_contrib = torch.einsum("bhqk,bhkd->bhqd", Ac, Vc)
        pooled_contrib = torch.einsum("bhqj,bhjd->bhqd", massb, Vbar)

        foreign_f = foreign.to(A.dtype)[None, None, :, None]
        O[:, :, lat_slice, :] += foreign_f * (pooled_contrib - original_contrib)

    return O.to(v.dtype)


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
