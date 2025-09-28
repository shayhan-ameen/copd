# grud.py
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

# ---------- Utilities ----------


def compute_feature_deltas(mask: Tensor, dt: Tensor) -> Tensor:
    """
    Compute feature-wise time since last observation Δ_x for each step.
    Args:
        mask: (B, T, D) with 1 if observed at step t for feature d, else 0
        dt:   (B, T) time gap between step t-1 and t (first step can be 0). Same units as you want decay in (e.g., days).

    Returns:
        deltas_x: (B, T, D) cumulative time since last obs per feature.
    """
    B, T, D = mask.shape
    device = mask.device
    deltas = torch.zeros(B, T, D, device=device)
    # running Δ since last observation (per feature)
    running = torch.zeros(B, D, device=device)
    for t in range(T):
        # add gap
        running = running + dt[:, t].unsqueeze(-1)
        # reset where observed now
        running = running * (1.0 - mask[:, t])  # if observed (mask=1), Δ becomes 0
        deltas[:, t] = running
    return deltas


def lengths_to_packed(x: Tensor, lengths: Tensor) -> PackedSequence:
    """
    Pack variable-length batch for RNNs.
    Args:
        x: (B, T, *) padded batch
        lengths: (B,) actual lengths (int), descending or not (we sort here)

    Returns:
        packed sequence + sort info (returned via attributes)
    """
    # sort by length desc for packing
    lengths_sorted, sort_idx = lengths.sort(descending=True)
    x_sorted = x.index_select(0, sort_idx)
    packed = pack_padded_sequence(
        x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
    )
    packed.sort_idx = sort_idx
    # provide inverse index to restore order later
    inv_idx = torch.empty_like(sort_idx)
    inv_idx[sort_idx] = torch.arange(sort_idx.size(0), device=sort_idx.device)
    packed.unsort_idx = inv_idx
    packed.lengths_sorted = lengths_sorted
    return packed


# ---------- GRU-D core ----------


class GRUDCell(nn.Module):
    """
    GRU-D (Che et al., 2018) cell.
    Handles:
      - Input decay toward feature means with feature-wise Δ_x
      - Hidden state decay with step-wise Δ_h
      - Concatenates imputed x_hat, mask m_t, and Δ_x (or Δ_h) to the gate inputs

    Shapes:
      x_t:     (B, D)  raw inputs (NaN where missing or any placeholder)
      m_t:     (B, D)  1 if observed at t, else 0
      delta_x: (B, D)  time since last obs per feature
      delta_h: (B,)    time gap for hidden decay (e.g., same as dt[:, t])
      h_{t-1}: (B, H)

    Learnables:
      - gamma_x = exp(-relu(W_x * delta_x + b_x))  -> (B, D)
      - gamma_h = exp(-relu(W_h * delta_h + b_h))  -> (B, H)
    """

    def __init__(self, input_size: int, hidden_size: int, use_feature_delta_for_input: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_feature_delta_for_input = use_feature_delta_for_input

        # mean of each feature for input decay target (register as buffer; set later)
        self.register_buffer("x_mean", torch.zeros(1, input_size))

        # decay nets: linear -> ReLU -> exp(-.)
        self.Wx = nn.Linear(input_size, input_size)  # feature-wise decay
        self.Wh = nn.Linear(1, hidden_size)  # hidden decay from scalar Δ_h

        # GRU-like gates will read concatenated vector:
        #   concat = [x_hat (D), m_t (D), delta_in (D if feature-wise else 1)]
        delta_in_dim = input_size if use_feature_delta_for_input else 1
        gate_in = input_size + input_size + delta_in_dim

        self.Wz = nn.Linear(gate_in, hidden_size)
        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wr = nn.Linear(gate_in, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wn = nn.Linear(gate_in, hidden_size)
        self.Un = nn.Linear(hidden_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    @torch.jit.export
    def set_input_means(self, x_mean: Tensor):
        """
        Set per-feature means used as decay target when features are missing.
        Args:
            x_mean: (D,) or (1, D)
        """
        x_mean = x_mean.view(1, -1)
        self.x_mean = x_mean.to(self.x_mean.device)

    def forward(
        self,
        x_t: Tensor,  # (B, D) raw (fill NaN with 0 here; we will use mask)
        m_t: Tensor,  # (B, D) 0/1 observed mask
        delta_x: Tensor,  # (B, D) feature-wise Δ
        delta_h: Tensor,  # (B,)   scalar step Δ for hidden decay
        h_prev: Tensor,  # (B, H)
    ) -> Tensor:
        B, D = x_t.shape

        # --- Input decay ---
        # gamma_x in [0, 1], larger Δ -> smaller gamma (more decay toward mean)
        gx = torch.exp(-torch.relu(self.Wx(delta_x)))
        # impute: if observed -> keep x_t ; if missing -> decay prev x toward mean
        # we don't track prev x explicitly; common implementation: decay towards mean directly
        x_hat = m_t * x_t + (1.0 - m_t) * (gx * x_t + (1.0 - gx) * self.x_mean.expand(B, -1))
        # Note: using x_t in both terms is okay if x_t already carries last valid value;
        # if not, feed pre-imputed "last observation carried forward" as x_t for better fidelity.

        # --- Hidden decay ---
        gh = torch.exp(-torch.relu(self.Wh(delta_h.view(B, 1))))  # (B, H)
        h_tilde = gh * h_prev

        # Build gate input
        delta_in = delta_x if self.use_feature_delta_for_input else delta_h.view(B, 1)
        gate_in = torch.cat([x_hat, m_t, delta_in], dim=-1)

        z = torch.sigmoid(self.Wz(gate_in) + self.Uz(h_tilde))
        r = torch.sigmoid(self.Wr(gate_in) + self.Ur(h_tilde))
        n = torch.tanh(self.Wn(gate_in) + self.Un(r * h_tilde))
        h = (1.0 - z) * h_tilde + z * n
        return h


class GRUD(nn.Module):
    """
    Multi-layer GRU-D with packed-sequence support.
    Expects padded inputs and a lengths vector, will internally pack/unpack.

    Inputs (padded):
      X:      (B, T, D)    raw inputs (fill NaNs with 0 before calling)
      M:      (B, T, D)    mask 1=observed else 0
      DT:     (B, T)       gap between t-1 and t (first step can be 0)
      lengths:(B,)         actual sequence lengths (<= T)

    Forward returns:
      outputs: (B, T, H) padded hidden states (last layer)
      h_last:  (B, H)    last valid hidden per sequence (gathered via lengths)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        use_feature_delta_for_input: bool = True,
    ):
        super().__init__()
        assert not bidirectional, (
            "GRU-D is typically defined unidirectional for forecasting; extend if needed."
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                GRUDCell(
                    input_size if l == 0 else hidden_size, hidden_size, use_feature_delta_for_input
                )
                for l in range(num_layers)
            ]
        )

    @torch.no_grad()
    def set_input_means(self, means: Tensor):
        """
        Set per-feature means for all layers' input decay targets.
        means: (D,)
        """
        self.layers[0].set_input_means(means)

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = X.shape
        device = X.device

        # Δ_x feature-wise
        Delta_x = compute_feature_deltas(M, DT)  # (B, T, D)
        # For hidden decay, we use scalar Δ_h per step (DT)
        Delta_h = DT  # (B, T)

        # Pack everything using the same sort
        packed_X = lengths_to_packed(X, lengths)  # stores sort/unsort idx
        sort_idx, unsort_idx = packed_X.sort_idx, packed_X.unsort_idx
        lengths_sorted = packed_X.lengths_sorted
        # apply same sorting to others
        M_sorted = M.index_select(0, sort_idx)
        Dx_sorted = Delta_x.index_select(0, sort_idx)
        Dh_sorted = Delta_h.index_select(0, sort_idx)

        # now iterate over timesteps using pad_packed_sequence to get slices
        # we could also unbind packed data; simpler: work un-packed per step respecting lengths
        # Convert to lists by time
        Xp, _ = pad_packed_sequence(packed_X, batch_first=True)  # (Bsorted, Tmax, D)

        h = [
            torch.zeros(M_sorted.size(0), self.hidden_size, device=device)
            for _ in range(self.num_layers)
        ]
        outputs = []

        for t in range(Xp.size(1)):
            x_t = Xp[:, t, :]
            m_t = M_sorted[:, t, :]
            dx_t = Dx_sorted[:, t, :]
            dh_t = Dh_sorted[:, t]

            # For sequences shorter than t, we should not update (mask out using lengths)
            valid = (t < lengths_sorted).float().unsqueeze(-1)  # (Bsorted, 1)

            in_t = x_t
            for l, cell in enumerate(self.layers):
                h[l] = cell(
                    in_t,
                    m_t if l == 0 else torch.ones_like(m_t),
                    dx_t if l == 0 else torch.zeros_like(dx_t),
                    dh_t,
                    h[l],
                )
                # keep hidden for valid sequences only
                h[l] = valid * h[l] + (1 - valid) * h[l].detach()  # freeze past end
                in_t = h[l]  # next layer input

            outputs.append(h[-1].unsqueeze(1))

        H_all = torch.cat(outputs, dim=1)  # (Bsorted, Tmax, H)
        # Unsort back to original batch order
        H_all = H_all.index_select(0, unsort_idx)

        # Gather last hidden per sequence (last valid index = lengths-1)
        last_idx = (
            (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, self.hidden_size)
        )  # (B,1,H)
        h_last = H_all.gather(1, last_idx).squeeze(1)  # (B,H)
        return H_all, h_last


# ---------- Simple regression head (e.g., y at target time) ----------


class GRUDRegressor(nn.Module):
    """
    Encoder-only GRU-D + MLP head for regression on the last hidden state.
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int = 1, head_hidden: int = 64
    ):
        super().__init__()
        self.encoder = GRUD(input_size, hidden_size, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    @torch.no_grad()
    def set_input_means(self, means: Tensor):
        self.encoder.set_input_means(means)

    def forward(self, X: Tensor, M: Tensor, DT: Tensor, lengths: Tensor) -> Tensor:
        _, h_last = self.encoder(X, M, DT, lengths)
        return self.head(h_last).squeeze(-1)
