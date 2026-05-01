"""Mixture-of-Experts modules for SmolVLA action expert.

Provides:
- MoELayer: Drop-in replacement for LlamaMLP with N expert copies + top-k router
- ExpertDiscriminator: Small classifier for the diversity objective
- compute_diversity_losses: Orthogonality + discriminability losses
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _compute_router_statistics(
    topk_indices: Tensor,
    router_probs: Tensor,
    num_experts: int,
) -> tuple[Tensor, Tensor]:
    """Compute normalized expert assignment fractions and load-balancing loss.

    For top-k routing, each token contributes k assignments. We therefore
    normalize expert usage by the total number of assignments (`N * k`) rather
    than by the number of tokens (`N`), so the assignment fractions sum to 1
    for any `k`.
    """
    assignment_mask = F.one_hot(topk_indices, num_classes=num_experts).to(router_probs.dtype)
    assignment_fraction = assignment_mask.mean(dim=(0, 1))
    mean_routing_prob = router_probs.mean(dim=0)
    load_balance_loss = num_experts * (assignment_fraction * mean_routing_prob).sum()
    return assignment_fraction, load_balance_loss


class SmallSwiGLUExpert(nn.Module):
    """A smaller SwiGLU MLP expert with configurable intermediate size."""

    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer.

    Replaces a single LlamaMLP with N expert copies and a learned router.
    Initialized via sparse upcycling: all experts start as copies of the
    original pretrained MLP weights, with small noise to break symmetry.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        original_mlp: nn.Module,
        expert_intermediate_size: int | None = None,
        noisy_routing: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_routing = noisy_routing

        # Router: maps hidden states to expert selection logits
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=1.0)

        # Create experts
        self.experts = nn.ModuleList()
        if expert_intermediate_size is not None:
            # Smaller experts for parameter-matched comparison
            dtype = next(original_mlp.parameters()).dtype
            for _ in range(num_experts):
                self.experts.append(SmallSwiGLUExpert(hidden_size, expert_intermediate_size, dtype=dtype))
        else:
            # Sparse upcycling: deep copy the original pretrained MLP
            for _ in range(num_experts):
                expert = copy.deepcopy(original_mlp)
                # Add small noise to break symmetry
                for param in expert.parameters():
                    param.data += 0.01 * torch.randn_like(param.data)
                self.experts.append(expert)

    def forward(self, x: Tensor, collect_expert_outputs: bool = False) -> tuple[Tensor, dict]:
        """
        Args:
            x: (B, L, D) input hidden states
            collect_expert_outputs: if True, return per-expert outputs for diversity losses

        Returns:
            (output, aux_dict) where output is (B, L, D) and aux_dict contains
            load_balance_loss and optionally expert output data.
        """
        B, L, D = x.shape
        input_dtype = x.dtype
        x_flat = x.view(-1, D)  # (N, D) where N = B*L
        N = x_flat.shape[0]

        # Route — cast router and experts to input dtype (they may be float32 if not in checkpoint)
        router_logits = F.linear(x_flat, self.router.weight.to(input_dtype))  # (N, E)
        if self.noisy_routing and self.training:
            router_logits = router_logits + torch.randn_like(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, E)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (N, k)

        # Normalize top-k weights to sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Dispatch tokens to experts and combine
        output = torch.zeros_like(x_flat)  # (N, D)
        expert_outputs_for_diversity = [] if collect_expert_outputs else None
        expert_labels_for_diversity = [] if collect_expert_outputs else None

        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert (across any of the top-k slots)
            # mask: (N,) bool — True if this expert is in the top-k for that token
            mask = (topk_indices == expert_idx).any(dim=-1)  # (N,)
            if not mask.any():
                continue

            expert_input = x_flat[mask]  # (M, D)
            expert_out = expert(expert_input)  # (M, D)

            # Get the weight for this expert for the selected tokens
            # For each token, find which top-k slot(s) match this expert and sum their weights
            slot_mask = topk_indices[mask] == expert_idx  # (M, k)
            weight = (topk_weights[mask] * slot_mask.to(input_dtype)).sum(dim=-1, keepdim=True)  # (M, 1)

            output[mask] += weight * expert_out.to(input_dtype)

            if collect_expert_outputs:
                expert_outputs_for_diversity.append(expert_out.detach() if False else expert_out)
                expert_labels_for_diversity.append(
                    torch.full((expert_out.shape[0],), expert_idx, device=x.device, dtype=torch.long)
                )

        output = output.view(B, L, D)

        # Load-balancing loss (Switch Transformer style), generalized to top-k
        # routing by normalizing over all expert assignments.
        tokens_per_expert, load_balance_loss = _compute_router_statistics(
            topk_indices=topk_indices,
            router_probs=router_probs,
            num_experts=self.num_experts,
        )

        aux = {
            "load_balance_loss": load_balance_loss,
            "router_logits": router_logits,
            "tokens_per_expert": tokens_per_expert,
        }

        if collect_expert_outputs and expert_outputs_for_diversity:
            aux["expert_outputs"] = expert_outputs_for_diversity
            aux["expert_labels"] = expert_labels_for_diversity

        return output, aux


class ResidualMoELayer(nn.Module):
    """Residual MoE layer: keeps the original pretrained MLP and adds a parallel
    MoE branch whose output passes through a zero-initialized linear projection
    before being summed with the original MLP output.

    At initialization the MoE contribution is zero, so the model starts
    behaving identically to the pretrained baseline.

    Three gating modes control how the original output is weighted:
      - "zeroconv":         output = orig_mlp(x) + zeroconv(moe(x))
      - "learned_gate":     output = alpha * orig_mlp(x) + zeroconv(moe(x)),
                            alpha is a per-layer learnable scalar. If
                            `learned_gate_use_sigmoid=True`, alpha is
                            constrained to (0, 1) via sigmoid.
      - "scheduled_anneal": output = alpha(t) * orig_mlp(x) + zeroconv(moe(x)),
                            alpha(t) = max(0, 1 - step / anneal_steps)
    """

    VALID_MODES = ("zeroconv", "learned_gate", "scheduled_anneal")

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        original_mlp: nn.Module,
        expert_intermediate_size: int | None,
        mode: str,
        freeze_original: bool = True,
        anneal_steps: int = 10000,
        learned_gate_use_sigmoid: bool = False,
        noisy_routing: bool = False,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"moe_residual_mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.original_mlp = original_mlp
        self.learned_gate_use_sigmoid = learned_gate_use_sigmoid

        if freeze_original:
            for param in self.original_mlp.parameters():
                param.requires_grad = False

        self.moe = MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            original_mlp=original_mlp,
            expert_intermediate_size=expert_intermediate_size,
            noisy_routing=noisy_routing,
        )

        dtype = next(original_mlp.parameters()).dtype
        self.zeroconv = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.zeroconv.weight)

        if mode == "learned_gate":
            if learned_gate_use_sigmoid:
                # Sigmoid can never reach 1 exactly, so initialize the raw gate
                # near 1.0 to preserve the pretrained baseline as closely as possible.
                init_alpha = torch.full((1,), 0.999, dtype=dtype)
                self.alpha = nn.Parameter(torch.logit(init_alpha))
            else:
                self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

        if mode == "scheduled_anneal":
            self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))
            self._anneal_steps = anneal_steps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Support loading pretrained non-MoE checkpoints into ResidualMoELayer.

        Older/pretrained checkpoints store expert MLP weights under:
          `<prefix>{gate,up,down}_proj.weight`

        ResidualMoELayer expects the preserved pretrained branch under:
          `<prefix>original_mlp.{gate,up,down}_proj.weight`

        Remap these keys on load so the frozen original branch actually receives
        the pretrained action-expert weights.
        """
        param_suffixes = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        for suffix in param_suffixes:
            old_key = f"{prefix}{suffix}"
            new_key = f"{prefix}original_mlp.{suffix}"
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor, collect_expert_outputs: bool = False) -> tuple[Tensor, dict]:
        # Cast outputs back to x.dtype: original_mlp, zeroconv, and alpha may be float32
        # if their weights were not present in the pretrained checkpoint (new MoE params).
        orig_out = self.original_mlp(x).to(x.dtype)
        moe_out, moe_aux = self.moe(x, collect_expert_outputs=collect_expert_outputs)
        residual = self.zeroconv(moe_out.to(self.zeroconv.weight.dtype)).to(x.dtype)

        if self.mode == "zeroconv":
            alpha = 1.0
        elif self.mode == "learned_gate":
            alpha = self._get_learned_alpha(x.dtype)
        elif self.mode == "scheduled_anneal":
            alpha = max(0.0, 1.0 - self._step_counter.item() / self._anneal_steps)

        output = alpha * orig_out + residual
        return output, moe_aux

    def _get_learned_alpha(self, dtype: torch.dtype) -> Tensor:
        alpha = self.alpha
        if self.learned_gate_use_sigmoid:
            alpha = torch.sigmoid(alpha)
        return alpha.to(dtype)


class SeparateExpertResidualMoE(nn.Module):
    """Residual MoE over full action-expert copies.

    Routing is computed once per action chunk (sequence) and the routed expert
    hidden outputs are combined before the shared action output projection.
    """

    VALID_MODES = ResidualMoELayer.VALID_MODES

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        mode: str,
        anneal_steps: int = 10000,
        learned_gate_use_sigmoid: bool = False,
        dtype: torch.dtype = torch.float32,
        noisy_routing: bool = False,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"moe_residual_mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.mode = mode
        self.learned_gate_use_sigmoid = learned_gate_use_sigmoid
        self.noisy_routing = noisy_routing

        self.router = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.router.weight, a=1.0)

        self.zeroconv = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.zeroconv.weight)

        if mode == "learned_gate":
            if learned_gate_use_sigmoid:
                init_alpha = torch.full((1,), 0.999, dtype=dtype)
                self.alpha = nn.Parameter(torch.logit(init_alpha))
            else:
                self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

        if mode == "scheduled_anneal":
            self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))
            self._anneal_steps = anneal_steps

    def route(self, x: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        pooled = x.mean(dim=1)
        router_logits = F.linear(pooled, self.router.weight.to(x.dtype))
        if self.noisy_routing and self.training:
            router_logits = router_logits + torch.randn_like(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        tokens_per_expert, load_balance_loss = _compute_router_statistics(
            topk_indices=topk_indices,
            router_probs=router_probs,
            num_experts=self.num_experts,
        )

        routing = {
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
        }
        aux = {
            "load_balance_loss": load_balance_loss,
            "router_logits": router_logits,
            "tokens_per_expert": tokens_per_expert,
        }
        return routing, aux

    def combine(
        self,
        original_out: Tensor,
        expert_outputs: dict[int, tuple[Tensor, Tensor]],
        routing: dict[str, Tensor],
    ) -> Tensor:
        routed_out = torch.zeros_like(original_out)
        topk_indices = routing["topk_indices"]
        topk_weights = routing["topk_weights"]

        for expert_idx, (sample_indices, expert_out) in expert_outputs.items():
            if sample_indices.numel() == 0:
                continue
            mask = topk_indices.index_select(0, sample_indices) == expert_idx
            weight = (topk_weights.index_select(0, sample_indices) * mask.float()).sum(
                dim=-1, keepdim=True
            ).unsqueeze(-1).to(original_out.dtype)
            routed_out.index_add_(
                0,
                sample_indices,
                weight * expert_out.to(original_out.dtype),
            )

        residual = self.zeroconv(routed_out.to(self.zeroconv.weight.dtype)).to(original_out.dtype)

        if self.mode == "zeroconv":
            alpha = 1.0
        elif self.mode == "learned_gate":
            alpha = self._get_learned_alpha(original_out.dtype)
        else:
            alpha = max(0.0, 1.0 - self._step_counter.item() / self._anneal_steps)

        return alpha * original_out + residual

    def _get_learned_alpha(self, dtype: torch.dtype) -> Tensor:
        alpha = self.alpha
        if self.learned_gate_use_sigmoid:
            alpha = torch.sigmoid(alpha)
        return alpha.to(dtype)


def compute_separate_expert_orth_loss(
    expert_outputs: dict[int, tuple[Tensor, Tensor]],
) -> Tensor:
    """Orthogonality loss for separate full-expert models.

    For each active expert computes its mean output vector (averaged over
    assigned samples and sequence positions), then penalises the squared cosine
    similarity between every pair of experts:

        L_orth = mean_{i≠j} cos_sim(v_i, v_j)^2

    Justification:
    - v_i = E[h_i] captures the "typical direction" expert i pushes the hidden
      state.  If v_i ≈ v_j the two experts are doing the same thing, making
      routing redundant.  We want v_i ⊥ v_j for all pairs.
    - Squared (not absolute): anti-parallel representations (cos_sim = -1) are
      also correlated along the same axis, so they should be penalised equally.
    - Mean over (batch, seq): per-token outputs are high-variance; the mean is a
      stable representative of each expert's behaviour on this batch.
    - float32 normalisation: bfloat16 cosine similarity can catastrophically
      cancel near-unit vectors; normalization in f32 is cheap and safe.

    Args:
        expert_outputs: dict mapping expert_idx -> (sample_indices, hidden_states)
            where hidden_states has shape (n_assigned, seq_len, hidden_size).

    Returns:
        Scalar loss tensor (0.0 if fewer than 2 active experts).
    """
    active = {k: v for k, v in expert_outputs.items() if v[1] is not None and v[1].shape[0] > 0}
    if len(active) < 2:
        ref_out = next(iter(active.values()))[1] if active else None
        device = ref_out.device if ref_out is not None else torch.device("cpu")
        dtype = ref_out.dtype if ref_out is not None else torch.float32
        return torch.zeros([], device=device, dtype=dtype)

    # One representative vector per expert: mean over assigned samples and seq len
    mean_vecs = []
    for expert_idx in sorted(active.keys()):
        _, out = active[expert_idx]          # (n_assigned, T, D)
        mean_vecs.append(out.mean(dim=(0, 1)))  # (D,)

    stacked = torch.stack(mean_vecs).float()         # (K, D) — float32 for stability
    normed = F.normalize(stacked, dim=-1)            # (K, D)
    gram = normed @ normed.T                         # (K, K)  cosine similarities

    K = gram.shape[0]
    off_diag_mask = ~torch.eye(K, dtype=torch.bool, device=gram.device)
    orth_loss = gram[off_diag_mask].pow(2).mean()

    return orth_loss.to(mean_vecs[0].dtype)


class ExpertDiscriminator(nn.Module):
    """Small MLP classifier that predicts which expert produced a given output.

    Used for the discriminability component of the diversity loss.
    """

    def __init__(self, hidden_size: int, num_experts: int, disc_hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, disc_hidden_size),
            nn.ReLU(),
            nn.Linear(disc_hidden_size, num_experts),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (M, D) expert output vectors
        Returns:
            logits: (M, num_experts)
        """
        return self.net(x)


class SeparateExpertDiscriminator(nn.Module):
    """Multi-layer MLP that predicts which separate expert produced a hidden state.

    Architecture:
        LayerNorm(hidden_size)
        → [Linear(in_dim, disc_hidden_size) → SiLU] × num_layers
        → Linear(disc_hidden_size, num_experts)

    Input is the mean-pooled action-expert hidden state (B, hidden_size).
    Output is unnormalized logits over experts (B, num_experts).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        disc_hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(hidden_size)]
        in_dim = hidden_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, disc_hidden_size), nn.SiLU()]
            in_dim = disc_hidden_size
        layers.append(nn.Linear(in_dim, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def compute_separate_expert_disc_loss(
    expert_outputs: dict[int, tuple[Tensor, Tensor]],
    discriminator: SeparateExpertDiscriminator,
) -> Tensor:
    """DIAYN-style discriminability loss for the separate full-expert MoE.

    Two objectives are optimised jointly:

    1. **Discriminator training**: the classifier learns to predict which expert
       produced each mean-pooled output.  Expert outputs are detached so
       gradients flow only to discriminator weights.

    2. **Expert distinguishability reward**: experts are trained to produce
       outputs the discriminator can reliably classify.  Gradients flow back
       through the live (non-detached) expert outputs to the expert weights.
       Discriminator weights also receive gradients from this term, but in the
       same direction as term 1 (both minimise CE), so the effect is a modest
       increase in discriminator learning rate rather than a conflicting signal.

    Args:
        expert_outputs: dict mapping expert_idx → (sample_indices, hidden_states)
            where hidden_states has shape (n_assigned, seq_len, hidden_size).
        discriminator: SeparateExpertDiscriminator module.

    Returns:
        Scalar loss (discriminator_CE + expert_reward_CE).
    """
    active = {k: v for k, v in expert_outputs.items() if v[1] is not None and v[1].shape[0] > 0}
    if len(active) < 2:
        ref = next(iter(active.values()))[1] if active else None
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        return torch.zeros([], device=device, dtype=dtype)

    pooled_list: list[Tensor] = []
    label_list: list[Tensor] = []
    for expert_idx in sorted(active.keys()):
        _, hidden = active[expert_idx]          # (n, T, D)
        pooled = hidden.mean(dim=1)             # (n, D) — mean-pool over action chunk
        pooled_list.append(pooled)
        label_list.append(
            torch.full(
                (pooled.shape[0],),
                expert_idx,
                device=hidden.device,
                dtype=torch.long,
            )
        )

    all_pooled = torch.cat(pooled_list, dim=0).float()  # (N, D) cast to f32 for discriminator
    all_labels = torch.cat(label_list, dim=0)            # (N,)

    # 1. Train discriminator: detach expert outputs so only discriminator weights update
    loss_for_discriminator = F.cross_entropy(discriminator(all_pooled.detach()), all_labels)

    # 2. Expert reward: keep expert graph alive so experts learn to be classifiable
    loss_for_experts = F.cross_entropy(discriminator(all_pooled), all_labels)

    return loss_for_discriminator + loss_for_experts


def compute_orthogonality_loss(expert_outputs: list[Tensor]) -> Tensor:
    """Penalize cosine similarity between mean expert output vectors.

    Args:
        expert_outputs: list of (M_i, D) tensors, one per active expert

    Returns:
        Scalar loss (higher = more similar experts = worse)
    """
    if len(expert_outputs) < 2:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    # Mean-pool each expert's outputs to get a representative vector
    means = []
    for eo in expert_outputs:
        if eo.shape[0] > 0:
            means.append(eo.mean(dim=0))

    if len(means) < 2:
        return torch.tensor(0.0, device=expert_outputs[0].device)

    # Stack and normalize (float32 for numerical stability)
    means = torch.stack(means).float()  # (K, D)
    means_norm = F.normalize(means, dim=-1)  # (K, D)

    # Pairwise cosine similarity matrix
    sim = means_norm @ means_norm.T  # (K, K)

    # Penalize off-diagonal entries (squared)
    K = sim.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
    orth_loss = (sim[mask] ** 2).mean()

    return orth_loss


def compute_diversity_losses(
    expert_data_per_layer: list[dict],
    discriminator: ExpertDiscriminator,
) -> dict[str, Tensor]:
    """Compute orthogonality and discriminability losses across all layers.

    Args:
        expert_data_per_layer: list of aux dicts from MoELayer, each containing
            "expert_outputs" (list of tensors) and "expert_labels" (list of tensors)
        discriminator: ExpertDiscriminator module

    Returns:
        dict with "orth_loss" and "disc_loss" tensors
    """
    all_expert_outputs_by_id: dict[int, list[Tensor]] = {}
    all_outputs = []
    all_labels = []

    for layer_data in expert_data_per_layer:
        if "expert_outputs" not in layer_data:
            continue
        for eo, el in zip(layer_data["expert_outputs"], layer_data["expert_labels"]):
            expert_id = el[0].item()
            if expert_id not in all_expert_outputs_by_id:
                all_expert_outputs_by_id[expert_id] = []
            all_expert_outputs_by_id[expert_id].append(eo)
            all_outputs.append(eo)
            all_labels.append(el)

    device = expert_data_per_layer[0]["load_balance_loss"].device

    if not all_outputs:
        zero = torch.tensor(0.0, device=device)
        return {"orth_loss": zero, "disc_loss": zero}

    # Orthogonality loss: per-expert mean vectors
    expert_mean_outputs = []
    for expert_id in sorted(all_expert_outputs_by_id.keys()):
        cat = torch.cat(all_expert_outputs_by_id[expert_id], dim=0)
        expert_mean_outputs.append(cat)
    orth_loss = compute_orthogonality_loss(expert_mean_outputs)

    # Discriminability loss (stop-gradient trick)
    all_outputs_cat = torch.cat(all_outputs, dim=0)  # (M_total, D)
    all_labels_cat = torch.cat(all_labels, dim=0)  # (M_total,)

    # Subsample if too many tokens to keep memory/compute reasonable
    max_disc_tokens = 4096
    if all_outputs_cat.shape[0] > max_disc_tokens:
        perm = torch.randperm(all_outputs_cat.shape[0], device=device)[:max_disc_tokens]
        all_outputs_cat = all_outputs_cat[perm]
        all_labels_cat = all_labels_cat[perm]

    # Cast to float32 for discriminator (small MLP, float32 is fine)
    all_outputs_f32 = all_outputs_cat.float()

    # Discriminator learns to classify (detached expert outputs)
    disc_logits_train = discriminator(all_outputs_f32.detach())
    disc_loss_for_disc = F.cross_entropy(disc_logits_train, all_labels_cat)

    # Experts are trained to make the discriminator's job easier, so the expert
    # path minimizes the same CE objective with a live computational graph.
    disc_logits_reward = discriminator(all_outputs_f32)
    disc_loss_for_experts = F.cross_entropy(disc_logits_reward, all_labels_cat)

    disc_loss = disc_loss_for_disc + disc_loss_for_experts

    return {"orth_loss": orth_loss, "disc_loss": disc_loss}
