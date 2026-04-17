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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

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

        # Route — cast router to input dtype (experts are already in input dtype)
        router_logits = F.linear(x_flat, self.router.weight.to(input_dtype))  # (N, E)
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
            weight = (topk_weights[mask] * slot_mask.float()).sum(dim=-1, keepdim=True)  # (M, 1)

            output[mask] += weight * expert_out

            if collect_expert_outputs:
                expert_outputs_for_diversity.append(expert_out.detach() if False else expert_out)
                expert_labels_for_diversity.append(
                    torch.full((expert_out.shape[0],), expert_idx, device=x.device, dtype=torch.long)
                )

        output = output.view(B, L, D)

        # Load-balancing loss (Switch Transformer style)
        # tokens_per_expert: fraction of tokens dispatched to each expert
        # mean_routing_prob: average router probability for each expert
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            tokens_per_expert[expert_idx] = (topk_indices == expert_idx).any(dim=-1).float().mean()
        mean_routing_prob = router_probs.mean(dim=0)  # (E,)
        load_balance_loss = self.num_experts * (tokens_per_expert * mean_routing_prob).sum()

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
                            alpha is a per-layer learnable scalar (init 1.0)
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
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"moe_residual_mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.original_mlp = original_mlp

        if freeze_original:
            for param in self.original_mlp.parameters():
                param.requires_grad = False

        self.moe = MoELayer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            original_mlp=original_mlp,
            expert_intermediate_size=expert_intermediate_size,
        )

        dtype = next(original_mlp.parameters()).dtype
        self.zeroconv = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        nn.init.zeros_(self.zeroconv.weight)

        if mode == "learned_gate":
            self.alpha = nn.Parameter(torch.ones(1, dtype=dtype))

        if mode == "scheduled_anneal":
            self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.long))
            self._anneal_steps = anneal_steps

    def forward(self, x: Tensor, collect_expert_outputs: bool = False) -> tuple[Tensor, dict]:
        orig_out = self.original_mlp(x)
        moe_out, moe_aux = self.moe(x, collect_expert_outputs=collect_expert_outputs)
        residual = self.zeroconv(moe_out.to(self.zeroconv.weight.dtype))

        if self.mode == "zeroconv":
            alpha = 1.0
        elif self.mode == "learned_gate":
            alpha = self.alpha
        elif self.mode == "scheduled_anneal":
            alpha = max(0.0, 1.0 - self._step_counter.item() / self._anneal_steps)

        output = alpha * orig_out + residual
        return output, moe_aux


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

    # Experts rewarded for being distinguishable (gradients flow to experts)
    disc_logits_reward = discriminator(all_outputs_f32)
    disc_loss_for_experts = -F.cross_entropy(disc_logits_reward, all_labels_cat)

    disc_loss = disc_loss_for_disc + disc_loss_for_experts

    return {"orth_loss": orth_loss, "disc_loss": disc_loss}
