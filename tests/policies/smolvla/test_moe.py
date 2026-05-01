import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.smolvla.moe import (
    ExpertDiscriminator,
    MoELayer,
    ResidualMoELayer,
    _compute_router_statistics,
    compute_diversity_losses,
)


def test_compute_router_statistics_normalizes_topk_assignments():
    router_probs = torch.full((3, 4), 0.25)
    topk_indices = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )

    assignment_fraction, load_balance_loss = _compute_router_statistics(
        topk_indices=topk_indices,
        router_probs=router_probs,
        num_experts=4,
    )

    expected_fraction = torch.tensor([1.0, 2.0, 2.0, 1.0]) / 6.0
    assert torch.allclose(assignment_fraction, expected_fraction)
    assert torch.isclose(assignment_fraction.sum(), torch.tensor(1.0))
    assert torch.isclose(load_balance_loss, torch.tensor(1.0))


def test_moe_layer_reports_normalized_expert_utilization_for_topk():
    torch.manual_seed(0)
    original_mlp = nn.Identity()
    moe = MoELayer(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        original_mlp=original_mlp,
    )

    with torch.no_grad():
        moe.router.weight.zero_()

    x = torch.randn(2, 3, 4)
    _, aux = moe(x)

    assert torch.isclose(aux["tokens_per_expert"].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(aux["load_balance_loss"], torch.tensor(1.0), atol=1e-6)


def test_compute_diversity_losses_uses_minimization_sign_for_experts():
    discriminator = ExpertDiscriminator(hidden_size=2, num_experts=2, disc_hidden_size=4)
    with torch.no_grad():
        first = discriminator.net[0]
        second = discriminator.net[2]
        first.weight.copy_(torch.eye(2, 2).repeat(2, 1))
        first.bias.zero_()
        second.weight.zero_()
        second.bias.zero_()
        second.weight[0, 0] = 8.0
        second.weight[1, 1] = 8.0

    expert0 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    expert1 = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    moe_aux = [
        {
            "load_balance_loss": torch.tensor(0.0),
            "expert_outputs": [expert0, expert1],
            "expert_labels": [torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long)],
        }
    ]

    losses = compute_diversity_losses(moe_aux, discriminator)
    expected_ce = F.cross_entropy(
        discriminator(torch.cat([expert0, expert1], dim=0).float()),
        torch.tensor([0, 0, 1, 1]),
    )

    assert torch.isclose(losses["disc_loss"], expected_ce * 2, atol=1e-6)


def test_residual_moe_learned_gate_can_be_sigmoid_constrained():
    layer = ResidualMoELayer(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        original_mlp=nn.Identity(),
        expert_intermediate_size=None,
        mode="learned_gate",
        learned_gate_use_sigmoid=True,
    )

    alpha = layer._get_learned_alpha(torch.float32)
    assert 0.0 < alpha.item() < 1.0
    assert torch.isclose(alpha, torch.tensor([0.999]), atol=1e-6).all()
