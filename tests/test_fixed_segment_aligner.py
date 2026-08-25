import torch
import torch.nn as nn
import pytest

from src.model import LearnableContrastiveAligner


def test_fixed_segment_alignment_requires_at_least_two_latents():
    with pytest.raises(ValueError, match="num_latent >= 2"):
        LearnableContrastiveAligner(dim=4, num_latent=1)


def test_soft_weights_are_convex_and_ignore_padding():
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, False],
            [True, False, True, True, False, True, True, True, False],
        ]
    )
    aligner = LearnableContrastiveAligner(dim=4, num_latent=3)
    weights, _, valid_sample = aligner._build_segment_weights(
        teacher_cot_mask=mask,
        num_latent=3,
        dtype=torch.float32,
    )

    # Each (sample, latent) row is a convex combination over valid tokens.
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))
    # Padding / masked positions carry zero weight.
    assert torch.count_nonzero(weights[0, :, -1]) == 0
    assert torch.count_nonzero(weights[1, :, 1]) == 0
    assert torch.count_nonzero(weights[1, :, 4]) == 0
    # Every sample with at least one valid token is used.
    assert valid_sample.tolist() == [True, True]


def test_centers_are_monotonic_and_span_unit_interval():
    aligner = LearnableContrastiveAligner(dim=4, num_latent=4)
    raw = aligner.segment_center_raw
    centers = torch.cumsum(torch.nn.functional.softplus(raw), dim=-1)
    centers = centers / centers[-1].clamp_min(1e-6)

    # Strictly increasing, last anchored at 1.0, all in (0, 1].
    assert torch.all(centers[1:] > centers[:-1])
    torch.testing.assert_close(centers[-1], torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_diagonal_alignment_is_better_than_permuted_and_ignores_padding():
    aligner = LearnableContrastiveAligner(dim=3, num_latent=3, temperature=0.1)
    aligner.student_proj = nn.Identity()
    aligner.teacher_proj = nn.Identity()

    # Concentrate soft weight near segment centers so each latent maps to its basis vector.
    # With identity projections and low temperature, diagonal alignment should win.
    teacher = torch.tensor(
        [[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [999.0, 999.0, 999.0],
        ]]
    )
    mask = torch.tensor([[True, True, True, True, True, True, False]])
    student = torch.eye(3).unsqueeze(0).requires_grad_()

    aligned_loss, weights = aligner(student, teacher, mask)
    permuted_loss, _ = aligner(student[:, [1, 0, 2]], teacher, mask)

    assert aligned_loss < permuted_loss
    assert torch.count_nonzero(weights[..., -1]) == 0

    aligned_loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_short_cot_sample_yields_finite_zero_loss():
    aligner = LearnableContrastiveAligner(dim=4, num_latent=3)
    student = torch.randn(1, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 2, 4)
    mask = torch.tensor([[True, True]])

    loss, _ = aligner(student, teacher, mask)

    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_empty_cot_batch_yields_finite_zero_loss():
    aligner = LearnableContrastiveAligner(dim=4, num_latent=3)
    student = torch.randn(2, 3, 4, requires_grad=True)
    teacher = torch.randn(2, 5, 4)
    mask = torch.zeros(2, 5, dtype=torch.bool)

    loss, _ = aligner(student, teacher, mask)

    assert torch.isfinite(loss)
    assert torch.count_nonzero(loss) == 0
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    # Projection heads stay in the graph so DDP doesn't flag them as unused.
    assert all(parameter.grad is not None for parameter in aligner.parameters())
