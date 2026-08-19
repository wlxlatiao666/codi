import torch
import torch.nn as nn
import pytest

from src.model import LearnableContrastiveAligner


def test_fixed_segment_alignment_requires_at_least_two_latents():
    with pytest.raises(ValueError, match="num_latent >= 2"):
        LearnableContrastiveAligner(dim=4, num_latent=1)


def test_segment_weights_are_contiguous_uniform_and_cover_valid_tokens():
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, False],
            [True, False, True, True, False, True, True, True, False],
        ]
    )

    weights, segment_len, valid_sample = (
        LearnableContrastiveAligner._build_segment_weights(
            teacher_cot_mask=mask,
            num_latent=3,
            dtype=torch.float32,
        )
    )

    # L=8 -> [0:2], [2:5], [5:8]; L=6 -> [0:2], [2:4], [4:6]
    assert segment_len.tolist() == [[2, 3, 3], [2, 2, 2]]
    assert valid_sample.tolist() == [True, True]
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 3))
    torch.testing.assert_close((weights > 0).sum(dim=1), mask.long())

    assert torch.nonzero(weights[0, 0], as_tuple=False).flatten().tolist() == [0, 1]
    assert torch.nonzero(weights[0, 1], as_tuple=False).flatten().tolist() == [2, 3, 4]
    assert torch.nonzero(weights[0, 2], as_tuple=False).flatten().tolist() == [5, 6, 7]

    # The second mask has holes; segmentation follows valid-token order and never
    # assigns masked positions.
    assert torch.nonzero(weights[1, 0], as_tuple=False).flatten().tolist() == [0, 2]
    assert torch.nonzero(weights[1, 1], as_tuple=False).flatten().tolist() == [3, 5]
    assert torch.nonzero(weights[1, 2], as_tuple=False).flatten().tolist() == [6, 7]


def test_diagonal_alignment_is_better_than_permuted_and_ignores_padding():
    aligner = LearnableContrastiveAligner(dim=3, num_latent=3, temperature=0.1)
    aligner.student_proj = nn.Identity()
    aligner.teacher_proj = nn.Identity()

    # Each two-token segment averages to one basis vector.
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

    assert aligned_loss < 1e-3
    assert permuted_loss > aligned_loss + 1.0
    assert torch.count_nonzero(weights[..., -1]) == 0

    aligned_loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_short_cot_sample_is_skipped_with_finite_zero_loss():
    aligner = LearnableContrastiveAligner(dim=4, num_latent=3)
    student = torch.randn(1, 3, 4, requires_grad=True)
    teacher = torch.randn(1, 2, 4)
    mask = torch.tensor([[True, True]])

    loss, _ = aligner(student, teacher, mask)

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad))
    assert all(parameter.grad is not None for parameter in aligner.parameters())
    assert all(
        torch.count_nonzero(parameter.grad) == 0
        for parameter in aligner.parameters()
    )


def test_short_sample_does_not_change_valid_sample_loss():
    torch.manual_seed(0)
    aligner = LearnableContrastiveAligner(dim=4, num_latent=3)
    valid_student = torch.randn(1, 3, 4)
    valid_teacher = torch.randn(1, 6, 4)
    valid_mask = torch.ones(1, 6, dtype=torch.bool)

    valid_loss, _ = aligner(valid_student, valid_teacher, valid_mask)

    mixed_student = torch.cat([valid_student, torch.randn(1, 3, 4)], dim=0)
    mixed_teacher = torch.cat([valid_teacher, torch.randn(1, 6, 4)], dim=0)
    mixed_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, False, False, False, False],
        ]
    )
    mixed_loss, _ = aligner(mixed_student, mixed_teacher, mixed_mask)

    torch.testing.assert_close(mixed_loss, valid_loss)
