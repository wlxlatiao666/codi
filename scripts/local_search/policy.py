"""Small, discrete, monotone cutpoint policies; no model or torch dependency.

Each pair is [latent checkpoint, omitted-CoT ratio numerator / 6].
"""
import math
import random

INITIAL = [[3, 2], [4, 2], [5, 4]]


def validate(pairs):
    assert len(pairs) == 3
    assert all(isinstance(v, int) for p in pairs for v in p)
    assert all(1 <= k <= 5 and 1 <= q <= 5 for k, q in pairs)
    assert all(pairs[i][0] < pairs[i + 1][0] for i in range(2))
    assert all(pairs[i][1] <= pairs[i + 1][1] for i in range(2))
    return pairs


def cut(m, q):
    assert m >= 2 and 1 <= q <= 5
    return min(m - 1, max(1, (m * q + 5) // 6))


def signature(pairs, lengths):
    return tuple((k, tuple(cut(m, q) for m in sorted(lengths))) for k, q in pairs)


def neighbors(pairs, lengths):
    validate(pairs)
    seen = {signature(pairs, lengths)}
    result = []
    for slot in range(3):
        for axis in range(2):
            for delta in (-1, 1):
                child = [list(p) for p in pairs]
                child[slot][axis] += delta
                try:
                    validate(child)
                except AssertionError:
                    continue
                sig = signature(child, lengths)
                if sig in seen:
                    continue
                seen.add(sig)
                result.append(dict(pairs=child, slot=slot,
                                   axis='k' if axis == 0 else 'ratio_sixths', delta=delta))
    return result


def propose(pairs, lengths, round_index, seed=20261015):
    options = neighbors(pairs, lengths)
    if not options:
        raise ValueError('No effective legal local mutations')
    # Balance the two mutation families where both are available.
    rng = random.Random(seed + 1000003 * round_index)
    axes = sorted({x['axis'] for x in options})
    axis = rng.choice(axes)
    return rng.choice([x for x in options if x['axis'] == axis])


def selection(parent, child, margin=0.01):
    """Counts from the same two validation snapshots; ties retain parent."""
    assert len(parent) == len(child) == 2
    assert all(p['total'] == c['total'] == parent[0]['total']
               for p, c in zip(parent, child))
    total = sum(x['total'] for x in parent)
    gain = sum(x['correct'] for x in child) - sum(x['correct'] for x in parent)
    minimum = math.ceil(margin * total)
    return dict(winner='child' if gain >= minimum else 'parent',
                net_correct_over_two_snapshots=gain, required_net_correct=minimum,
                mean_accuracy_gain=gain / total, margin=margin)
