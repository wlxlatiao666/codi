"""Search invariants, including integer-cut plateaus and conservative selection."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('policy', Path(__file__).resolve().parents[1] / 'scripts/local_search/policy.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PolicyTests(unittest.TestCase):
    def test_integer_cut_and_bounds(self):
        for m in range(2, 100):
            values = [p.cut(m, q) for q in range(1, 6)]
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(1 <= j < m for j in values))
        self.assertEqual([p.cut(2, q) for q in range(1, 6)], [1] * 5)
        self.assertEqual(p.cut(3, 2), 1)

    def test_only_effective_neighbors(self):
        self.assertTrue(all(x['axis'] == 'k' for x in p.neighbors(p.INITIAL, [2])))
        for x in p.neighbors(p.INITIAL, range(2, 9)):
            p.validate(x['pairs'])
            changes = sum(a != b for old, new in zip(p.INITIAL, x['pairs']) for a, b in zip(old, new))
            self.assertEqual(changes, 1)

    def test_reproducible_proposals(self):
        self.assertEqual(p.propose(p.INITIAL, [2, 3, 4], 0), p.propose(p.INITIAL, [2, 3, 4], 0))

    def test_margin_and_ties(self):
        parent = [dict(correct=100, total=512)] * 2
        self.assertEqual(p.selection(parent, parent)['winner'], 'parent')
        self.assertEqual(p.selection(parent, [dict(correct=105, total=512)] * 2)['winner'], 'parent')
        result = p.selection(parent, [dict(correct=105, total=512), dict(correct=106, total=512)])
        self.assertEqual(result['winner'], 'child')
        self.assertEqual(result['required_net_correct'], 11)


if __name__ == '__main__':
    unittest.main()
