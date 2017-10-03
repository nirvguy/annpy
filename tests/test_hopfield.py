import unittest
import numpy as np
from numpy import testing
from numpy import random as np_rnd
from annpy.hopfield import Hopfield, HebbsRule
from itertools import product
import torch

class HopfieldReconstructTest(unittest.TestCase):
    def test_pattern_retrive_same_pattern_when_fit(self):
        rn = Hopfield(4)
        training_rule = HebbsRule(rn)

        pattern = torch.Tensor([1, -1, 1, 1])
        patterns = [pattern]

        training_rule.step(patterns)

        retrived_pattern = rn.reconstruct(torch.Tensor(pattern))
        self.assertTrue((retrived_pattern == pattern).all())

    def test_destroyed_pattern_retrive_correct_pattern(self):
        rn = Hopfield(4)
        training_rule = HebbsRule(rn)

        pattern = torch.Tensor([1, 1, -1, 1])
        training_rule.step([pattern])

        destroyed_patterns = map(torch.Tensor, product([1, -1], repeat=4))
        for destroyed_pattern in destroyed_patterns:
            self.assertTrue((rn.reconstruct(pattern) == pattern).all())

    @staticmethod
    def random_state(n):
        return torch.Tensor([1 if np_rnd.choice(2) == 1 else -1 for i in range(n)])

    @staticmethod
    def destroyed_pattern(pattern, n, m):
        for _ in range(m):
            mult_list = [1] * (len(pattern) - n) + [-1] * n
            np_rnd.shuffle(mult_list)
            yield pattern * torch.Tensor(mult_list)

    def test_multiple_patterns(self):
        inputs = 100
        rn = Hopfield(inputs=inputs)
        training_rule = HebbsRule(rn)

        number_of_patterns = 5
        tests_per_pattern = 4

        patterns = [self.random_state(inputs) for _ in range(number_of_patterns)]

        training_rule.step(patterns)

        for pattern in patterns:
            for i in range(10):
                for try_pattern in self.destroyed_pattern(pattern, i, tests_per_pattern):
                    self.assertTrue((rn.reconstruct(try_pattern) == pattern).all())

if __name__ == '__main__':
    unittest.main()
