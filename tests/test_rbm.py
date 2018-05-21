# See LICENSE file for copyright and license details.
import unittest
import numpy as np
from numpy import testing
from annpy.models.rbm import RBM
import torch

class RBMTestCase(unittest.TestCase):
    def assertEqualTensor(self, t1, t2):
        self.assertTrue((t1==t2).all())

    def assertAlmostEqual(self, t1, t2):
        testing.assert_array_almost_equal(t1.numpy(), t2.numpy())

    def test_network_should_initialize_with_correct_dimentions(self):
        m = RBM(19, 14)

        self.assertEqual(m.visible_units, 19)
        self.assertEqual(m.hidden_units,  14)
        self.assertEqual(repr(m), 'RBM (19 -> 14)')
        self.assertEqual(m.weights.data.shape, (19, 14))
        self.assertEqual(m.visible_biases.data.shape, (19,))
        self.assertEqual(m.hidden_biases.data.shape, (14,))

    def test_network_initialize_correctly_weights(self):
        w = torch.Tensor([[1,   0.2],
                          [0.2,   1],
                          [1,   0.2]])
        v = torch.Tensor([1, 0.5, 0.5])
        h = torch.Tensor([0, 1])
        m = RBM(3, 2, weights=w, visible_biases=v, hidden_biases=h)

        self.assertEqual(m.visible_units, 3)
        self.assertEqual(m.hidden_units,  2)
        self.assertEqual(repr(m), 'RBM (3 -> 2)')
        self.assertEqualTensor(m.weights.data, w)
        self.assertEqualTensor(m.visible_biases.data, v)
        self.assertEqualTensor(m.hidden_biases.data, h)

    def test_network_gives_correct_stimolous(self):
        w = torch.Tensor([[1,   0.2],
                          [0.2,   1],
                          [1,   0.2]])
        v = torch.Tensor([1, 0.5, 1.0])
        h = torch.Tensor([2, 1])
        m = RBM(3, 2, weights=w, visible_biases=v, hidden_biases=h)

        h1 = torch.Tensor([1, 0])
        h2 = torch.Tensor([0, 1])

        v1 = torch.Tensor([1, 0, 0])
        v2 = torch.Tensor([0, 1, 0])
        v3 = torch.Tensor([0, 0, 1])

        sigmoid = lambda x: 1.0 / (1.0 + torch.exp(-x))

        self.assertAlmostEqual(m._net_visible(h1), w[:,0] + v)
        self.assertAlmostEqual(m._net_visible(h2), w[:,1] + v)

        self.assertAlmostEqual(m._net_hidden(v1), w[0] + h)
        self.assertAlmostEqual(m._net_hidden(v2), w[1] + h)
        self.assertAlmostEqual(m._net_hidden(v3), w[2] + h)

    def test_network_associate_per_batch(self):
        w = torch.Tensor([[1,   0.2],
                          [0.2,   1],
                          [1,   0.2]])
        v = torch.Tensor([1, 0.5, 1.0])
        h = torch.Tensor([2, 1])
        m = RBM(3, 2, weights=w, visible_biases=v, hidden_biases=h)

        h1 = torch.Tensor([[1, 0],
                           [0, 1],
                           [0, 1],
                           [1, 0]])

        v1 = torch.Tensor([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

        self.assertAlmostEqual(m._net_visible(h1), torch.Tensor([[2   , .7   , 2] ,
                                                                 [1.2 , 1.5  , 1.2],
                                                                 [1.2 , 1.5  , 1.2],
                                                                 [2   , .7   , 2]]))

        self.assertAlmostEqual(m._net_hidden(v1), torch.Tensor([[3   , 1.2],
                                                                [2.2 , 2.0],
                                                                [2.2 , 2.0],
                                                                [3   , 1.2]]))
