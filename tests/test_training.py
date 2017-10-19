# See LICENSE file for copyright and license details.
import sys
import unittest
from annpy.training import Trainer
from annpy.base import LearningRule
import numpy as np
import torch

class TestModel(object):
    pass

class TestLR(LearningRule):
    def __init__(self, model):
        super(TestLR, self).__init__(model)
        self._batchs = []

    def step(self, batch):
        self._batchs.append(batch)

    def batchs(self):
        return self._batchs

class TrainerTest(unittest.TestCase):
    def assertEqualTensor(self, t1, t2):
        self.assertTrue((t1==t2).all())

    def setUp(self):
        self.training_batch = [[[torch.Tensor([0])], [1]], [[2], [3]]]
        self.model = TestModel()
        self.lr = TestLR(self.model)
        self.trainer = Trainer(self.lr)

    def test01_epoch_should_be_zero_before_training(self):
        self.assertEqual(self.trainer.epoch, 0)
        self.assertEqual(self.lr.batchs(), [])

    def test02_should_not_train_with_no_batchs(self):
        self.trainer.train([])
        self.assertEqual(self.trainer.epoch, 0)
        self.assertEqual(self.lr.batchs(), [])

    def test03_trains_one_epoch_one_batch_well(self):
        self.trainer.train([torch.Tensor([[0]])])
        self.assertEqual(self.trainer.epoch, 1)
        self.assertEqual(len(self.lr.batchs()), 1)
        self.assertEqualTensor(self.lr.batchs()[0], torch.Tensor([[0]]))

    def test04_training_should_fail_on_incorrect_batch_types(self):
        try:
            self.trainer.train([[[0]]])
            self.fail()
        except Exception as e:
            self.assertEqual(self.trainer.epoch, 0)
            self.assertEqual(len(self.lr.batchs()), 0)
            self.assertEqual(str(e), "Batchs must be torch.Tensor's")

        try:
            self.trainer.train([torch.Tensor([[0]]), [[1]]])
            self.fail()
        except Exception as e:
            self.assertEqual(self.trainer.epoch, 0)
            self.assertEqual(len(self.lr.batchs()), 0)
            self.assertEqual(str(e), "Batchs must be torch.Tensor's")

    def test05_training_should_fail_on_incorrect_batch_size(self):
        try:
            self.trainer.train([torch.Tensor([0])])
            self.fail()
        except Exception as e:
            self.assertEqual(self.trainer.epoch, 0)
            self.assertEqual(len(self.lr.batchs()), 0)
            self.assertEqual(str(e), "Batch shape must have at least dimension two")

    def test06_training_is_cumulative(self):
        self.trainer.train([torch.Tensor([[0], [1]])])
        self.trainer.train([torch.Tensor([[2], [3]])])

        self.assertEqual(self.trainer.epoch, 2)

        self.assertEqual(len(self.lr.batchs()), 2)
        self.assertEqualTensor(self.lr.batchs()[0], torch.Tensor([[0], [1]]))
        self.assertEqualTensor(self.lr.batchs()[1], torch.Tensor([[2], [3]]))

    def test06_can_not_train_zero_epochs(self):
        self.trainer.train([torch.Tensor([[0], [1]])], epochs=0)
        self.assertEqual(self.trainer.epoch, 0)
        self.assertEqual(len(self.lr.batchs()), 0)

    def test07_can_train_multiple_epochs(self):
        self.trainer.train([torch.Tensor([[0], [1]]),
                            torch.Tensor([[2], [3]])], epochs=2)

        self.assertEqual(self.trainer.epoch, 2)

        self.assertEqual(len(self.lr.batchs()), 4)

        self.assertEqualTensor(self.lr.batchs()[0], torch.Tensor([[0], [1]]))
        self.assertEqualTensor(self.lr.batchs()[1], torch.Tensor([[2], [3]]))
        self.assertEqualTensor(self.lr.batchs()[2], torch.Tensor([[0], [1]]))
        self.assertEqualTensor(self.lr.batchs()[3], torch.Tensor([[2], [3]]))
