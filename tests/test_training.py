# See LICENSE file for copyright and license details.
import sys
import unittest
from annpy.training import Trainer, TrainingHook
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

from annpy.training import ErrorHook, StatusHook

class Hooks(unittest.TestCase):
    def assertOutputEquals(self, expectedOutput):
        output = sys.stdout.getvalue().strip()
        self.assertEqual(output, expectedOutput)

    def setUp(self):
        class SimpleModel:
            def __init__(self, x):
                self.x = x

            def eval(self, y):
                return self.x * y

        class SimpleLR(TestLR):
            def step(self, batch):
                self._model.x = sum(batch)/len(batch)

        self.model = SimpleModel(x=0)
        self.lr = SimpleLR(self.model)
        self.trainer = Trainer(self.lr)

    def test01_hooks_can_be_attached(self):
        hook = StatusHook(self.trainer)
        self.assertIs(hook.trainer, self.trainer)

        self.assertOutputEquals('')

    def test02_hook_should_log_after_trainig(self):
        hook = StatusHook(self.trainer)
        self.trainer.train([torch.Tensor([[0]])])

        self.assertOutputEquals('Training started...\n'
                                '\tEpoch 1 started...\n'
                                '\tEpoch 1 finished.\n'
                                'Training ended.')

    def test03_hook_should_log_in_each_epoch(self):
        hook = StatusHook(self.trainer)
        self.trainer.train([torch.Tensor([[0]]), torch.Tensor([[1]])], epochs=2)

        self.assertOutputEquals('Training started...\n'
                                '\tEpoch 1 started...\n'
                                '\tEpoch 1 finished.\n'
                                '\tEpoch 2 started...\n'
                                '\tEpoch 2 finished.\n'
                                'Training ended.')
    def test04_hook_should_not_log_if_not_training(self):
        hook = StatusHook(self.trainer)
        self.trainer.train([])

        self.assertOutputEquals('')


    def test05_initally_errors_should_be_empty(self):
        dataset = [torch.Tensor([1])]

        hook = ErrorHook(self.trainer,
                         error_calculator=lambda: sum((self.model.eval(x)-x) for x in dataset)[0],
                         measure_before_training=False)

        self.assertEqual(hook.errors, [])

    def test05_error_hook_should_log_error_after_one_epoch(self):
        dataset = [torch.Tensor([1])]

        hook = ErrorHook(self.trainer,
                         error_calculator=lambda: sum((self.model.eval(x)-x) for x in dataset)[0],
                         measure_before_training=False)

        self.trainer.train([torch.Tensor([[0], [1]])])

        self.assertEqual(hook.errors, [-0.5])
        self.assertOutputEquals('Epoch 1 error: -0.5.')

    def test06_error_hook_should_log_error_in_each_epoch(self):
        dataset = [torch.Tensor([0.5])]

        hook = ErrorHook(self.trainer,
                         error_calculator=lambda: sum((self.model.eval(x)-x) for x in dataset)[0],
                         measure_before_training=False)

        self.trainer.train([torch.Tensor([[4],[2]])])
        self.trainer.train([torch.Tensor([[1],[3]])])

        self.assertEqual(hook.errors, [1.0, 0.5])
        self.assertOutputEquals("Epoch 1 error: 1.0.\n"
                                "Epoch 2 error: 0.5.")

    def test07_error_hook_can_log_error_before_training(self):
        dataset = [torch.Tensor([0.5])]

        hook = ErrorHook(self.trainer,
                         error_calculator=lambda: sum((self.model.eval(x)-x) for x in dataset)[0],
                         measure_before_training=True)

        self.trainer.train([torch.Tensor([[4],[2]])])
        self.trainer.train([torch.Tensor([[1],[3]])])

        self.assertEqual(hook.errors, [1.0, 0.5])
        self.assertOutputEquals("Error before training: -0.5.\n"
                                "Epoch 1 error: 1.0.\n"
                                "Error before training: 1.0.\n"
                                "Epoch 2 error: 0.5.")

    def test08_trainer_can_be_hooked_with_multiple_hooks(self):
        dataset = [torch.Tensor([1])]

        hook1 = StatusHook(self.trainer)

        hook2 = ErrorHook(self.trainer,
                          error_calculator=lambda: sum((self.model.eval(x)-x) for x in dataset)[0],
                          measure_before_training=False)

        self.trainer.train([torch.Tensor([[0], [1]])])


        self.assertIs(hook1.trainer, self.trainer)
        self.assertIs(hook2.trainer, self.trainer)

        self.assertEqual(hook2.errors, [-0.5])
        self.assertOutputEquals('Training started...\n'
                                '\tEpoch 1 started...\n'
                                '\tEpoch 1 finished.\n'
                                'Epoch 1 error: -0.5.\n'
                                'Training ended.')


if __name__ == '__main__':
    unittest.main()
