# See LICENSE file for copyright and license details.
import time
from datetime import timedelta

import torch

class Trainer(object):
    def __init__(self, learning_rule):
        self._learning_rule = learning_rule
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    @staticmethod
    def check_batch(batch):
        if not isinstance(batch, torch.Tensor):
            raise Exception("Batchs must be torch.Tensor's")
        if len(batch.shape) <= 1:
            raise Exception("Batch shape must have at least dimension two")

    def train(self, batchs, epochs=1):
        if len(batchs) == 0:
            return

        for batch in batchs:
            self.check_batch(batch)

        for _ in range(epochs):
            for batch in batchs:
                self._learning_rule.step(batch)
            self._epoch += 1
