# See LICENSE file for copyright and license details.
import torch

class Trainer(object):
    def __init__(self, learning_rule):
        self._learning_rule = learning_rule
        self._epoch = 0
        self._hooks = []
        self._remaining_epochs = 0

    @property
    def epoch(self):
        return self._epoch

    @staticmethod
    def check_batch(batch):
        if not isinstance(batch, torch.Tensor):
            raise Exception("Batchs must be torch.Tensor's")
        if len(batch.shape) <= 1:
            raise Exception("Batch shape must have at least dimension two")

    def _notify(self, msg):
        for hook in self._hooks:
            hook.notify(msg)

    def train(self, batchs, epochs=1):
        if len(batchs) == 0:
            return

        for batch in batchs:
            self.check_batch(batch)

        self._remaining_epochs = epochs

        self._notify('pre_training')

        for _ in range(epochs):
            self._notify('pre_epoch')
            for batch in batchs:
                self._learning_rule.step(batch)
            self._notify('post_epoch')
            self._epoch += 1
            self._remaining_epochs -= 1

        self._notify('post_training')

    def remaining_epochs(self):
        return self._remaining_epochs


    def attach(self, hook):
        self._hooks.append(hook)
