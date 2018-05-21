import torch
from torch import nn
from torchtrainer.base import BatchTrainer, BatchValidator, ValidationGranularity
from torchtrainer.meters import Averager

class GANValidator(BatchValidator):
    def __init__(self, model, meters):
        super(GANValidator, self).__init__(model, meters)
        self._d_criterion = nn.BCEWithLogitsLoss()

    def real_targets(self, batch_size):
        target = torch.ones(batch_size, 1)
        return self._prepare_tensor(target)

    def fake_targets(self, batch_size):
        target = torch.zeros(batch_size, 1)
        return self._prepare_tensor(target)

    def validate_batch(self, x):
        batch_size = len(x.data)
        real_targets = self.real_targets(batch_size)
        fake_targets = self.fake_targets(batch_size)

        output_d_real = self.model.discriminate(x)
        loss_d_real = self._d_criterion(output_d_real, real_targets)
        generated_x = self.model.sample(batch_size)
        output_d_generated = self.model.discriminate(generated_x)
        loss_d_generated = self._d_criterion(output_d_generated, fake_targets)
        # Combine losses
        loss = loss_d_real + loss_d_generated

        self.meters['val_loss'].measure(loss.data[0])
        self.meters['val_d_loss'].measure(loss_d_real.data[0])
        self.meters['val_g_loss'].measure(loss_d_generated.data[0])

class GANTrainer(BatchTrainer):
    def create_validator(self):
        return GANValidator(self.model, self.val_meters)

    def __init__(self,
                 model,
                 d_optimizer,
                 g_optimizer,
                 callbacks=[],
                 logging_frecuency=10,
                 soft_labels_eps = 0):
        train_meters = {'train_loss' : Averager(),
                        'train_g_loss' : Averager(),
                        'train_d_loss' : Averager()}

        val_meters = {'val_loss' : Averager(),
                      'val_g_loss' : Averager(),
                      'val_d_loss' : Averager()}

        super(GANTrainer, self).__init__(model=model,
                                         train_meters=train_meters,
                                         val_meters=val_meters,
                                         callbacks=callbacks,
                                         logging_frecuency=logging_frecuency,
                                         validation_granularity=ValidationGranularity.AT_EPOCH)

        self._d_optimizer = d_optimizer
        self._g_optimizer = g_optimizer
        self._soft_labels_eps = soft_labels_eps
        self._d_criterion = nn.BCEWithLogitsLoss()

    def real_targets(self, batch_size):
        target = torch.rand(batch_size, 1) * self._soft_labels_eps + torch.ones(batch_size, 1) * (1 - self._soft_labels_eps/2)
        return self._prepare_tensor(target)

    def fake_targets(self, batch_size):
        target = torch.rand(batch_size, 1) * self._soft_labels_eps
        return self._prepare_tensor(target)

    def update_batch(self, x):
        batch_size = len(x.data)
        real_targets = self.real_targets(batch_size)
        fake_targets = self.fake_targets(batch_size)

        self._d_optimizer.zero_grad()

        # Discriminate real data
        output_d_real = self.model.discriminate(x)
        loss_d_real = self._d_criterion(output_d_real, real_targets)
        loss_d_real.backward()
        # Discriminate on generated data
        generated_x = self.model.sample(batch_size)
        output_d_generated = self.model.discriminate(generated_x)
        loss_d_generated = self._d_criterion(output_d_generated, fake_targets)
        loss_d_generated.backward()
        # Combine losses
        loss = loss_d_real + loss_d_generated
        self._d_optimizer.step()

        self.train_meters['train_loss'].measure(loss.data[0])
        self.train_meters['train_d_loss'].measure(loss_d_real.data[0])
        self.train_meters['train_g_loss'].measure(loss_d_generated.data[0])

        self._g_optimizer.zero_grad()
        generated_x = self.model.sample(batch_size)
        output_d_generated = self.model.discriminate(generated_x)
        loss_g = self._d_criterion(output_d_generated, real_targets)
        loss_g.backward()
        self._g_optimizer.step()
