from abc import abstractmethod
import torch
from torch.autograd import Variable
from torch import nn
from torchtrainer.utils.mixins import CudaMixin

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        # print("Initializing Linear")
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.001)
    elif classname.find('BatchNorm') != -1:
        # print("Initializing BatchNorm")
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.normal_(0.0, 0.001)

class Generator(nn.Module):
    @abstractmethod
    def sample_noise_input(self, batch_size):
        pass


class Discriminator(nn.Module):
    @abstractmethod
    def log_proba_discriminate(self, x):
        return 


class GAN(CudaMixin, nn.Module):
    USE_CUDA = False

    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.eval()

    def _sample_input(self, batchs):
        x = torch.randn(batchs, *self.generator.input_shape)
        return self._tensor_to_cuda(x)

    def sample(self, batchs):
        return self.generator(Variable(self._sample_input(batchs)))

    def discriminate(self, x):
        return self.discriminator(x)

    def discriminate_proba(self, x):
        return nn.Sigmoid()(self.discriminator(x))

    def cuda(self):
        super(GAN, self).cuda()
        self.generator.cuda()
        self.discriminator.cuda()

    def cpu(self):
        super(GAN, self).cpu()
        self.generator.cpu()
        self.discriminator.cpu()

class ConditionalGAN(GAN):
    def sample(self, labels):
        x = self._sample_input(len(labels))
        return self.generator(Variable(x), labels)

    def discriminate(self, x, labels):
        return self.discriminator(x, labels)

    def discriminate_proba(self, x, labels):
        return nn.Sigmoid()(self.discriminator(x, labels))
