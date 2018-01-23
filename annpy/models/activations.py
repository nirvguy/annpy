import torch

class Softmax(torch.nn.Softmax):
    def forward(self, x):
        if x.dim() == 1:
            return super(MySoftmax, self).forward(x.unsqueeze(0)).view(-1)
        else:
            return super(MySoftmax, self).forward(x)
