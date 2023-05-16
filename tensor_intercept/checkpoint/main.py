'''

Small model training code with checkpoints

'''

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class ExampleModel(nn.Module):
    def __init__(self):
        super(ExampleModel, self).__init__()
        self.layers = nn.ModuleList([nn.Conv2d(64 if i > 0 else 3, 64, kernel_size=3) for i in range(10)])

    def forward(self, x):
        checkpoint_layers = [0, 1, 2, 5, 6, 7]  # Layers to use checkpoint, zero-indexed
        for idx, layer in enumerate(self.layers):
            if idx in checkpoint_layers:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        return x
    
model = ExampleModel()
input = torch.randn(1, 3, 224, 224, requires_grad=True)  # Set requires_grad=True here
output = model(input)
output.backward(torch.ones_like(output))
