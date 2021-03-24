import torch
import torch.nn as nn
import torch.nn.functional as F
import time

thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function

# Define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if input.requires_grad:
            ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


class SCNN(nn.Module):
    def __init__(self, device, decay=0.8, init_gain=2.5, input_size=(304, 240, 2)): # input_size (x, y, c)
        super(SCNN, self).__init__()
        self.device = device
        self.decay = decay
        self.layers = []
        
        self.x = input_size[0]
        self.y = input_size[1]
        self.c = input_size[2]
        self.init_gain = init_gain  # Ensure starting spikerate
        self.s = "Input: " + str((1, self.c, self.y, self.x)) + "\n"  # string representation

        self.model = nn.ModuleList() # For pytorch functionality 
        
        # Define network layout
        self.layers = (
            self.Conv2d(self, 16),
            self.MaxPool(self),
            self.Conv2d(self, 32),
            self.MaxPool(self),
            self.Conv2d(self, 64),
            self.MaxPool(self),
            self.Conv2d(self, 128),
            self.MaxPool(self),
            self.Conv2d(self, 256),
            self.MaxPool(self, stride=1),
            self.Conv2d(self, 512),
            self.MaxPool(self, stride=1),
            self.Conv2d(self, 1024)
        )
        
        self.output_spikes = torch.zeros(1, self.c, self.y, self.x, device=self.device)
        
    # Feed new inputs for continuous feedforward
    def feed(self, inputs):
        # Calculate firing based on input probabilities (approximates non-binary values such as gray)
        x = (inputs > torch.rand(inputs.size(), device=self.device)).float()
        
        for _, layer in enumerate(self.layers):
            x = layer.forward(x)
        
        self.output_spikes += x

    # Collect output from network
    def collect(self):
        outputs = self.output_spikes
        self.clear()
        return outputs

    # Clear output spikes
    def clear(self):
        self.output_spikes = torch.zeros(self.output_spikes.shape, device=self.device)

    # Reset membrane potentials and output spikes
    def reset_potentials(self):
        for i, layer in enumerate(self.layers):
            layer.reset()
        self.clear()

    def __str__(self):
        return self.s

    class Conv2d():
        def __init__(self, snn, f, size=3, stride=1):
            padding=1
            if snn.x % stride != 0 or snn.y % stride != 0:
                raise ValueError("Layer input", x, y, "would be rounded with stride", stride)

            self.snn = snn
            self.conv = nn.Conv2d(snn.c, f, kernel_size=size, stride=stride, padding=padding)
            nn.init.xavier_uniform_(self.conv.weight.data, gain=snn.init_gain) # Ensures an initial spike-rate

            snn.c = f
            snn.x = snn.x // stride
            snn.y = snn.y // stride
            snn.s += "Conv: " + str((1, snn.c, snn.y, snn.x)) + "\n"

            self.mem = self.spike = torch.zeros(1, snn.c, snn.y, snn.x, device=snn.device)
            snn.model.extend([self.conv]) # Used internally by pytorch

        # Iterative LIF model
        def forward(self, x):
            # Reset potential if last iteration spiked, otherwise decay it. Add new input.
            self.mem = self.mem * (1 - self.snn.decay) * (1. - self.spike) + self.conv(x)  
            self.spike = act_fun(self.mem)  # Approximation firing
            return self.spike
        
        # Reset membrane potentials
        def reset(self):
            self.mem = self.spike = torch.zeros(self.mem.shape, device=self.snn.device)            

    class AvgPool():
        def __init__(self, snn, size=2, stride=2):
            self.size = size
            self.stride = stride

            if snn.x % stride != 0 or snn.y % stride != 0:
                raise ValueError("Layer input size", snn.x, snn.y, "would get rounded with stride", stride)
            
            if stride == 1:  # For 0 padding
                snn.x -= size-1
                snn.y -= size-1
            else:
                snn.x = snn.x // stride
                snn.y = snn.y // stride
            snn.s += "Avg pool: " + str((1, snn.c, snn.y, snn.x)) + "\n"

        def forward(self, x):
            return F.avg_pool2d(x, self.size, stride=self.stride)

        def reset(self):
            return
        
    class MaxPool():
        def __init__(self, snn, size=2, stride=2):
            self.size = size
            self.stride = stride
            
            if snn.x % stride != 0 or snn.y % stride != 0:
                raise ValueError("Layer input size", snn.x, snn.y, "would get rounded with stride", stride)

            if stride == 1:  # For 0 padding
                snn.x -= size-1
                snn.y -= size-1
            else:
                snn.x = snn.x // stride
                snn.y = snn.y // stride
            snn.s += "Max pool: " + str((1, snn.c, snn.y, snn.x)) + "\n"

        def forward(self, x):
            return F.max_pool2d(x, self.size, stride=self.stride)

        def reset(self):
            return
