import torch
import torch.nn as nn


import torch
import torch.nn as nn

class DoGConv2D(nn.Module):
    '''
    Center-surround convolution module

    Params:
        in_channels (int): Number of channels in input feature map
        out_channels (int): Number of channels in output feature map (should be same as in_channels)
        k (int): Kernel size = 2*k + 1
        stride (int): 2D convolutional stride
        padding (int): Input padding, as in nn.Conv2D
        dilation (int): Convolution dilation, as in nn.Conv2D
        groups (int): Convolution groups, as in nn.Conv2D
        bias (bool): Add bias to convolution output
    '''
    def __init__(self, in_channels, out_channels, k, stride, padding, dilation=1, groups=1, bias=True):
        super(DoGConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = int(k)
        self.kernel_size = 2*k+1
        self.stride = stride
        self.padding = 2*padding+1
        self.dilation = dilation
        self.groups = groups
        self.kernels = None

        x_coords = torch.arange(1, self.kernel_size+1, dtype=torch.float32).repeat(self.kernel_size, 1)
        y_coords = torch.arange(1, self.kernel_size+1, dtype=torch.float32).view(-1, 1).repeat(1, self.kernel_size)
        coords = torch.cat([x_coords.unsqueeze(0), y_coords.unsqueeze(0)], dim=0)
        kernel_dists = torch.sum(torch.square((coords-(k+1))), dim=0)
        self.kernel_dists = nn.Parameter(kernel_dists, requires_grad=False)
        self.sigma1 = nn.Parameter(torch.rand((out_channels*in_channels, 1, 1))+(k/2), requires_grad=True) # Excitatory gaussian sigma
        self.sigma2_scale = nn.Parameter(torch.rand((out_channels*in_channels, 1, 1))+(k/2), requires_grad=True) # Inhibitory gaussian sigma=sigma2_scale*sigma1
        self.total_scale = nn.Parameter(torch.randn((out_channels*in_channels, 1, 1))*2., requires_grad=True) # total scaling factor of DoG
        
        if bias:
            self.bias = nn.Parameter(torch.randn(self.out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=False)
        
        

    def forward(self, x):
        sigma1 = torch.clamp(self.sigma1, min=0.1)
        excite_component = (1/torch.pi*sigma1)*torch.exp(-1*self.kernel_dists/sigma1)
        sigma2 = sigma1 * torch.clamp(self.sigma2_scale, min=1+1e-4)
        inhibit_component = (1/torch.pi*sigma2)*torch.exp(-1*self.kernel_dists/torch.clamp(sigma2, min=1e-6))
        kernels = (excite_component - inhibit_component)*self.total_scale*-1.0
        kernels = torch.nn.functional.normalize(kernels, dim=0)
        kernels = kernels.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.kernels = kernels
        return nn.functional.conv2d(x, self.kernels, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class DoGConv2DLayer(nn.Module):
    '''
    Wrapper for center surround convolution

    Params:
        dog_channels (int): Number of channels to apply center-surround convolution to 
                            (should be less than or equal to number of channels in input feature map).
                            Remaining channels will not be changed.
        k (int):  Center-surround kernsl size = 2*k + 1
        stride (int): 2D convolutional stride
        padding (int): Input padding, as in nn.Conv2D
        dilation (int): Convolution dilation, as in nn.Conv2D
        groups (int): Convolution groups, as in nn.Conv2D
        bias (bool): Add bias to convolution output
    '''
    def __init__(self, dog_channels, k, stride, padding, dilation=1, groups=1, bias=True):
        super(DoGConv2DLayer, self).__init__()
        self.dog_conv = DoGConv2D(dog_channels, dog_channels, k=k, stride=stride, padding=padding, \
                                     dilation=dilation, groups=groups, bias=bias)
        self.dog_channels = dog_channels
        self.non_dog_transform = nn.Identity()

    def forward(self, x):
        x_dog = self.dog_conv(x[:,:self.dog_channels,:,:])
        x_non_dog = self.non_dog_transform(x[:,self.dog_channels:,:,:])
        return torch.concat([x_dog, x_non_dog], dim=1)