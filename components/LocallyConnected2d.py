import torch
from torch import nn
from torch.nn.modules.utils import _pair


class LocallyConnected2d(nn.Module):
    '''
    Locally connected layer.  Implementation derived from @ptrblck
    https://discuss.pytorch.org/t/locally-connected-layers/26979

    Parameters:
        in_channels (int): number of channels in input feature map
        out_channels (int): number of channels in output feature map
        output_size (int): spatial dimension of output size (expected square output size)
        kernel_size (int): Size of equivalent convolutional kernel.  Size of locally connected neighborhood
        stride (int): Kernel stride (equivalent effect to that in nn.Conv2d) 
        padding (int): input padding (equivalent effect to that in nn.Conv2d)
        bias (bool): Include bias addition after convolution
    '''
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        self.padding = tuple([padding]*4)
        
    def forward(self, x):
        x = nn.functional.pad(x, self.padding)
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out