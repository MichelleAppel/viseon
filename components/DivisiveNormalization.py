import torch
import torch.nn as nn

class DivisiveNormalizationCirincione(nn.Module):
    '''
    Divisive normalization implemented following to the details provided in 
    https://openreview.net/pdf?id=KAAbo44qhJV

    Parameters:
        channels (int): Number of channels in input feature map
        gaussian_kernel_size (int): kernel size of spatially integrating gaussian filter

    '''
    def __init__(self, channels, gaussian_kernel_size):
        super(DivisiveNormalizationCirincione, self).__init__()
        self.two_pi = (2.*torch.pi)
        self.gaussian_channels = channels
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_padding = self.gaussian_kernel_size//2
        x_coords = torch.arange(1,self.gaussian_kernel_size+1,dtype=torch.float32).repeat(self.gaussian_kernel_size,1)
        y_coords = torch.arange(1,self.gaussian_kernel_size+1,dtype=torch.float32).view(-1,1).repeat(1, self.gaussian_kernel_size)
        self.register_buffer('x_coords', x_coords)
        self.register_buffer('y_coords', y_coords)
        
        kernel_param_size = (self.gaussian_channels,self.gaussian_channels, 1, 1)
        self.A = nn.Parameter(torch.rand(kernel_param_size), requires_grad = True)
        self.rho = nn.Parameter(torch.rand(kernel_param_size)+1.0, requires_grad = True)
        self.sigma = nn.Parameter(torch.rand(kernel_param_size)+1.0, requires_grad = True)
        self.theta = nn.Parameter(torch.rand(kernel_param_size)*self.two_pi, requires_grad = True)
        self.v = nn.Parameter(torch.randn(kernel_param_size)+(gaussian_kernel_size//2), requires_grad = True)
        self.w = nn.Parameter(torch.randn(kernel_param_size)+(gaussian_kernel_size//2), requires_grad = True)
        self.bias = nn.Parameter(torch.rand(1)+1., requires_grad = True)
        
    def forward(self, x):
        x_rot = self.x_coords*torch.cos(self.theta) + self.y_coords*torch.sin(self.theta)
        y_rot = -1*self.x_coords*torch.sin(self.theta) + self.y_coords*torch.cos(self.theta)

        kernel_dists = (((x_rot-self.v)**2)/(torch.clamp(self.rho**2, min=0.1))) \
                        + (((y_rot-self.w)**2)/(torch.clamp(self.sigma**2, min=0.1)))
        gaussian_kernels = torch.abs((self.A/torch.clamp((self.two_pi*self.rho*self.sigma), 0.1))) \
                            * torch.exp(-0.5*(kernel_dists))
        normalizer = nn.functional.conv2d(x, gaussian_kernels, padding=self.gaussian_padding)
        return x/(normalizer+torch.abs(self.bias))


class OpponentChannelInhibition(nn.Module):
    '''
    Tuned normalization.

    Parameters:
        n_channels (int): Number of channels in input feature map
    '''
    def __init__(self, n_channels):
        super(OpponentChannelInhibition, self).__init__()
        self.n_channels = n_channels
        channel_inds = torch.arange(n_channels, dtype=torch.float32)+1.
        channel_inds = channel_inds-channel_inds[n_channels//2]
        self.channel_inds = torch.abs(channel_inds)
        channel_distances = []
        for i in range(n_channels):
            channel_distances.append(torch.roll(self.channel_inds, i))
        self.channel_distances = nn.Parameter(torch.stack(channel_distances), requires_grad=False)
        self.sigmas = nn.Parameter(torch.rand(n_channels)+(n_channels/8), requires_grad=True)

    def forward(self, x):
        sigmas = torch.clamp(self.sigmas, min=0.5)
        gaussians = (1/(2.5066*sigmas))*torch.exp(-1*(self.channel_distances**2)/(2*(sigmas**2))) # sqrt(2*pi) ~= 2.5066
        gaussians = gaussians/torch.sum(gaussians, dim=0)
        gaussians = gaussians.view(self.n_channels,self.n_channels,1,1)
        weighted_channel_inhibition = nn.functional.conv2d(x, weight=gaussians, stride=1, padding=0)
        return x/(weighted_channel_inhibition+1)

