import torch
from torchvision import transforms

import torch.nn as nn


class PolarTransform(nn.Module):
    def __init__(self, input_h, input_w, output_h, output_w, radius_bins, angle_bins, interpolation=None):
        """
        Polar Transform module that converts an input image into polar coordinates.

        Args:
            input_h (int): Height of the input image.
            input_w (int): Width of the input image.
            output_h (int): Height of the output image.
            output_w (int): Width of the output image.
            radius_bins (list): List of radius bins for polar transformation.
            angle_bins (list): List of angle bins for polar transformation.
            interpolation (str, optional): Interpolation mode for resizing. Defaults to None.
        """
        super(PolarTransform, self).__init__()

        # Polar coordinate rho and theta vals for each input location
        center_h, center_w = int(input_h / 2), int(input_w / 2)
        x_coords = torch.arange(input_w).repeat(input_h, 1) - center_w
        y_coords = center_h - torch.arange(input_h).unsqueeze(-1).repeat(1, input_w)
        distances = torch.sqrt(x_coords ** 2 + y_coords ** 2)
        angles = torch.atan2(y_coords, x_coords)
        angles[y_coords < 0] += 2 * torch.pi

        # Registers the distance and angle tensors as buffers, 
        # ensuring they are not treated as parameters and do not require gradient computation.
        self.register_buffer('distances', distances)
        self.register_buffer('angles', angles)
        
        self.radius_bins = radius_bins
        self.angle_bins = angle_bins
        self.n_radii = len(radius_bins) - 1
        self.n_angles = len(angle_bins) - 1
        self.edge_radius = min(center_h, center_w)

        # Create pooling masks for each radius and angle bin
        pooling_masks = []
        for i, (min_dist, max_dist) in enumerate(zip(radius_bins, radius_bins[1:])):
            in_distance = torch.logical_and(distances >= min_dist, distances < max_dist)
            for j, (min_angle, max_angle) in enumerate(zip(angle_bins, angle_bins[1:])):
                in_angle = torch.logical_and(angles >= min_angle, angles < max_angle)
                ind_mask = torch.logical_and(in_distance, in_angle).to(torch.float32)
                pooling_masks.append(ind_mask)
        pooling_masks = torch.stack(pooling_masks).view(self.n_radii * self.n_angles, input_h * input_w)

        # Interpolate the pooling masks to fill in missing values
        if interpolation:
            for mask_idx in range(0, pooling_masks.shape[0], self.n_angles):
                radius = radius_bins[mask_idx // self.n_angles]
                if radius > self.edge_radius:
                    continue
                radius_masks = pooling_masks[mask_idx:mask_idx + self.n_angles]
                nonzero_masks = radius_masks[torch.sum(radius_masks, dim=1).to(torch.bool)]
                interpolated_masks = torch.nn.functional.interpolate(
                    nonzero_masks.view(-1, input_h, input_w).permute(1, 2, 0), size=self.n_angles, mode=interpolation).permute(2, 0, 1)
                pooling_masks[mask_idx:mask_idx + self.n_angles] = interpolated_masks.view(-1, input_h * input_w)

        pooling_mask_counts = torch.sum(pooling_masks, dim=1)
        pooling_mask_counts[pooling_mask_counts == 0] = 1

        # The pooling masks are created once during initialization and stored as buffers.
        # This avoids recomputation during each forward pass.
        self.register_buffer('pooling_masks', pooling_masks)
        self.register_buffer('pooling_mask_counts', pooling_mask_counts)

        self.output_transform = transforms.Resize((output_h, output_w), 
                                                  interpolation=transforms.InterpolationMode.BILINEAR, 
                                                  antialias=False)

    def forward(self, x):
        """
        Forward pass of the Polar Transform module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Transformed output tensor of shape (batch_size, channels, output_h, output_w).
        """
        n, c, h, w = x.size()
        x_reshaped = x.view(n * c, -1)  # Reshape to (n*c, input_h * input_w)

        # Uses matrix multiplication for efficient weighted summation, 
        # replacing the element-wise multiplication and summation approach.
        weighted_sum = torch.matmul(self.pooling_masks, x_reshaped.t()).t()
        weighted_sum = weighted_sum.view(n, c, self.n_radii, self.n_angles)
        out = weighted_sum / self.pooling_mask_counts.view(1, 1, self.n_radii, self.n_angles)

        out = torch.nn.functional.pad(out, (1, 1, 0, 0), mode='reflect')
        out = self.output_transform(out)
        return out
