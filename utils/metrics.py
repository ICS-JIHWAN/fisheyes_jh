import numpy as np
import torch

from dataset.dataloader import SEG_COLOR


def gaussuian_filter(kernel_size, sigma=1, mu=0):
    # Generating 2D grids 'x' and 'y' using meshgrid with 10 evenly spaced points from -1 to 1
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))

    # Calculating the Euclidean distance 'd' from the origin using the generated grids 'x' and 'y'
    d = np.sqrt(x * x + y * y)

    # Calculating the Gaussian-like distribution 'g' based on the distance 'd', sigma, and mu
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    return g


def create_color_map(gt_map):
    arg_gt = torch.argmax(gt_map, dim=1, keepdim=True).cpu()
    arg_gt = arg_gt.repeat(1, 3, 1, 1)
    B, C, H, W = arg_gt.shape

    color_map = torch.zeros_like(arg_gt)
    color_map = color_map.type_as(arg_gt).cpu()
    for k in SEG_COLOR.keys():
        seg = torch.tensor(SEG_COLOR[k]).unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, H, W).repeat(B, 1, 1, 1)
        color_map = torch.where(arg_gt == k, seg, color_map)

    return color_map


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='border'):
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x).to(flow.device)

    vgrid = grid.repeat(B, 1, 1, 1)
    # vgrid = grid + flow
    # vgrid = grid * flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0

    flow = flow.squeeze(3)
    vgrid_x = vgrid_x + flow[:, :, :, 0]
    vgrid_y = vgrid_y + flow[:, :, :, 1]
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = torch.nn.functional.grid_sample(x.to(flow.device), vgrid_scaled, mode=interp_mode, align_corners=True,
                                             padding_mode=padding_mode)
    return output
