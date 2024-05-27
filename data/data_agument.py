import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

SEG_COLOR = {
    0: (0, 0, 0),  # Road and background
    1: (255, 255, 0),  # Sidewalk
    2: (102, 102, 0),  # Construction
    3: (255, 0, 255),  # Fence
    4: (102, 0, 102),  # Pole
    5: (0, 255, 0),  # Traffic light
    6: (0, 0, 255),  # Traffic sign
    7: (102, 255, 102),  # Nature
    8: (204, 255, 255),  # Sky
    9: (255, 0, 0),  # Person
    10: (0, 102, 51),  # Rider
    11: (153, 0, 255)  # Car
}


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


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
        new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


def barrel_distortion_np(img, img_size):
    k1, k2 = 0.5, 0.2
    rows, cols = img.shape[:2]
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    mapx = 2 * mapx / (cols - 1) - 1
    mapy = 2 * mapy / (rows - 1) - 1

    r, theta = cv2.cartToPolar(mapx, mapy)
    ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4))

    mapx_, mapy_ = cv2.polarToCart(ru, theta)
    mapx_ = ((mapx_ + 1) * cols - 1) / 2
    mapy_ = ((mapy_ + 1) * rows - 1) / 2

    # Top index of distorted image
    # By finding the cross zero point
    top_y = np.where(np.logical_and(mapy_[:, int((cols - 1) / 2)] > -1, mapy_[:, int((cols - 1) / 2)] < 1))[
        0].item()
    left_x = np.where(np.logical_and(mapx_[int((rows - 1) / 2), :] > -1, mapx_[int((rows - 1) / 2), :] < 1))[
        0].item()

    img = cv2.remap(img, mapx_, mapy_, cv2.INTER_CUBIC)
    img = cv2.resize(
        img[top_y:rows - top_y, left_x:cols - left_x].copy(),
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )

    return img


def barrel_distortion_tensor(img, interp_mode='bilinear', padding_mode='zeros'):
    k1, k2 = 0.5, 0.2
    B, C, rows, cols = img.size()
    mapy, mapx = torch.meshgrid(torch.arange(0, rows), torch.arange(0, cols))

    grid = torch.stack((mapx, mapy), 2).float()  # W(x), H(y), 2
    grid = grid.repeat(B, 1, 1, 1)
    grid = grid.type_as(img).to(img.device)

    # scale grid to [-1,1]
    vgrid_x = 2.0 * grid[:, :, :, 0] / max(cols - 1, 1) - 1.0
    vgrid_y = 2.0 * grid[:, :, :, 1] / max(rows - 1, 1) - 1.0

    r, theta = cart2pol(vgrid_x, vgrid_y)
    ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4))


    mapx_, mapy_ = pol2cart(ru, theta)
    vgrid_scaled = torch.stack((mapx_, mapy_), dim=3)

    output = torch.nn.functional.grid_sample(
        img, vgrid_scaled, mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=True
    )

    mapx_ = ((mapx_ + 1) * cols - 1) / 2
    mapy_ = ((mapy_ + 1) * rows - 1) / 2

    top_y = torch.where(torch.logical_and(mapy_[:, :, int((cols - 1) / 2)] > -1, mapy_[:, :, int((cols - 1) / 2)] < 1))
    left_x = torch.where(torch.logical_and(mapx_[:, int((rows - 1) / 2), :] > -1, mapx_[:, int((rows - 1) / 2), :] < 1))
    output = output[:, :, top_y[1][0]:rows - top_y[1][0], left_x[1][0]:cols - left_x[1][0]]

    output = torch.nn.functional.interpolate(output, (rows, cols), mode='bilinear')
    return output


def distorted_img(input, output, cal_ru=False, device='cpu'):
    b, c, rows, cols = input.shape

    distorted_list = []
    for i in range(b):
        img = input[i, :3]
        img = np.array(img.permute(1, 2, 0).cpu().detach())

        mapy, mapx = np.indices((rows, cols), dtype=np.float32)
        mapx = 2 * mapx / (cols - 1) - 1
        mapy = 2 * mapy / (rows - 1) - 1

        r, theta = cv2.cartToPolar(mapx, mapy)

        if not cal_ru:
            # k1, k2 = round(output[0][0].item(), 3), round(output[0][1].item(), 3)
            k1, k2 = 0.5, 0.2
            ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4))
            mapx, mapy = cv2.polarToCart(ru, theta)
            mapx = ((mapx + 1) * cols - 1) / 2
            mapy = ((mapy + 1) * rows - 1) / 2

            distorted_list.append(
                torch.tensor(cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)).permute(2, 0, 1).unsqueeze(dim=0))
        else:
            ru = output[i]
            mapx, mapy = pol2cart(ru, torch.tensor(theta, dtype=torch.float32).to(device))
            distorted_list.append(
                torch.nn.functional.grid_sample(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device),
                                                torch.stack((mapx, mapy), dim=2).unsqueeze(0),
                                                mode='bilinear',
                                                align_corners=True))

    return torch.cat(distorted_list, dim=0)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='border'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        x (Ist' Tensor): size (N, C, H, W)
        flow (logits ru Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
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
    vgrid_x = vgrid_x * flow
    vgrid_y = vgrid_y * flow
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = torch.nn.functional.grid_sample(x.to(flow.device), vgrid_scaled, mode=interp_mode, align_corners=True, padding_mode=padding_mode)
    return output

    # -----
    # distorted = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    #
    # plt.subplot(2, num_img, i + 1)
    # plt.imshow(distorted)
    # plt.subplot(2, num_img, i+3)
    # plt.imshow(img)
    # -----


def cart2pol(x, y):
    b, h, w = x.size()
    divide = int(h / 2)
    torch.pi = torch.acos(torch.zeros(1)).item() * 2
    #
    theta = torch.zeros_like(x)
    # field of 1
    theta[:, divide:, divide:] = torch.atan(y[:, divide:, divide:] / x[:, divide:, divide:])
    # field of 2
    theta[:, divide:, :divide] = torch.pi + torch.atan(y[:, divide:, :divide] / x[:, divide:, :divide])
    # field of 3
    theta[:, :divide, :divide] = torch.pi + torch.atan(y[:, :divide, :divide] / x[:, :divide, :divide])
    # field of 4
    theta[:, :divide, divide:] = 2 * torch.pi + torch.atan(y[:, :divide, divide:] / x[:, :divide, divide:])
    #
    return (x ** 2 + y ** 2).sqrt(), theta


def pol2cart(ru, theta):
    x = ru * torch.cos(theta)
    y = ru * torch.sin(theta)

    return x, y

if __name__ == '__main__':
    img_path = '/storage/jhchoi/fish_eyes/train_source_image/TRAIN_SOURCE_0000.png'
    # device = torch.device('cuda:{}'.format(1)) if True is not None else torch.device('cpu')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(
        img,
        (512, 512),
        interpolation=cv2.INTER_AREA,
    )
    img = torch.tensor(img.transpose((2, 0, 1)) / 255.).unsqueeze(dim=0).to('cuda:1')
    img = torch.cat([img, img, img], dim=0)
    # img_np = barrel_distortion_np(ismg, 512)
    img_tensor = barrel_distortion_tensor(img)
