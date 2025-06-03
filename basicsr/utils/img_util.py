import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid

import torch


def cal_activation2(input_shape, offsets, kernel_size=3, padding=1, dilation=1, stride=1, group=4):

    offsets = torch.nan_to_num(offsets, nan=0.0, posinf=0.0, neginf=0.0)

    B, C, H, W = input_shape
    activation_counts = torch.zeros(B, 1, H, W, dtype=torch.int32).cuda()

    # 创建网格
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=-1).cuda()  # [H, W, 2]
    coords = coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # 应用每组的偏移量
    for group in range(4):
        group_offsets = offsets[:, group*18:(group+1)*18]  # 获取当前组的偏移量 [B, 18, H, W]
        group_offsets = group_offsets.reshape(B, 9, 2, H, W)  # [B, 9, 2, H, W]

        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                dy, dx = group_offsets[:, idx, 0], group_offsets[:, idx, 1]  # 偏移量Y, X [B, H, W]
            
                # 计算偏移后的坐标
                ny = (coords[..., 0] + dy).round().clamp(0, H-1).long()
                nx = (coords[..., 1] + dx).round().clamp(0, W-1).long()
            
                # 计算索引并且统计激活次数
                indices = (ny * W + nx).long()
                activation_counts += torch.bincount(indices.flatten(), minlength=B*H*W).view(B, 1, H, W)
    
    return activation_counts




def cal_activation(input_shape, offset, kernel_size=3, padding=1, dilation=1, stride=1, group=4):

    ##input_shape:(B,C,H,W)
    ##offset:(B,C,H,W)
    B, C, H, W = input_shape  # 批次大小, 通道数, 高度, 宽度
    
    activation_count = torch.zeros(B, 1, H, W).cuda()

    cc = kernel_size * kernel_size * 2

    # 遍历每个输出像素位置
    for i in range(-padding, H - padding):
        for j in range(-padding, W - padding):
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    # 计算每个位置对应的输入特征图的索引
                    input_i = i + ki * dilation
                    input_j = j + kj * dilation

                    # 确保索引在边界内
                    if 0 <= input_i < H and 0 <= input_j < W:
                        for gg in range(group):
                            for b in range(B):
                                offset_index = cc * gg + (ki * kernel_size + kj) * 2
                                di = torch.round(offset[b, offset_index, input_i, input_j])  # Y offset
                                dj = torch.round(offset[b, offset_index + 1, input_i, input_j])  # X offset

                                # 计算偏移后的位置
                                ni = input_i + di.long()
                                nj = input_j + dj.long()

                            
                                ni = torch.clamp(ni, 0, H - 1)
                                nj = torch.clamp(nj, 0, W - 1)
                       
                                activation_count[b, :, ni, nj] += 1
    return activation_count



def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
