"""
pytorch tensor based utilities.
CY Xu (cxu@ucsb.edu)
"""
import os
import math
import logging
import datetime
import random
import psutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmic_conn.cr_pipeline.banzai_stats import robust_standard_deviation


def tensor2np(gpu_tensor):
    # convert torch tensor to numpy array
    if isinstance(gpu_tensor, torch.Tensor):
        return gpu_tensor.detach().cpu().numpy()
    else:
        return gpu_tensor


def remove_nan(image):
    # replace NaN with 0.0 if exist
    if np.sum(np.isnan(image)) > 0:
        image = np.nan_to_num(image, copy=False, nan=0.0)
    return image


def memory_check(device):
    # check if the machien/GPU has sufficient memory for
    # full image detection without movign to swap
    # otherwise the image is sliced into stamps

    CPU_THRESHOLD = 16 * (1024**3)  # 16 GB free memory
    GPU_THRESHOLD = 8 * (1024**3)  # 8 GB free memory

    if str(device) == 'cpu':
        # available memory on CPU is not reliable, defult to 1024 stamps
        # available_memory = psutil.virtual_memory()[1]
        # full_image_detection = available_memory > CPU_THRESHOLD
        full_image_detection = False

    else:
        # GPU available memory
        t = torch.cuda.get_device_properties(device).total_memory
        # r = torch.cuda.memory_reserved(device)
        # a = torch.cuda.memory_allocated(device)
        # available_memory = r-a  # free inside reserved
        full_image_detection = t > GPU_THRESHOLD

    if not full_image_detection:
        msg = f"...available memory not sufficient for whole image detection."
        msg2 = f"...image will be sliced into stamps."
        logging.warning(msg)
        logging.warning(msg2)

    return full_image_detection


# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
def DiceLoss(prediction, targets, smooth=1):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # inputs = F.sigmoid(inputs)

    # prediction = median_weight * prediction

    # flatten label and prediction tensors
    prediction = prediction.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (prediction * targets).sum()
    dice = (2.0 * intersection + smooth) / \
        (prediction.sum() + targets.sum() + smooth)

    return 1 - dice


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.0
    variance = sigma ** 2.0

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        groups=channels,
        bias=False,
    )

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    if torch.cuda.is_available():
        gaussian_filter.to(torch.device("cuda"))

    return gaussian_filter


# permanent
gaussian_filter = get_gaussian_kernel(kernel_size=5, sigma=2, channels=1)

# https://discuss.pytorch.org/t/solved-class-weight-for-bceloss/3114/2


def median_weighted_bce(p, y, median_stack=None, imbalance_alpha=1.0):
    eps = 1e-7

    # prevent gradient diminishing or explosion
    p = torch.clamp(p, eps, (1 - eps))

    if median_stack is None:
        loss = y * torch.log(p) + \
            (1 - y) * torch.log(1 - p)
    else:
        assert median_stack.shape == p.shape

        median_weighted_mask = torch.zeros_like(median_stack)

        if imbalance_alpha < 1.0:
            with torch.no_grad():

                # batch size for loop
                for i in range(median_stack.shape[0]):

                    median_curr = median_stack[i: i + 1]

                    robust_std = robust_standard_deviation(
                        tensor2np(median_curr.squeeze())
                    ) + eps

                    median_curr -= torch.median(median_curr)

                    floor = max(0, robust_std)
                    ceil = min(median_curr.max().item(), 5 * robust_std)

                    mask_curr = torch.clamp(median_curr, floor, ceil)
                    mask_curr -= floor

                    mask_curr = gaussian_filter(mask_curr)

                    # avoid zero division
                    if mask_curr.max() > 0:
                        mask_curr /= mask_curr.max()

                    mask_curr = torch.clamp(mask_curr, imbalance_alpha, 1.0)

                    # populatet the processed mask to a stack
                    median_weighted_mask[i] = mask_curr

        else:
            median_weighted_mask = 1.0

        p = p.contiguous()
        y = y.contiguous()

        # median weighted BCE
        loss = (y * torch.log(p)) + \
            median_weighted_mask * ((1 - y) * torch.log(1 - p))

    return torch.neg(torch.mean(loss))


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction="none"
            )
        else:
            BCE_loss = F.binary_cross_entropy(
                inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


def replace_boundary(array, boundary, replace_value=0):
    array[0:boundary, :] = replace_value
    array[-boundary:, :] = replace_value
    array[:, 0:boundary] = replace_value
    array[:, -boundary:] = replace_value
    return array


def erase_boundary_tensor(tensor, boundary_width=0):
    in_shape = tensor.shape

    # replace boundary with 0 to avoid occasional issues on CCD boundary
    if boundary_width == 0:
        return tensor

    if boundary_width > 0:
        reshaped = tensor.view(-1, in_shape[-2], in_shape[-1])

        for i in range(reshaped.shape[0]):
            reshaped[i] = replace_boundary(reshaped[i], boundary_width)

    tensor = reshaped.view(in_shape)

    return tensor


def erase_boundary_np(ndarray, boundary_width=0, replace_value=0):
    # replace boundary with 0 to avoid occasional issues on CCD boundary
    if boundary_width == 0:
        return ndarray

    if boundary_width > 0:
        copy = ndarray.copy()

        if len(copy.shape) == 2:
            copy = replace_boundary(copy, boundary_width, replace_value)
            return copy

        elif len(copy.shape) == 3:
            for i in range(copy.shape[0]):
                copy[i] = replace_boundary(
                    copy[i], boundary_width, replace_value)
            return copy


def clean_large(image, model, patch=1024, overlap=0, ret_numpy=False):
    """
        modified from deepCR implementation
        given input image
        return cosmic ray mask and (optionally) clean image
        mask could be binary or probabilistic
    :param img0: (np.ndarray) 2D input image
    :return: mask or binary mask; or None if internal call
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image).float()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    image = image.to(device)

    shape = image.shape
    stride = patch - 2 * overlap
    hh = math.ceil(shape[0] / stride)
    ww = math.ceil(shape[1] / stride)
    mask = torch.zeros_like(image)

    for i in range(hh):
        for j in range(ww):
            # overlapping crop at stride
            h_start = i * stride
            w_start = j * stride
            h_stop = min(h_start + patch, shape[0])
            w_stop = min(w_start + patch, shape[1])

            crop = image[h_start:h_stop, w_start:w_stop]
            crop_h, crop_w = crop.shape

            # sky subtraction
            crop -= crop.median()

            if crop_h < patch or crop_w < patch:
                crop = modulus_boundary_crop(crop)

            if 0 in crop.shape:
                continue

            # expand dimension to fit the model
            pdt = model(dim_expand(crop))
            pdt = pdt.squeeze()

            h_pad = min(1, i) * overlap
            w_pad = min(1, j) * overlap

            h_start_ = h_pad + h_start
            w_start_ = w_pad + w_start

            effective_pdt = pdt[h_pad:, w_pad:]

            h_stop_ = h_start_ + effective_pdt.shape[0]
            w_stop_ = w_start_ + effective_pdt.shape[1]

            mask[h_start_:h_stop_, w_start_:w_stop_] = effective_pdt

    if ret_numpy:
        mask = tensor2np(mask)

    return mask


def center_crop(tensor, crop_size=0):
    # center cropping for efficiency
    if crop_size == 0:
        return tensor

    h, w = tensor.shape

    if h < crop_size or w < crop_size:
        return tensor

    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    h_stop = min(h_offset + crop_size, h)
    w_stop = min(w_offset + crop_size, w)

    return tensor[h_offset:h_stop, w_offset:w_stop]


def center_crop_npy(array_in, crop_size=0):
    # center crop a quare img from 2D array
    if crop_size == 0:
        return array_in
    else:
        h, w = array_in.shape
        assert h > crop_size and w > crop_size, ValueError(
            f"array smaller than crop size"
        )

        cropped = np.zeros((crop_size, crop_size))

        left, top = int(w / 2 - crop_size / 2), int(h / 2 - crop_size / 2)
        cropped = array_in[top: top + crop_size, left: left + crop_size]

        return cropped


def modulus_boundary_crop(array, modulo=16):
    """
    crop a largest possible area of N modulo
    to simply the problem, start from top-left corner to the max valid area
    expect input as a 2D array
    """
    h, w = array.shape[0], array.shape[1]
    new_h = h - (h % modulo)
    new_w = w - (w % modulo)

    if new_h == h and new_w == w:
        return array

    new_array = array[:new_h, :new_w]

    return new_array


def dim_expand(tensor):
    if len(tensor.shape) == 4:
        return tensor

    elif len(tensor.shape) == 2:
        return tensor.unsqueeze(dim=0).unsqueeze(dim=0)

    elif len(tensor.shape) == 3:
        return tensor.unsqueeze(dim=0)

    else:
        raise ValueError(
            f"tensor shape {tensor.shape} is not valid, \
            array shape [n, 1, h, w] is expected."
        )


# PyTroch version
# https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
SMOOTH = 1e-6


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(
        (1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0

    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded


def _worker_init_fn(worker_id: int) -> None:
    """Sets a unique but deterministic random seed for background workers.

    Only sets the seed for NumPy because PyTorch and Python's own RNGs
    take care of reseeding on their own.
    See https://github.com/numpy/numpy/issues/9650."""
    # Modulo 2**32 because np.random.seed() only accepts values up to 2**32 - 1
    initial_seed = torch.initial_seed() % 2 ** 32
    worker_seed = initial_seed + worker_id
    np.random.seed(worker_seed)


def extract_random_frame(stacks):
    # randomly pick one frame from a sequence of three frames
    batch = stacks.shape[0]
    frames = stacks.shape[1]
    idx = random.randint(0, frames - 1)

    frames = [stacks[i, idx] for i in range(batch)]
    frames = torch.stack(frames).unsqueeze(dim=1)

    return frames


def subtract_sky(tensor, remove_negative=False):
    in_shape = tensor.shape

    tensor = tensor.view(-1, in_shape[-2], in_shape[-1])

    for i in range(tensor.shape[0]):
        tensor[i] -= tensor[i].median()

    if remove_negative:
        tensor = torch.clamp(tensor, 0.0, tensor.max())

    tensor = tensor.view(in_shape)

    return tensor


def tensor2uint8(gpu_tensor, dtype="uint8", method="clipping"):
    dims = len(gpu_tensor.shape)

    if dims == 3:
        array = tensor2np(gpu_tensor)
    elif dims == 2:
        array = tensor2np(gpu_tensor.unsqueeze(0))

    if method == "clipping":
        img = np.clip(array, 0, 255)
    elif method == "normalize":
        img = array - array.min()
        img = img / img.max()
        img *= 255

    return img.astype(dtype)


def remove_ext(filename):
    if filename.endswith(".fz"):
        return filename[:-8]
    elif filename.endswith(".fitz"):
        return filename[:-5]
    else:
        return os.path.splitext(filename)[0]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkCheckpointDir(suffix, continue_train):
    if continue_train:
        directory = continue_train
        path = f"./checkpoints/{continue_train}"

    else:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M")
        directory = str(timestamp) + "_" + suffix
        path = f"./checkpoints/{directory}"
        mkdir(path)

    return path, directory


EXTENSIONS = ["npz", "npy"]


def is_numpy_file(filename):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in EXTENSIONS)


def read_npy(directory):
    # read from multiple sub-dirs, output a sequential list of noisy images, masks and file names
    noisy_frms = []
    masks = []
    filenames = []

    for root, dirs, files in os.walk(directory):
        if "mask" in root and len(files) > 0:
            files.sort()
            for f in files:
                if f.endswith("mask.npy"):
                    filenames.append(f)
                    mask_path = os.path.join(root, f)
                    noisy_path = os.path.join(root, f.replace("mask", "noisy"))
                    try:
                        # if shape_n[0] < 2000 or shape_m[0] < 2000:
                        #     print(shape_m, shape_n)
                        #     print(f'{f} size smaller than sensor, file (not) removed')
                        # else:
                        noisy_frms.append(noisy_path)
                        masks.append(mask_path)
                    except:
                        print(
                            f"{f} is missing mask/noisy pair, file (not) removed")
                        # os.remove(path)
                        continue

    dataset_size = len(noisy_frms)
    assert dataset_size > 0, f"No valid training data found in {directory}"
    print(f"Training dataset size: {dataset_size}")

    return noisy_frms, masks, filenames


def gen_train_test_data(noisy_frms, masks, filenames, train_ratio=0.9):
    dataset_size = len(noisy_frms)
    train_size = int(dataset_size * train_ratio)

    train_idx = np.random.choice(
        dataset_size, train_size, replace=False).tolist()
    # complement of train_idx is the test idx
    test_idx = list(set(np.arange(dataset_size).tolist()) - set(train_idx))

    noisy_train = [noisy_frms[i] for i in train_idx]
    mask_train = [masks[i] for i in train_idx]
    file_train = [filenames[i] for i in train_idx]

    cropped_test = [noisy_frms[i] for i in test_idx]
    mask_test = [masks[i] for i in test_idx]
    file_test = [filenames[i] for i in test_idx]

    return noisy_train, mask_train, file_train, cropped_test, mask_test, file_test
