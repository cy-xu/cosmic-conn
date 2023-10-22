# -*- coding: utf-8 -*-

"""
Helper functions for numpy array and image manipulations.
CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np

from skimage import exposure, segmentation, io

from astropy.stats import sigma_clipped_stats

from cosmic_conn.cr_pipeline.zscale import zscale
from cosmic_conn.cr_pipeline.banzai_stats import (
    robust_standard_deviation,
    sigma_clipped_mean,
)


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def subtract_sky(ndarray, remove_negative=False):
    if remove_negative:
        # subtracted = np.where(subtracted < 0, 0, subtracted)
        ndarray[ndarray < 0.0] = 0.0

    if len(ndarray.shape) == 3:
        median = np.median(ndarray, axis=(1, 2), keepdims=True)
        ndarray -= median
    elif len(ndarray.shape) == 2:
        ndarray -= np.median(ndarray)

    return ndarray.astype("float32")


def variation_intensity(vector):
    # modified from coefficient of variation
    # this function is better at spotting sudden bright pixel in a vector (CR)
    # https://en.wikipedia.org/wiki/Coefficient_of_variation

    # subtract negative numbers, minimize standard deviation
    if vector.min() < 0:
        vector -= vector.min()

    eps = 1
    median = np.median(vector)

    variation_intensity = (np.max(vector) - median) / (median + eps)

    return variation_intensity


def sigma_clip(ndarray, sigma=0.0, clip_source=False):
    # clamp extreme outlier pixels for better stats
    if sigma == 0.0:
        return ndarray

    else:
        mean, median, std = sigma_clipped_stats(ndarray, sigma=sigma)

        if clip_source:
            # clip extreme bright pixels
            ceil = mean + sigma * std
            ndarray[ndarray > ceil] = ceil

        # clip extreme low (negative) pixels
        floor = mean - sigma * std
        ndarray[ndarray < floor] = median

        return ndarray


def extract_CR(frames, median_frm, cr_mask, root, filenames):
    # extract actual CR values and output masks for data augmentation

    # frame is aleady sky subtracted
    cr_frames = frames * cr_mask

    # save all cr frames as npy
    out_dir = os.path.join(root, "cr_frames")
    mkdir(out_dir)
    for i in range(len(cr_frames)):
        fname = os.path.join(out_dir, filenames[i] + "_cr")
        np.savez_compressed(fname, cr_frames[i], allow_pickle=True)

    # export cleaned frames for png preview
    clean_frames = np.where(cr_mask > 0, median_frm, frames)

    return clean_frames, cr_frames


def replace_boundary(array, boundary, replace_value=0):
    array[0:boundary, :] = replace_value
    array[-boundary:, :] = replace_value
    array[:, 0:boundary] = replace_value
    array[:, -boundary:] = replace_value
    return array


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


def acquire_valid_area(valid_masks):
    # reprojection shifted/rotated frames in a sequence
    # leveing black pixels near edges, valid_masks are binary
    valid_mask = np.ones_like(valid_masks[0])

    for i in range(valid_masks.shape[0]):
        valid_mask *= valid_masks[i]

    assert valid_mask.max() <= 1.0, "valid mask max over 1."

    return valid_mask.astype("uint8")


def center_crop_npy(array, crop_size=0, crop_ratio=1):
    # center cropping a 2D array
    h, w = array.shape
    if crop_size == 0 and crop_ratio == 1:
        return array

    if crop_size > 0:
        if h < crop_size or w < crop_size:
            print(f"array smaller than crop size")
            return array
        else:
            left, top = int(w / 2 - crop_size / 2), int(h / 2 - crop_size / 2)
            return array[top: top + crop_size, left: left + crop_size]

    if 0 < crop_ratio < 1:
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        w_start, h_start = int((w - new_w) / 2), int((h - new_h) / 2)
        return array[h_start: h_start + new_h, w_start: w_start + new_w]


def median_cut_off(img, n_std=3.0):
    img -= np.median(img)
    upper_bound = sigma_clipped_mean(img, sigma=3.5) + n_std * np.std(img)

    img = np.where(img < 0, 0, img)
    img = np.where(img > upper_bound, upper_bound, img)

    # linear normalize to increase contrast
    return img


def histogram_equalization(img, adaptive=False):
    img -= np.median(img) + 2 * robust_standard_deviation(img)
    img = np.where(img < 0, 0, img)

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    if adaptive:
        # Adaptive Equalization
        img_eq = exposure.equalize_adapthist(img_rescale, clip_limit=0.03)
    else:
        # Equalization
        img_eq = exposure.equalize_hist(img_rescale)

    # scale to uint8 data range and data type
    img_uint8 = scale2uint8(img_eq)

    return img_uint8


def zscale_cutoff(array, z1z2=None, nsamples=600, contrast=0.25):
    # 600 samples, contrast 0.25, same as DS9

    if array.max() < 250:
        return array, z1z2

    if z1z2 is None:
        z1, z2 = zscale(array, nsamples, contrast)
        z1z2 = [z1, z2]
    else:
        z1, z2 = z1z2

    # array = np.where(array < z1, z1, array)
    # array = np.where(array > z2, z2, array)
    array[array < z1] = z1
    array[array > z2] = z2
    array -= array.min()

    return array, z1z2


def zscale_stack_cutoff(noisy_stack, nsamples=10_000, contrast=0.1, cut_offs=None):
    if cut_offs is None:
        # normalize all noisy frames in a stack to same level
        cut_offs = [zscale(nsy, nsamples, contrast) for nsy in noisy_stack]

    z1 = min([c[0] for c in cut_offs])
    z2 = max([c[1] for c in cut_offs])

    noisy_vis = np.clip(noisy_stack, z1, z2)

    return noisy_vis, cut_offs


def scale2uint8(array):
    array_new = array - array.min()
    assert (
        array_new.max() != 0
    ), f"zero division when normalizing image, array max == 0 {array.shape}"

    array_new /= array_new.max()
    array_new = (array_new * 255).astype("uint8")
    return array_new


def save_as_png(
    array,
    dir,
    filename,
    file_type="uint8",
    zscale=False,
    z1z2=None,
    gamma=False,
    equalization=False,
    grid=False,
):

    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # array = array.astype("int") if "scrappy" in filename else array
    array = array * 255.0 if "mask" in file_type else array

    hdr = False
    for key in ["noisy", "clean", "cleaned", "median"]:
        if key in file_type:
            hdr = True

    if hdr:
        if zscale:
            # 600 samples, contrast 0.25, same as DS9
            array, z1z2 = zscale_cutoff(
                array, z1z2, nsamples=600, contrast=0.25)

        if gamma:
            array -= array.min()
            # STIFF default gamma 2.2
            array = exposure.adjust_gamma(array, 2.2)

        if equalization:
            file_type += "_equal_"
            array = histogram_equalization(array, adaptive=False)

        if not (zscale or gamma or equalization):
            array = median_cut_off(array, n_std=2)

    # scale to uint8 data range and data type
    array = scale2uint8(array)

    if grid:
        draw_grid(array, interval=100, grid_color=32)

    # save as uint8 image
    mkdir(dir)
    filename = filename + "_" + file_type + ".png"
    path = os.path.join(dir, filename)
    io.imsave(path, array, check_contrast=False)

    return filename, z1z2


def draw_grid(array, interval=100, grid_color=1):

    # Modify the image to include the grid
    array[:, interval::interval] = grid_color
    array[interval::interval, :] = grid_color

    return array
