# -*- coding: utf-8 -*-

"""
Helper functions for numpy array and image manipulations.
CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np

from skimage import exposure, morphology, segmentation, io
from scipy import ndimage, signal

import sep
from astropy.table import Table
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


def prune_nans_from_table(table):
    """
    From banzai
    https://github.com/LCOGT/banzai/blob/master/banzai/utils/array_utils.py
    """
    nan_in_row = np.zeros(len(table), dtype=bool)
    for col in table.colnames:
        nan_in_row |= np.isnan(table[col])
    return table[~nan_in_row]


def sep_source_mask(frames, valid_mask):
    """
    Following Banzai's SEP procedure with some modifications
    https://github.com/LCOGT/banzai/blob/master/banzai/photometry.py
    SEP manual
    https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf

    1. Extrat sources from the median frame, more reliable than SEP on individual frames
    2. Calculate each soruce's SEP stats, ellipse, window size
    3. Expand sources' rectangle window, acquire a local background mask
    4. Local background mask shuold be very close across frames, except for extreme PSF wings
    5. Calculate local background stats for each source and extract sources' mask from background
    6. Generate a source mask for each frame
    """

    # lower threshold as we work on the median frame
    threshold = 3.5
    min_area = 9

    # prep median frame for SEP source extractor
    median_frame = np.median(frames, axis=0)

    # increase limit as we work on the median frame
    ny, nx = median_frame.shape
    sep.set_extract_pixstack(int(nx * ny * 0.10))

    # subtract background
    bkg = sep.Background(median_frame, bw=32, bh=32, fw=3, fh=3)
    bkg_img = bkg.back()
    median_frame -= bkg_img

    # SEP applies a gaussian blur kernel
    # [[1,2,1], [2,4,2], [1,2,1]] by default
    try:
        # corner case when active sources pixel exceeds limit will crash
        sources = sep.extract(
            median_frame, threshold, err=bkg.globalrms, minarea=min_area
        )
    except:
        return None

    # Convert the detections into a table
    sources = Table(sources)

    if len(sources) > 3000:
        return None

    # We remove anything with a detection flag >= 8
    # This includes memory overflows and objects that are too close the edge
    sources = sources[sources["flag"] < 8]

    sources = prune_nans_from_table(sources)

    # Calculate the ellipticity and elongation
    sources["ellipticity"] = 1.0 - (sources["b"] / sources["a"])
    sources["elongation"] = sources["a"] / sources["b"]

    # Fix any value of theta that are invalid due to floating point rounding
    # -pi / 2 < theta < pi / 2
    sources["theta"][sources["theta"] > (np.pi / 2.0)] -= np.pi
    sources["theta"][sources["theta"] < (-np.pi / 2.0)] += np.pi

    # Calculate the FWHMs of the stars:
    fwhm = 2.0 * (np.log(2) * (sources["a"]
                               ** 2.0 + sources["b"] ** 2.0)) ** 0.5
    sources["fwhm"] = fwhm

    # small fwhm are often bad pixels/dust on CCD (consistent across frames)
    sources = sources[fwhm > 1.0]

    """
    Generate local mask for each source in each frame
    """
    sources_masks = []

    img_height = median_frame.shape[0]
    img_width = median_frame.shape[1]

    for frame in frames:
        sources_mask = np.zeros_like(median_frame, dtype="uint8")

        for i in range(len(sources)):
            src = sources[i]

            src_ellipse = np.zeros_like(median_frame, dtype="uint8")

            # soruce's ellipse based on the median frame
            sep.mask_ellipse(
                src_ellipse, src["x"], src["y"], src["a"], src["b"], src["theta"], r=2
            )

            # expand source window by 1.5x so more sky and PSF wings could be included
            x_expand = int((src["xmax"] - src["xmin"]) * 0.25)
            y_expand = int((src["ymax"] - src["ymin"]) * 0.25)

            # min, max to handle boundary cases
            ymin = max(0, src["ymin"] - y_expand)
            ymax = min(src["ymax"] + y_expand, img_height)
            xmin = max(0, src["xmin"] - x_expand)
            xmax = min(src["xmax"] + x_expand, img_width)

            # fail safe #0, if source too close to valid_mask boundary, pass
            # we got valid mask from reprojecting consecutive frames to same coordinates
            # so some frame will have missing information near boundary
            # valid_count = np.sum(valid_mask[ymin:ymax, xmin:xmax] == 0)
            # if valid_count > 0:
            #     continue

            src_window = np.zeros_like(median_frame, dtype="uint8")
            src_window[ymin:ymax, xmin:xmax] = 1

            # remove soruce ellipse from source window mask
            bkg_mask = src_window * (1 - src_ellipse)

            # fail safe #1, if ellipse larger than window, use ellipse directly
            if bkg_mask.sum() == 0.0:
                sources_mask[src_ellipse > 0] = 1
                continue

            # window crop background pixels from each frame
            bkg_frm = bkg_mask * frame

            # shift background and source pxiels values
            # valid bkg area will be above zero, all other area == 0
            zero_offset = np.min(bkg_frm) - 1
            src_frm = (frame - zero_offset) * src_window
            bkg_frm = (bkg_frm - zero_offset) * bkg_mask

            # if effective background usable, get local background stats
            # consider Lucy smoothing
            bkg_vec = bkg_frm[ymin:ymax, xmin:xmax].flatten()
            bkg_vec = bkg_vec[bkg_vec != 0]

            # use 2 sigma above local background median as the spread boundary
            # robust standard deviation ensures to reject PFS wings from stats
            # 2 sigma widen the source's extent slightly but includes more soft contour
            bkg_std = robust_standard_deviation(bkg_vec)
            bkg_upperbound = np.median(bkg_vec) + 2.0 * bkg_std

            src_peak_value = np.max(src_frm)
            # src_peak_value = max(src_peak, np.max(src_frm))

            # fail safe #2, if background upperbound higher than source peak
            if bkg_upperbound >= src_peak_value:
                sources_mask[src_ellipse > 0] = 1
                continue

            # effective pixel values range for the source
            tolerance = src_peak_value - bkg_upperbound

            # starting point for floodfill, from true peak pixel
            seed_point = np.unravel_index(src_frm.argmax(), src_frm.shape)
            # seed_point = (src['ypeak'], src['xpeak'])

            # flood filling is better than thresholding
            # as the mask expands from source center, CR in the source window won't be flagged
            src_mask = segmentation.flood(
                src_frm, seed_point, tolerance=tolerance
            ).astype("uint8")

            sources_mask[src_ellipse > 0] = 1
            sources_mask[src_mask > 0] = 1

        sources_masks.append(sources_mask)

    sources_masks = np.stack(sources_masks)

    # dilate CR by n pixel to avoid introducing artificl sharp edges
    struct = ndimage.generate_binary_structure(2, 2)
    sources_dilation = np.zeros_like(sources_masks)

    for i in range(sources_masks.shape[0]):
        sources_dilation[i] = ndimage.binary_dilation(
            sources_masks[i], structure=struct, iterations=2
        )

    return sources_dilation


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
