# -*- coding: utf-8 -*-

"""
Main file to define the CR labeling pipelien for LCO imaging data
CY Xu (cxu@ucsb.edu)
"""

import time
import numpy as np

from skimage import morphology
from skimage.segmentation import flood_fill
from scipy import ndimage

from cosmic_conn.cr_pipeline.utils_img import (
    sigma_clip,
    subtract_sky,
    erase_boundary_np,
)
from cosmic_conn.cr_pipeline.utils_img import sep_source_mask, center_crop_npy
from cosmic_conn.cr_pipeline.utils_io import (
    hdul_to_array,
    save_fits_with_CR,
    save_as_png,
    save_array_png,
)
from cosmic_conn.cr_pipeline.utils_io import remove_outlier_frm
from cosmic_conn.cr_pipeline.banzai_stats import median_absolute_deviation


class CR_reduction:
    def __init__(self, hdul=None, opt=None):
        self.hdul = hdul
        self.opt = opt
        self.reject_dir = opt.root + "/rejected_frms"
        self.pre_process()

    def pre_process(self):
        header = self.hdul["SCI"].header

        self.fname = header["filename"]
        self.exp_time = header["exptime"]
        self.frmtotal = header["frmtotal"]
        self.min_cr_size = self.opt.min_cr_size
        rdnoise = header["rdnoise"]

        # read hdul as ndarrays of shape N,H,W
        self.frames_banzai, self.valid_mask, gains, self.filenames = hdul_to_array(
            self.hdul
        )

        # pre-processing, a separate copy for mask generation
        frames = sigma_clip(self.frames_banzai.copy(),
                            sigma=3.0, clip_source=False)

        frames_median_subtracted = subtract_sky(frames.copy())
        self.source_masks = sep_source_mask(
            frames_median_subtracted, self.valid_mask)

        # re-calculate gain for each 0m4 exposure with source mask
        if "0m4" in self.fname and self.source_masks is not None:
            # sovle for the gain for each image
            self.new_gains = self.sovle_for_gain(
                frames, rdnoise, self.source_masks)

            # for 0m4 data, update image and read_noise with new gain
            self.frames = frames * self.new_gains
            self.rdnoises = rdnoise * self.new_gains

            with open(self.opt.log_file, "a") as log:
                msg = f"{self.fname}, new gains {list(self.new_gains.ravel())}\n\n"
                log.write(msg)
                print(msg)
        else:
            self.frames = frames
            self.rdnoises = rdnoise
            self.new_gains = np.array(gains).reshape(frames.shape[0], 1, 1)

        # remove no_data pixels, potential bad boundary CCD pixels (100px),
        # and potential bad SEP analysis near the boundary, included in ignore mask
        self.ignore_boundary = 100

        self.frames *= self.valid_mask
        self.frames = erase_boundary_np(
            self.frames, boundary_width=self.ignore_boundary
        )

    def generate_CR_masks(self):
        """
        This is the main CR detection loop
            estimate_reject_refine() is called one or multiple times for CR detection
            detect CR in the sequence of frames, reject outlier pixels and frames

        if estimate_reject_refine() returns a string, it's a error message during detection
        print the error message and abort this sequence
        """
        tic = time.perf_counter()
        high_thres = self.opt.snr_thres
        low_thres = self.opt.snr_thres_low

        # return error message if source mask not valid
        if self.source_masks is None:
            return "rejected_SEP_sources_over_3k"

        # initial estimation to reject outlier frame
        cr_frac = self.estimate_reject_refine(high_thres, low_thres)

        if isinstance(cr_frac, str):
            self.output_png_preview(reason=cr_frac)
            return cr_frac

        # if an outlier frame is rejected, detect again with the new sequence because the
        # CR frac between frames might change, until all frames considered well aligned
        while len(self.reject_idx) > 0:
            cr_frac = self.estimate_reject_refine(high_thres, low_thres)
            if isinstance(cr_frac, str):
                return cr_frac

        # input frames exposure time capped at 100s, no frame should have CR less than 1e-5
        if cr_frac < 1e-5:
            self.output_png_preview(reason="rejected_low_cr_frac")
            return f"CR frac too low {cr_frac}, exposure time {self.exp_time}"

        # all rejection passed, print stats of the useable frames
        with open(self.opt.log_file, "a") as log:
            msg = (
                f"{self.fname}, exp_time {self.exp_time}\n"
                f"CR count {[np.sum(m) for m in self.cr_mask]}, CR fraction {round(cr_frac, 6)}\n\n"
            )
            log.write(msg)
            print(msg)

        debug_frames = self.mask_high, self.mask_low, self.snr_deviation

        # if everythings checks out, output new fits with masks
        save_fits_with_CR(
            self.opt,
            self.hdul,
            self.new_gains,
            self.cr_mask,
            self.ignore_mask,
            self.filenames,
            debug_frames,
        )

        # replace CR pixel with median frame and output png for quick preview
        self.clean_frames = np.where(
            self.cr_mask > 0, self.median_frame, self.frames)

        # output png preview or numpy as training set (no longer using numpy)
        if not self.opt.no_png:
            self.output_png_preview(self.opt.comment)

    def estimate_reject_refine(self, high_thres, low_thres, reject_frm=True):
        """
        The main CR detection happens here:
            1. estimate two masks based on an initial SNR threshold and
               a lower threshold (for peripheral CR edges)

            2. refine CR to incldue soft edges via dilation/flood filling

            3. generate the ignore mask, definition:
                OK (0)
                boundary (1)
                no data (2)
                sources (4)
                hot pixels (8)

            4. multiply CR mask with ignore mask again

            5. reject frame outliers with excessive CR in a sequence
               often false positives caused by satellite trails or mis-aligned frames 
        """

        # 1.
        mask_low, mask_high = self.estimate_two_masks(
            self.frames, self.rdnoises, high_thres, low_thres
        )
        # from here on, self.frames are sky subtracted

        # 2.
        self.cr_mask, single_pixels = self.maximize_cr_edges(
            mask_low, mask_high, self.min_cr_size
        )

        # return error message
        if isinstance(self.cr_mask, str):
            return self.cr_mask

        # 3. ignore mask include sources, single hot pixels, and empty boundary
        self.ignore_mask = erase_boundary_np(
            np.zeros_like(self.frames, dtype="uint8"),
            boundary_width=self.ignore_boundary,
            replace_value=1,
        )
        self.ignore_mask += (1 - self.valid_mask) * 2
        self.ignore_mask += self.source_masks * 4
        self.ignore_mask += single_pixels * 8

        # 4.
        self.cr_mask *= (self.ignore_mask == 0).astype("uint8")

        # 5.
        if reject_frm:
            self.reject_idx = self.reject_outlier_frm()
        else:
            self.reject_idx = []

        # return error message
        if self.frmtotal < 3:
            return "rejected_less_than_3_frames"

        # calculate CR fraction/ratio, used for adaptive SNR threshold
        total_pixels = self.cr_mask.size
        cr_frac = np.sum(self.cr_mask) / total_pixels

        return cr_frac

    def compute_SNR(self, frames, read_noise):
        # assuming frames in an N_images x Ny x Nx array
        N_frames = frames.shape[0]

        bkg_noise = np.median(frames, axis=(1, 2), keepdims=True)

        # calculate uncertainties
        frame_uncertainty = np.sqrt(
            read_noise ** 2 + np.abs(frames) + bkg_noise)

        median_frame_uncertainty = np.median(frame_uncertainty, axis=0) / np.sqrt(
            N_frames
        )

        # calculate the total noise including our model (approximately for the median)
        total_uncertainty = np.sqrt(
            frame_uncertainty ** 2 + median_frame_uncertainty ** 2
        )

        # subtract sky and get correct median
        frames = subtract_sky(frames, remove_negative=False)
        self.median_frame = np.median(frames, axis=0)

        # The median acts as our model of what the frame should look like,
        # so we can look for deviations from that
        # without np.abs(frames), only consider positive deviations
        snr_deviation = (frames - self.median_frame) / total_uncertainty

        return snr_deviation

    def estimate_two_masks(self, frames, read_noise, high_thres, low_thres):
        """
        noisy_stack: ndarray
            a stack of noisy frames of shape (n, h, w)

        Rerturn:
            mask_low, retrieved from a lower threshold, more pixel detected but more false positive as well
            mask_high, retrieved from a higher threshold to reject false positive
        """

        snr_deviation = self.compute_SNR(frames, read_noise)

        mask_high = (snr_deviation > high_thres).astype("uint8")

        # lower threshold to get the soft edges of CR
        mask_low = (snr_deviation > low_thres).astype("uint8")

        # for debugging
        self.snr_deviation = snr_deviation
        self.mask_high, self.mask_low = mask_high, mask_low

        return mask_low, mask_high

    def sovle_for_gain(self, frames, rdnoise, source_mask):
        # if no vlaid source masks, return 1 for now, this sequence will be skipped
        if source_mask is None:
            return np.array(np.ones((frames.shape[0], 1, 1)))

        # ignore sources and hot pixels
        bkg_imgs = frames * (1 - source_mask)
        new_gains = []

        for i in range(frames.shape[0]):
            # crop 100px boundary
            bkg = center_crop_npy(bkg_imgs[i], crop_ratio=0.9)

            mad = median_absolute_deviation(bkg)
            median = np.median(bkg)

            # solve for (mad * 1.48)**2 - gain * median - gain**2 * read_noise ** 2 = 0
            coeff = [-1 * rdnoise ** 2, -1 * median, (mad * 1.48) ** 2]
            gain = np.roots(coeff)[1]
            new_gains.append(round(gain, 6))

        return np.array(new_gains).reshape(-1, 1, 1)

    def maximize_cr_edges(self, mask_low, mask_high, min_cr_size):
        shape = mask_high.shape

        # reject CR candidates over-lapping with sources
        mask_high = mask_high * (1 - self.source_masks)

        if self.opt.flood_fill:
            # Flood fill captures less false positive isolated pixels
            # but it also misses small true positive CRs
            mask_common = np.logical_and(mask_high, mask_low)

            for i in range(shape[0]):
                # create labels in mask_high
                imglab, total = morphology.label(
                    mask_common[i].astype("int"),
                    background=0,
                    return_num=True,
                    connectivity=2,
                )

                # any pixle insdie idx can be the starting point for flood fill
                for j in range(1, total + 1, 1):
                    index = np.where(imglab == j)
                    idx = (index[0][0], index[1][0])
                    mask_low[i] = flood_fill(mask_low[i], idx, 2)

            cr_mask = mask_low == 2

        else:
            # dilation expands CR edges by N pixels, more conservative than flood fill
            # isolated false positive pixels are included in the ignroe mask
            struct = ndimage.generate_binary_structure(2, 2)
            mask_dilation = np.zeros_like(mask_high)

            for i in range(shape[0]):
                mask_dilation[i] = ndimage.binary_dilation(
                    mask_high[i], structure=struct, iterations=self.opt.dilation
                )

            cr_mask = np.logical_and(mask_dilation, mask_low)

        # second pass rejection in case source included after dialtion
        cr_mask = cr_mask * (1 - self.source_masks)

        # save before removing single pixels to add them to ignore mask
        cr_mask = cr_mask.astype("bool")
        cr_mask_single_pixels = cr_mask.copy()

        # remove flagged CR smaller than certain pixels from final mask
        if min_cr_size > 1:
            for i in range(shape[0]):
                # create labels in segmented image
                cr_mask[i] = morphology.remove_small_objects(
                    cr_mask[i], min_size=min_cr_size, connectivity=2
                )

        # add single pixels and source mask to make the ignore mask
        single_pixels = np.logical_xor(
            cr_mask_single_pixels, cr_mask).astype("uint8")
        cr_mask = cr_mask.astype("uint8")

        return cr_mask, single_pixels

    def reject_outlier_frm(self):
        shape = self.cr_mask.shape

        # cr counts in each frame
        cr_counts = np.sum(self.cr_mask, axis=(1, 2))

        total_cr = 1 if np.sum(cr_counts) == 0 else np.sum(cr_counts)
        frame_cr_contribution = cr_counts / total_cr

        # if the # of CR from one frames is more than any of the two conditions
        #   1. twice the average,
        #   2. two standard deviation of avergae contribution (outlier)
        # reject this frame, as it might have satellite trail, mis-aligned, or blurry

        mean = np.mean(frame_cr_contribution)
        reject_thres = min(2.0 * mean, mean + 2.0 *
                           np.std(frame_cr_contribution))

        reject_idx = np.where(frame_cr_contribution > reject_thres)[0].tolist()

        # save rejected frames for reference
        if len(reject_idx) > 0:

            for i in range(shape[0]):
                name = "noisy_rejected" if i in reject_idx else "noisy_accepted"
                save_as_png(
                    self.frames[i],
                    self.reject_dir,
                    self.filenames[i],
                    name,
                    zscale=True,
                )

            with open(self.opt.log_file, "a") as log:
                msg = (
                    f"{self.fname} rejected, reject_thres {reject_thres}\n"
                    f"{frame_cr_contribution}\n\n"
                )
                log.write(msg)
                print(msg)

        # remove rejected frames from frame stack and fits
        if len(reject_idx) > 0:
            self.frames = remove_outlier_frm(self.frames, reject_idx)
            self.frmtotal = self.frames.shape[0]

            self.frames_banzai = remove_outlier_frm(
                self.frames_banzai, reject_idx)

            self.cr_mask = remove_outlier_frm(self.cr_mask, reject_idx)

            self.filenames = remove_outlier_frm(
                np.array(self.filenames), reject_idx
            ).tolist()

            self.hdul = remove_outlier_frm(
                self.hdul, reject_idx, update_hdul=True)

        return reject_idx

    def output_png_preview(self, reason):
        if "reject" in reason:
            save_array_png(
                self.frames_banzai,
                None,
                None,
                None,
                self.opt.root,
                self.filenames,
                save_np=False,
                save_png=True,
                header=self.hdul[1].header,
                comment=reason,
            )
        else:
            save_array_png(
                self.frames,
                self.cr_mask,
                self.clean_frames,
                self.ignore_mask,
                self.opt.root,
                self.filenames,
                save_np=False,
                save_png=True,
                header=self.hdul[1].header,
                comment=reason,
            )
