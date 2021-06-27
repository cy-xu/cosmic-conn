# -*- coding: utf-8 -*-

"""
Main file to define the CR labeling pipelien for LCO NRES data
CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np
from astropy.io import fits

from skimage import morphology
from scipy import ndimage

from cosmic_conn.cr_pipeline.utils_io import save_as_png, save_array_png
from cosmic_conn.cr_pipeline.utils_io import remove_outlier_frm, remove_ext, mkdir


class NRES_CR_detector:
    def __init__(self, hdul=None, opt=None):
        self.hdul = hdul
        self.opt = opt

        header = self.hdul["SPECTRUM"].header

        self.fname = header["filename"]
        self.exp_time = header["exptime"]
        self.rdnoise = header["rdnoise"]
        self.gain = header["gain"]
        maxlin = header["maxlin"]

        self.min_cr_size = self.opt.min_cr_size

        frames, self.filenames = [], []
        for i in range(1, 4):
            frames.append(hdul[i].data)
            self.filenames.append(hdul[i].header["filename"])

        # 0. maxlin header cutoff
        frames = np.stack(frames).astype("float32")
        frames[frames > maxlin] = maxlin

        # 3. saturated pixel radius 3 dilation to ignore
        ignore_mask = (frames >= maxlin).astype("uint8")
        struct = ndimage.generate_binary_structure(2, 2)

        for i in range(ignore_mask.shape[0]):
            ignore_mask[i] = ndimage.binary_dilation(
                ignore_mask[i], structure=struct, iterations=3
            )
        self.ignore_mask = ignore_mask

        # 1. subtract overscan (right hand)
        overscan = frames[:, :, -35:]
        frames -= np.median(overscan, axis=(1, 2), keepdims=True)

        # 2. multiply gain
        self.frames = frames * self.gain
        # self.rdnoise *= self.gain

        # pre-processing, a separate copy for mask generation
        # self.frames = sigma_clip(self.frames, sigma=3.0, clip_bright_source=False)

        self.reject_dir = self.opt.root + "/rejected_frms"

    def generate_CR_masks(self):
        """
        This is the main CR detection loop
            estimate_reject_refine() is called one or multiple times for CR detection
            detect CR in the sequence of frames, reject outlier pixels and frames

        if estimate_reject_refine() returns a string, it's a error message during detection
        print the error message and abort this sequence
        """
        high_thres = self.opt.snr_thres
        low_thres = self.opt.snr_thres_low

        # initial estimation to reject outlier frame
        cr_frac = self.estimate_reject_refine(high_thres, low_thres)

        if isinstance(cr_frac, str):
            self.output_png_preview(reason=cr_frac)
            return cr_frac

        # if an outlier frame is rejected, detect again with the new sequence because the
        # CR frac between frames might change, until all frames considered well aligned
        # while len(self.reject_idx) > 0:
        #     cr_frac = self.estimate_reject_refine(high_thres, low_thres)
        #     if isinstance(cr_frac, str): return cr_frac

        # all rejection passed, print stats of the useable frames
        print(
            f"{self.fname}, exp_time {self.exp_time}, {[np.sum(m) for m in self.cr_mask]}\n",
            f"SNR {high_thres}, {low_thres}, CR fraction {round(cr_frac, 6)}\n",
        )

        # append CR mask
        for i in range(3):
            self.hdul.append(
                fits.CompImageHDU(
                    self.cr_mask[i],
                    header=self.hdul[i].header,
                    name="CR",
                    ver=i,
                    uint=True,
                )
            )

        for i in range(3):
            self.hdul.append(
                fits.CompImageHDU(
                    self.ignore_mask[i],
                    header=self.hdul[i].header,
                    name="IGNORE",
                    ver=i,
                    uint=True,
                )
            )

        new_path = os.path.join(self.opt.data, "masked_fits")
        new_filename = f"{remove_ext(self.fname)}_masks.fits"

        # save aligned fits to sub-directory
        mkdir(new_path)
        out_path = os.path.join(new_path, new_filename)

        self.hdul.writeto(out_path, overwrite=True)

        # output png preview or numpy as training set (no longer using numpy)
        if not self.opt.no_png:
            self.clean_frames = np.where(
                self.cr_mask > 0, self.median_frame, self.frames
            )
            self.output_png_preview(self.opt.comment)

    # @profile
    def estimate_reject_refine(self, high_thres, low_thres, reject_frm=True):
        """
        The main CR detection happens here:
            1. estimate two masks based on an initial SNR threshold and
               a lower threshold (for peripheral CR edges)

            4. refine CR to incldue soft edges via dilation/flood filling

            7. reject frame outliers with excessive CR in a sequence
               often false positives caused by satellite trails or mis-aligned frames 
        """

        # 1.
        mask_low, mask_high = self.estimate_two_masks(
            self.frames, self.rdnoise, high_thres, low_thres
        )
        # from here on, self.frames are sky subtracted

        # 4.
        self.cr_mask = self.maximize_cr_edges(
            mask_low, mask_high, self.min_cr_size)

        # return error message
        if isinstance(self.cr_mask, str):
            return self.cr_mask

        # 7.
        # if reject_frm:
        #     self.reject_idx = self.reject_outlier_frm()
        # else:
        #     self.reject_idx = []

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
        # frame_uncertainty = np.sqrt(read_noise**2 + np.abs(frames))

        median_frame_uncertainty = np.median(frame_uncertainty, axis=0) / np.sqrt(
            N_frames
        )

        # calculate the total noise including our model (approximately for the median)
        total_uncertainty = np.sqrt(
            frame_uncertainty ** 2 + median_frame_uncertainty ** 2
        )

        # subtract sky and get correct median
        # frames = subtract_sky(frames, remove_negative=False)
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

    def maximize_cr_edges(self, mask_low, mask_high, min_cr_size):
        shape = mask_high.shape
        cr_mask = np.zeros_like(mask_low, dtype="uint8")

        # dilation expands CR edges by N pixels, more conservative than flood filling
        # current 3 iterations will expand high mask by 3 pixels

        mask_high *= 1 - self.ignore_mask

        struct = ndimage.generate_binary_structure(2, 2)
        mask_dilation = np.zeros_like(mask_high)

        for i in range(shape[0]):
            mask_dilation[i] = ndimage.binary_dilation(
                mask_high[i], structure=struct, iterations=self.opt.dilation
            )

        cr_mask = np.logical_and(mask_dilation, mask_low).astype("uint8")

        cr_mask *= 1 - self.ignore_mask

        # save before removing single pixels to add them to ignore mask
        cr_mask_single_pixels = cr_mask.copy().astype("uint8")

        # remove flagged CR smaller than certain pixels from final mask
        if min_cr_size > 1:
            for i in range(shape[0]):
                # create labels in segmented image
                imglab = morphology.label(cr_mask[i], connectivity=2)
                # remove small patches less than min_cr_size
                cr_mask[i] = morphology.remove_small_objects(
                    imglab, min_size=min_cr_size, connectivity=2
                )

        cr_mask = (cr_mask > 0).astype("uint8")

        # add single pixels and source mask to make the ignore mask
        self.single_pixels = cr_mask_single_pixels - cr_mask
        self.ignore_mask = (
            (self.ignore_mask + self.single_pixels) > 0).astype("uint8")

        cr_mask *= 1 - self.ignore_mask

        return cr_mask

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

            print(f"rejected {self.fname}, reject_thres {reject_thres}")
            print(frame_cr_contribution)
            print()

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
                None,
                self.opt.root,
                self.filenames,
                save_np=False,
                save_png=True,
                header=self.hdul[1].header,
                comment=reason,
            )
