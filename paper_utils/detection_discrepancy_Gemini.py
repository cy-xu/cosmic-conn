import os
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
import astroscrappy.astroscrappy as lac
from skimage.morphology import dilation, square

from sys import path
from os.path import dirname as dir

path.append(dir(path[0]))

from inference_cr import init_model


def find_outstanding_diff(threshold, mask_a, preferred_mask):
    patches = []
    h_padding = 0
    w_padding = 0
    patch = 512

    shape = mask_a.shape
    hh = (shape[0] - 2 * h_padding) // patch
    ww = (shape[1] - 2 * w_padding) // patch

    for i in range(hh):
        for j in range(ww):
            # overlapping crop at stride
            h_start = h_padding + i * patch
            w_start = w_padding + j * patch
            h_stop = min(h_start + patch, shape[0])
            w_stop = min(w_start + patch, shape[1])

            indices = [h_start, h_stop, w_start, w_stop]

            patch_a = mask_a[h_start:h_stop, w_start:w_stop]
            patch_preferred = preferred_mask[h_start:h_stop, w_start:w_stop]

            # at least 3 pixels different
            if patch_a.sum() >= patch_preferred.sum() + threshold:
                patches.append([np.sum(patch_a), indices])

    # sort patches by sum
    sorted_patches = sorted(patches, key=lambda kv: kv[0])

    # pick top 5 as outstanding CRs and plot a figure
    return [x[1] for x in sorted_patches]


def plot_outstanding_CR(
    frame,
    mask_gt,
    mask_cosmic,
    mask_deepcr,
    indices,
    outdir,
    fname,
):

    for index in indices:
        h_start, h_stop, w_start, w_stop = index

        frm = frame[h_start:h_stop, w_start:w_stop]
        gt = 1 - mask_gt[h_start:h_stop, w_start:w_stop]
        cos = 1 - mask_cosmic[h_start:h_stop, w_start:w_stop]
        dep = 1 - mask_deepcr[h_start:h_stop, w_start:w_stop]

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams['axes.linewidth'] = 0.5 #set the value globally

        plt.clf()
        fig, ax = plt.subplots(1, 4, gridspec_kw = {'wspace':0.05, 'hspace':0})

        # Frame
        frm = frm - np.min(frm) + 1
        vmin = np.log10(np.percentile(frm, 10)) * -1
        vmax = np.log10(np.percentile(frm, 99)) * -1

        ax[0].imshow(np.log10(frm) * -1, cmap="gray", vmin=vmax, vmax=vmin)

        patches = [frm, gt, cos, dep]
        h, w = frm.shape

        for i in range(1, 4, 1):
            patch = patches[i].copy()
            # mark diff for red channel
            xor = (gt - patch != 0)

            patch *= 255
            patch_rgb = np.stack([patch, patch, patch])

            # mark incorrect red
            patch_rgb[0][xor] = 255
            patch_rgb[1][xor] = 0
            patch_rgb[2][xor] = 0

            patch_rgb = patch_rgb.transpose((1,2,0)).astype("uint8")

            ax[i].imshow(patch_rgb)

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        plt.savefig(f"{outdir}/{fname}_{index}.png", bbox_inches="tight")
        # plt.savefig(f"{outdir}/{fname}_{index}.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"saved {fname}_{index}.png")

if __name__ == "__main__":

    # init models
    cosmic, _ = init_model("cosmic_conn")
    deepcr, _ = init_model("deepcr")
    selem = square(3)

    # settings
    threshold = 20  # number of XOR pixels

    original_data = "data/gemini_data/masked_fits/2x2_binning/hsigma5"
    out_dir = "paper_utils/detection_discrepancy_vis/Gemini_data/"
    os.makedirs(out_dir, exist_ok=True)

    for root, dirs, files in os.walk(original_data):
        files = [f for f in files if not f[0] == "."]

        for f in files:
            if "N20190312S0159_CR_dlt0_hsigma5" not in f:
                continue

            with fits.open(os.path.join(root, f)) as hdul:
                frame = np.array(hdul[1].data, dtype="float32")
                mask_ref = np.array(hdul[3].data == 8, dtype="float32")
                mask_ignore = np.array(hdul[3].data > 0, dtype="float32")
                mask_object = np.array(hdul[4].data > 0, dtype="float32")
                filename = hdul[0].header["ORIGNAME"]

            # Gemini ignore mask contain multiple classes, post process
            mask_ignore -= mask_ref
            mask_ignore = np.logical_or(mask_object, mask_ignore)

            # dilate the ignroe mask by 1px so the detector gaps are removed
            # inverse the ignore mask for computation
            mask_ignore = (1 - dilation(mask_ignore, selem)).astype("uint8")

            if "1x1" in original_data:
                effective_area = [[600, 3900, 1400, 4900]]
            else:
                effective_area = [[400, 1900, 800, 2500]]

            # Split frame into three effective areas and evaluate separately
            for i, a in enumerate(effective_area):

                frm = frame[a[0] : a[1], a[2] : a[3]]
                ref = mask_ref[a[0] : a[1], a[2] : a[3]]
                ign = mask_ignore[a[0] : a[1], a[2] : a[3]]
                fname = filename + f"_{i}"

                # test with two deep learning methods
                mask_cosmic = cosmic.detect_2d_array(frm, ret_numpy=True)
                mask_cosmic = (mask_cosmic >= 0.96).astype("uint8")

                mask_deepcr = deepcr.detect_2d_array(frm, ret_numpy=True)
                mask_deepcr = (mask_deepcr >= 0.82).astype("uint8")

                # apply ignore mask as GMOS reduction recipe did
                # ign = erase_boundary_np(ign, boundary_width=64)
                mask_cosmic *= ign
                mask_deepcr *= ign

                xor_cosmic = np.logical_xor(ref, mask_cosmic)
                xor_deepcr = np.logical_xor(ref, mask_deepcr)

                # xor = np.logical_xor(mask_cosmic, mask_deepcr)

                indices = find_outstanding_diff(threshold, xor_deepcr, xor_cosmic)

                # xor_ignore = np.logical_xor(mask_ignore, (mask_ignore_new > 0))

                plot_outstanding_CR(
                    frm,
                    ref,
                    mask_cosmic,
                    mask_deepcr,
                    indices,
                    out_dir,
                    f
                )

