from inference_cr import init_model, detect_2d_array
from enum import auto
import os
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import astroscrappy.astroscrappy as lac

from sys import path
from os.path import dirname as dir

path.append(dir(path[0]))


def find_outstanding_diff(threshold, mask_a, preferred_mask):
    patches = []
    h_padding = 100
    w_padding = 100
    patch = 32
    id = 0

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
    mask_lac,
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
        lac = 1 - mask_lac[h_start:h_stop, w_start:w_stop]
        cos = 1 - mask_cosmic[h_start:h_stop, w_start:w_stop]
        dep = 1 - mask_deepcr[h_start:h_stop, w_start:w_stop]

        plt.rcParams["figure.dpi"] = 300
        plt.rcParams['axes.linewidth'] = 0.5  # set the value globally

        plt.clf()
        fig, ax = plt.subplots(1, 5, gridspec_kw={'wspace': 0.05, 'hspace': 0})
        # fig, ax = plt.subplots(1, 5, sharex=True, sharey=True)
        # plt.subplots_adjust(wspace=0, hspace=0)
        # fig.set_size_inches(5, 1)

        # Frame
        frm = frm - np.min(frm) + 1
        vmin = np.log10(np.percentile(frm, 10)) * -1
        vmax = np.log10(np.percentile(frm, 99)) * -1

        ax[0].imshow(np.log10(frm) * -1, cmap="gray", vmin=vmax, vmax=vmin)

        patches = [frm, gt, lac, cos, dep]
        h, w = frm.shape

        for i in range(1, 5, 1):
            patch = patches[i].copy()
            # mark diff for red channel
            xor = (gt - patch != 0)

            patch *= 255
            patch_rgb = np.stack([patch, patch, patch])

            # mark incorrect red
            patch_rgb[0][xor] = 255
            patch_rgb[1][xor] = 0
            patch_rgb[2][xor] = 0

            patch_rgb = patch_rgb.transpose((1, 2, 0)).astype("uint8")

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

    # settings
    threshold = 4  # number of XOR pixels

    original_data = "data/Cosmic_CoNN_CR_dataset/test_set/masked_fits"
    out_dir = "paper_utils/detection_discrepancy_vis/prefer_cosmic"
    os.makedirs(out_dir, exist_ok=True)

    for root, dirs, files in os.walk(original_data):
        files = [f for f in files if not f[0] == "."]

        for f in files:
            if "tfn0m414-kb25-20190108-0221" not in f:
                continue

            with fits.open(os.path.join(root, f)) as hdul:
                # test multiple frames in one fits
                frmtotal = hdul[0].header["frmtotal"]

                for j in range(frmtotal):
                    print(f"processing frame {j} of {f}")

                    assert hdul[1 + j].header["EXTNAME"] == "SCI"
                    frame = hdul[1 + j].data.astype("float32")
                    mask_gt = hdul[3 + frmtotal + j].data
                    mask_ignore = hdul[3 + 3 + frmtotal + j].data
                    mask_ignore[mask_ignore > 0] = 1

                    # test with Astro-SCRAPPY
                    saturate = hdul[1].header["SATURATE"]
                    gain = hdul[1 + j].header["gain"]
                    rdnoise = hdul[1 + j].header["rdnoise"]

                    objlim = 0.5 if "0m4" in f else 2.0
                    sigfrac = 0.1
                    sigclip = 5

                    mask_lac, _ = lac.detect_cosmics(
                        frame,
                        objlim=objlim,
                        sigclip=sigclip,
                        sigfrac=sigfrac,
                        gain=gain,
                        readnoise=rdnoise,
                        satlevel=saturate,
                        sepmed=False,
                        cleantype="meanmask",
                        niter=4,
                    )

                    # test with two deep learning methods
                    mask_cosmic = cosmic.detect_2d_array(frame, ret_numpy=True)
                    mask_cosmic = mask_cosmic >= 0.46

                    mask_deepcr = deepcr.detect_2d_array(frame, ret_numpy=True)
                    mask_deepcr = mask_deepcr >= 0.52

                    # mask_lac = mask_lac * (1 - mask_ignore)
                    # mask_cosmic = mask_cosmic * (1 - mask_ignore)
                    # mask_deepcr = mask_deepcr * (1 - mask_ignore)

                    xor_lac = np.logical_xor(mask_gt, mask_lac)
                    xor_cosmic = np.logical_xor(mask_gt, mask_cosmic)
                    xor_deepcr = np.logical_xor(mask_gt, mask_deepcr)

                    # xor = np.logical_xor(mask_cosmic, mask_deepcr)

                    indices = find_outstanding_diff(
                        threshold, xor_deepcr, xor_cosmic)

                    # xor_ignore = np.logical_xor(mask_ignore, (mask_ignore_new > 0))

                    plot_outstanding_CR(
                        frame,
                        mask_gt,
                        mask_lac,
                        mask_cosmic,
                        mask_deepcr,
                        indices,
                        out_dir,
                        f
                    )
