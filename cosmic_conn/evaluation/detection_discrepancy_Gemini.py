import os
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
import astroscrappy.astroscrappy as lac
from skimage.morphology import dilation, square

from cosmic_conn.inference_cr import init_model
from cosmic_conn.data_utils import console_arguments


def find_outstanding_diff(threshold, mask_a, mask_b):
    patches = []
    h_padding = 0
    w_padding = 0
    patch = 200

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
            patch_b = mask_b[h_start:h_stop, w_start:w_stop]

            # at least 3 pixels different
            if np.sum(np.logical_xor(patch_a, patch_b)) >= threshold:
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
        plt.rcParams['axes.linewidth'] = 0.5  # set the value globally

        plt.clf()
        fig, ax = plt.subplots(1, 4, gridspec_kw={'wspace': 0.05, 'hspace': 0})

        # Frame
        frm = frm - np.min(frm) + 1
        vmin = np.log10(np.percentile(frm, 10)) * -1
        vmax = np.log10(np.percentile(frm, 99)) * -1

        ax[0].imshow(np.log10(frm) * -1, cmap="gray", vmin=vmax, vmax=vmin)

        patches = [frm, gt, dep, cos]
        h, w = frm.shape

        for i in range(1, 4, 1):
            patch = patches[i].copy()
            # mark diff for red channel
            xor = (gt - patch != 0)

            patch *= 255
            patch_rgb = np.stack([patch, patch, patch])

            # mark incorrect red
            patch_rgb[0][xor] = 255
            patch_rgb[1][xor] = 100
            patch_rgb[2][xor] = 100

            patch_rgb = patch_rgb.transpose((1, 2, 0)).astype("uint8")

            ax[i].imshow(patch_rgb)

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

        plt.savefig(f"{outdir}/{fname}_{index}.png", bbox_inches="tight")
        plt.close()
        print(f"saved {fname}_{index}.png")


if __name__ == "__main__":

    # init models
    cosmic, _ = init_model("ground_imaging")

    opt = console_arguments()
    opt.norm = 'batch'
    deepcr, _ = init_model(
        "checkpoints/2021_03_14_16_42_LCO_deepCR_continue/models/cr_checkpoint_5370.pth.tar", opt=opt
    )

    selem = square(3)

    # settings
    threshold = 0  # number of XOR pixels

    original_data = "/home/cyxu/astro/Cosmic_CoNN_datasets/GEMINI_testset/masked_fits"
    out_dir = "paper_PDFs/detection_discrepancy/Gemini_Cos_deepCR"
    os.makedirs(out_dir, exist_ok=True)

    for root, dirs, files in os.walk(original_data):
        files = [f for f in files if not f[0] == "."]

        for f in files:
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

            for i in range(1):
                frm = frame
                ref = mask_ref
                ign = mask_ignore
                fname = filename

                # test with two deep learning methods
                mask_cosmic = cosmic.detect_cr(frm, ret_numpy=True)
                mask_cosmic = (mask_cosmic >= 0.5).astype("uint8")

                mask_deepcr = deepcr.detect_cr(frm, ret_numpy=True)
                mask_deepcr = (mask_deepcr >= 0.5).astype("uint8")

                # apply ignore mask as GMOS reduction recipe did
                # ign = erase_boundary_np(ign, boundary_width=64)
                mask_cosmic *= ign
                mask_deepcr *= ign

                xor_cosmic = np.logical_xor(ref, mask_cosmic)
                xor_deepcr = np.logical_xor(ref, mask_deepcr)

                indices = find_outstanding_diff(
                    threshold, xor_deepcr, xor_cosmic)

                plot_outstanding_CR(
                    frm,
                    ref,
                    mask_cosmic,
                    mask_deepcr,
                    indices,
                    out_dir,
                    f
                )
