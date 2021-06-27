import os
import math
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

if __name__ == "__main__" and __package__ is None:
    from sys import path
    from os.path import dirname as dir
    path.append(dir(path[0]))
    __package__ = "root_dir"

from inference_cr import init_model, detect_2d_array
# os.environ['CUDA_VISIBLE_DEVICES']="1"

threshold = 0.5
preferred_model = 'cosmic_conn'
# preferred_model = 'deepcr'

# input_dir = 'data/pod_sync/SEP_TEST_SET/masked_fits/'
# input_dir = 'paper_utils/prediction_comparison/scrappy_results/sigfrac0.1_with_ignore_1px_dilation'
input_dir = 'data/pod_sync/SEP_dilation5/'

# init models
cosmic_model, opt = init_model('cosmic_conn')
deepcr_model, opt = init_model('deepcr')

# out dir
out_dir = f'paper_utils/prediction_comparison/LCO_predictions_{preferred_model}'
os.makedirs(out_dir, exist_ok=True)


def find_outstanding_CR(xor_cosmic, xor_deepcr):
    patches = []
    border = 64
    patch = 64
    id = 0

    shape = xor_cosmic.shape
    hh = (shape[0] - 2 * border) // patch
    ww = (shape[1] - 2 * border) // patch

    for i in range(hh):
        for j in range(ww):
            # overlapping crop at stride
            h_start = border + i * patch
            w_start = border + j * patch
            h_stop = min(h_start + patch, shape[0])
            w_stop = min(w_start + patch, shape[1])

            cos = xor_cosmic[h_start: h_stop, w_start: w_stop]
            dep = xor_deepcr[h_start: h_stop, w_start: w_stop]
            indices = [h_start, h_stop, w_start, w_stop]

            if preferred_model == 'deepcr':
                diff_score = np.sum(cos) - np.sum(dep)
            else:
                diff_score = np.sum(dep) - np.sum(cos)

            # at least 3 pixels different
            if diff_score > 5:
                patches.append([diff_score, indices])

    # sort patches by sum
    sorted_patches = sorted(patches, key=lambda kv: kv[0])

    # pick top 5 as outstanding CRs and plot a figure
    return [x[1] for x in sorted_patches]


def plot_outstanding_CR(frame, mask_gt, mask_lac, pdt_cosmic, pdt_deepcr, indices, outdir, fname):

    vmin = np.percentile(frame, 1)
    vmax = np.percentile(frame, 99.5)

    for index in indices:
        h_start, h_stop, w_start, w_stop = index

        frm = frame[h_start:h_stop, w_start:w_stop]
        gt = mask_gt[h_start:h_stop, w_start:w_stop]
        cosmic = pdt_cosmic[h_start:h_stop, w_start:w_stop]
        deepcr = pdt_deepcr[h_start:h_stop, w_start:w_stop]
        scrappy = mask_lac[h_start:h_stop, w_start:w_stop]

        plt.rcParams['figure.dpi'] = 600
        plt.rcParams.update({'font.size': 2})

        # fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
        fig, ax = plt.subplots(1, 5)
        fig.set_size_inches(3, 1)
        # plt.subplots_adjust(wspace=0, hspace=0)

        # Frame
        ax[0].imshow(frm, cmap='gray', vmin=vmin, vmax=vmax)
        ax[0].set_title('Frame')

        ax[1].imshow(gt, cmap='gray')
        ax[1].set_title('GT')

        ax[2].imshow(scrappy, cmap='gray')
        ax[2].set_title('AstroScrappy')

        ax[3].imshow(deepcr, cmap='gray')
        ax[3].set_title('deepCR')

        ax[4].imshow(cosmic, cmap='gray')
        ax[4].set_title('Cosmic-CoNN')

        # plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        # plt.subplots_adjust(wspace=0.05, hspace=0.05)

        [axi.set_axis_off() for axi in ax.ravel()]
        plt.savefig(f'{outdir}/{fname}_{index}.png', bbox_inches='tight')
        plt.close()
        print(f'saved {fname}_{index}.png')


for root, dirs, files in os.walk(input_dir):
    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']

    if 'predictions' not in root:
        for f in files:

            with fits.open(os.path.join(root, f)) as hdul:
                # test multiple frames in one fits
                frmtotal = hdul[0].header['frmtotal']

                for j in range(frmtotal):
                    try:
                        assert hdul[1 + j].header['EXTNAME'] == 'SCI'
                        frame = hdul[1 + j].data.astype('float32')
                        mask_gt = hdul[3 + frmtotal + j].data.astype('float32')
                        mask_ignore = hdul[3 + 3 +
                                           frmtotal + j].data.astype('float32')
                        mask_lac = hdul[3 + 3 + 3 +
                                        frmtotal + j].data.astype('uint8')
                    except:
                        print(f'check {f}')

                    # convert byteorder
                    if '>' in frame.dtype.byteorder:
                        frame = frame.byteswap(True).newbyteorder()

                    # some value could be Nan
                    frame = np.nan_to_num(frame)

                    pdt_cosmic = detect_2d_array(
                        cosmic_model, frame, 1024, return_numpy=True)
                    pdt_deepcr = detect_2d_array(
                        deepcr_model, frame, 1024, return_numpy=True)

                    pdt_cosmic = (pdt_cosmic > threshold).astype(
                        'uint8') * (1-mask_ignore)
                    pdt_deepcr = (pdt_deepcr > threshold).astype(
                        'uint8') * (1-mask_ignore)
                    mask_lac = mask_lac * (1-mask_ignore)

                    xor_cosmic = np.logical_xor(mask_gt, pdt_cosmic)
                    xor_deepcr = np.logical_xor(mask_gt, pdt_deepcr)
                    xor_lac = np.logical_xor(mask_gt, mask_lac)

                    indices = find_outstanding_CR(xor_cosmic, xor_deepcr)
                    plot_outstanding_CR(
                        frame, mask_gt, mask_lac, pdt_cosmic, pdt_deepcr, indices, out_dir, f)

                    indices = find_outstanding_CR(xor_cosmic, xor_lac)
                    plot_outstanding_CR(
                        frame, mask_gt, mask_lac, pdt_cosmic, pdt_deepcr, indices, out_dir, f)
