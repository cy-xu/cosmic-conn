import os
import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cosmic_conn.inference_cr import init_model

os.environ['CUDA_VISIBLE_DEVICES']="1"


full_frame = 'paper_utils/cr_shape_fits/coj0m405-kb56-20191027-0113-e91_9frms_masks_max_source_dil5_min3.fits'

with fits.open(full_frame) as hdu:
    frame = np.stack([hdu[1].data, hdu[2].data, hdu[3].data])

    predictions = []
    cr_model, opt = init_model('ground_imaging')
    # cr_model, opt = init_model('deepcr')

    for f in frame:
        pdt_mask = cr_model.detect_cr(f, ret_numpy=True)
        predictions.append(pdt_mask)

    height = 80
    width = 160
    y = 1295
    x = 2280
    win = [y, y + height, x, x + width]

    # prep frames for paper vis
    frame = frame - np.min(frame) + 1
    frame = np.log10(frame)

    vmax = np.percentile(frame, 99.9) * -1
    vmin = np.percentile(frame, 1) * -1

    frame = frame[:, win[0]:win[1], win[2]:win[3]]

    # prep GT masks for paper vis
    cr_mask = np.stack([hdu[6].data, hdu[7].data, hdu[8].data])
    cr_mask = cr_mask[:, win[0]:win[1], win[2]:win[3]]

    # prep predicted masks for paper vis
    predictions = np.stack(predictions)
    predictions = (predictions > 0.1).astype('uint8')

    # Clip and normalize to 1 to increase contrast for white background
    predictions = np.clip(predictions, 0., 0.2) * 5.0
    predictions = predictions[:, win[0]:win[1], win[2]:win[3]]

    # src_mask = np.stack([hdu[9].data, hdu[9].data, hdu[9].data])
    # src_mask = src_mask[:, win[0]:win[1], win[2]:win[3]]

plt.rcParams['figure.dpi'] = 600
plt.rcParams.update({'font.size': 12})
# plt.rcParams['xtick.labelsize'] = 5
# plt.rcParams['ytick.labelsize'] = 5

# fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
fig.set_size_inches(11.5, 6)
plt.subplots_adjust(wspace=0, hspace=0)

# Frames
ax[0,0].imshow(frame[0] * -1, cmap='gray', vmin=vmax, vmax=vmin)
ax[0,0].set_ylabel('Frames')
ax[0,0].set_ylim(0, height)
ax[0,0].set_xlim(0, width)
ellipse = Ellipse((61, 43), width=16, height=106, angle=45, alpha=0.5, facecolor='none', edgecolor='r', linestyle='--')
ax[0,0].add_patch(ellipse)


ax[0,1].imshow(frame[1] * -1, cmap='gray', vmin=vmax, vmax=vmin)
ax[0,1].set_ylim(0, height)
ax[0,1].set_xlim(0, width)
ellipse = Ellipse((75, 47), width=10, height=10, angle=45, alpha=0.5, facecolor='none', edgecolor='r', linestyle='--')
ax[0,1].add_patch(ellipse)


ax[0,2].imshow(frame[2] * -1, cmap='gray', vmin=vmax, vmax=vmin)
ax[0,2].set_ylim(0, height)
ax[0,2].set_xlim(0, width)
ellipse = Ellipse((138, 13), width=16, height=16, angle=0, alpha=0.5, facecolor='none', edgecolor='r', linestyle='--')
ax[0,2].add_patch(ellipse)

# CR mask
ax[1,0].imshow(cr_mask[0] * -1, cmap='gray')
ax[1,0].set_ylabel('Ground truth')
ax[1,0].set_ylim(0, height)
ax[1,0].set_xlim(0, width)

ax[1,1].imshow(cr_mask[1] * -1, cmap='gray')
ax[1,1].set_ylim(0, height)
ax[1,1].set_xlim(0, width)

ax[1,2].imshow(cr_mask[2] * -1, cmap='gray')
ax[1,2].set_ylim(0, height)
ax[1,2].set_xlim(0, width)


# MODEL prediction
ax[2,0].imshow(predictions[0] * -1, cmap='gray')
ax[2,0].set_ylabel('Our predictions')
ax[2,0].set_ylim(0, height)
ax[2,0].set_xlim(0, width)
ax[2,0].set_xlabel('frame 1')

ax[2,1].imshow(predictions[1] * -1, cmap='gray')
ax[2,1].set_ylim(0, height)
ax[2,1].set_xlim(0, width)
ax[2,1].set_xlabel('frame 2')

ax[2,2].imshow(predictions[2] * -1, cmap='gray')
ax[2,2].set_ylim(0, height)
ax[2,2].set_xlim(0, width)
ax[2,2].set_xlabel('frame 3')

# plt.subplot(ax[i, 3])
# plt.imshow(imgg * mask_lacosmic * -1, cmap='gray', vmin=vmax, vmax=vmin, alpha=1)
# ax[i, 1].imshow(imgg * mask_deepcr * -1, cmap='gray', vmin=vmax, vmax=vmin)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.03, hspace=0.03)
plt.savefig('paper_utils/cr_triplets.pdf', bbox_inches='tight')

print('CR shape figure saved to cr_triplets.pdf')