import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cosmic_conn.inference_cr import init_model


full_frame = 'paper_utils/data/N20180811S0070_CR_dlt0_hsigma5.0.fits'

cr_model, opt = init_model('ground_imaging')

with fits.open(full_frame) as hdu:
    frame = hdu[1].data.astype('float32')
    pdt_mask = cr_model.detect_cr(frame, ret_numpy=True)

height = 1200
width = 1500
y = 2400
x = 2450
win = [y, y + height, x, x + width]

# prep frames for paper vis
frame = frame - np.min(frame) + 1
frame = np.log10(frame)

vmax = np.percentile(frame, 99.8) * -1
vmin = np.percentile(frame, 50) * -1

frame = frame[win[0]:win[1], win[2]:win[3]]
cr_mask = pdt_mask[win[0]:win[1], win[2]:win[3]]


plt.rcParams['figure.dpi'] = 600
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.linewidth': 0.2})

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
fig.set_size_inches(8, 3.5)

# Frames
ax[0].imshow(frame * -1, cmap='gray', vmin=vmax, vmax=vmin)
ax[0].set_ylim(0, height)
ax[0].set_xlim(0, width)

# CR mask
ax[1].imshow(cr_mask * -1, cmap='gray')
ax[1].set_ylim(0, height)
ax[1].set_xlim(0, width)

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.subplots_adjust(wspace=0.03, hspace=0.0)
plt.savefig('paper_utils/fig11_gemini_results_demo.pdf', bbox_inches='tight')

print('figure saved to fig11_gemini_results_demo.pdf')
