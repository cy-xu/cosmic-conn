import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

from cosmic_conn.inference_cr import init_model


full_frame = 'paper_utils/data/lscnrs01-fl09-20180129-0034-e00.fits.fz'

with fits.open(full_frame) as hdu:
    frame = hdu[1].data.astype('float32')

    cr_model, opt = init_model('NRES')
    pdt_mask = cr_model.detect_cr(frame, ret_numpy=True)

height = 250
width = 450
x = 1980
y = 1050
win = [y, y + height, x, x + width]

# prep frames for paper vis
frame = frame - np.min(frame) + 1
frame = np.log10(frame)

vmax = np.percentile(frame, 99.9) * -1
vmin = np.percentile(frame, 10) * -1

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
plt.savefig('paper_utils/nres_demo_img.pdf', bbox_inches='tight')

print('figure saved to nres_demo_img.pdf')
