# -*- coding: utf-8 -*-

"""
Plot the median-weighted mask Figure for paper
CY Xu (cxu@ucsb.edu)
"""

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import gaussian_filter

from cosmic_conn.cr_pipeline.banzai_stats import robust_standard_deviation

full_frame = 'paper_utils/data/cpt1m010-fa16-20181228-0010-e91_11frms_masks.fits'

with fits.open(full_frame) as hdu:
    frames = np.stack([hdu[1].data, hdu[2].data, hdu[3].data])
    cr_masks = np.stack([hdu[6].data, hdu[7].data, hdu[8].data])
    median_frame = np.median(frames, axis=0)

    height = 110
    width = 110
    y = 1550
    x = 1180
    win = [y, y + height, x, x + width]

    # crop the image stamps
    frames = frames[:, win[0]:win[1], win[2]:win[3]]
    cr_masks = cr_masks[:, win[0]:win[1], win[2]:win[3]]
    median_frm = median_frame[win[0]:win[1], win[2]:win[3]]


plt.rcParams['figure.dpi'] = 600
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

fig = plt.figure()
fig.set_size_inches(10, 8)

# ============
# First image
# ============

ax = fig.add_subplot(2, 2, 1)

img_frm = frames[2]

img_frm = img_frm - np.min(img_frm) + 1
img_frm = np.log10(img_frm)

img_frm = np.flipud(img_frm)

vmin = np.percentile(img_frm, 99.9) * -1
vmax = np.percentile(img_frm, 1) * -1

ax.imshow(img_frm * -1, cmap='gray', vmin=vmin,
          vmax=vmax, extent=[0, height, 0, width])
ax.set_title('(a) image stamp')

# ============
# Second plot
# ============

# Make data.
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
x, y = np.meshgrid(x, y)


median, std = np.median(median_frm), np.std(median_frm)

# Plot the surface.  The triangles in parameter space determine which x, y, z
# points are connected by an edge.
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.set_zticks([])

surf1 = ax.plot_surface(x, y, median_frm,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=True)

fig.colorbar(surf1, ax=ax, shrink=0.8, aspect=20)

ax.set_title('(b) median frame in 3D')


# ============
# Third plot
# ============

cr_mask = cr_masks[2]
cr_mask = np.flipud(cr_mask)

# Plot the surface.
ax = fig.add_subplot(2, 2, 3)

ax.imshow(cr_mask * -1, cmap='gray', extent=[0, height, 0, width])

# ax.set_zlim(0, 1.0)
ax.set_title('(c) CR mask')

# ============
# Fourth plot
# ============

# .. paper method
robust_std = robust_standard_deviation(median_frm) + 1e-7
median_median = np.median(median_frm)
median_frm -= median_median

floor = max(0, robust_std)
ceil = min(5 * robust_std, median_frm.max())
mask = np.clip(median_frm, floor, ceil)
mask -= floor

mask = gaussian_filter(mask, sigma=2)

median_weighted_mask = mask / mask.max()

# post processing for vis only, not in training
median_weighted_mask -= 0.01
median_weighted_mask *= 1.02

# clip the alpha
median_weighted_mask = np.clip(
    median_weighted_mask, 0.1, median_weighted_mask.max())

# Plot the surface.
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.set_zticks([0.1])

surf2 = ax.plot_surface(x, y, median_weighted_mask, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, vmin=0, vmax=1.0)

fig.colorbar(surf2, ax=ax, shrink=0.8, aspect=20,
             ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_zlim(0, 1.0)
ax.set_title('(d) median-weighted mask')

plt.subplots_adjust(wspace=-0.45, hspace=0.3)

plt.savefig('./paper_utils/fig4_median_weighted_mask.pdf',
            bbox_inches='tight')
print('figure saved to median_weighted_mask.pdf')
