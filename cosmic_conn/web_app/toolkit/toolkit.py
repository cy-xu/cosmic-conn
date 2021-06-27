# -*- coding: utf-8 -*-

"""
Boning Dong, CY Xu (cxu@ucsb.edu)
"""

import math
import numpy as np


def find_noise_thumbnail_coords(mask, threshold=0.5, patch_size=32):
    """
    Return CR thumbnails's top-left corner coordinate
    """
    confidence_coord_tuples = []  # the coords have format [x, y]
    border = 64

    shape = mask.shape
    hh = math.ceil((shape[0] - 2 * border) // patch_size)
    ww = math.ceil((shape[1] - 2 * border) // patch_size)

    for i in range(hh):
        for j in range(ww):
            # overlapping crop at stride
            h_start = border + i * patch_size
            w_start = border + j * patch_size

            if h_start + patch_size > (shape[0] - border):
                h_start = shape[0] - border - patch_size

            if w_start + patch_size > (shape[1] - border):
                w_start = shape[1] - border - patch_size

            h_stop = h_start + patch_size
            w_stop = w_start + patch_size

            patch = mask[h_start:h_stop, w_start:w_stop]
            # np.where(patch > threshold, 1, 0)

            confidence_coord_tuples.append((patch.sum(), [w_start, h_start]))

    confidence_coord_tuples.sort(key=lambda elem: elem[0], reverse=True)

    return [x[1] for x in confidence_coord_tuples]
