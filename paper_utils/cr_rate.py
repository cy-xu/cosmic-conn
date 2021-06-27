# -*- coding: utf-8 -*-

"""
produce CR rates and other statistics of the LCO CR dataset
CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np

from astropy.io import fits
from scipy.ndimage import label, generate_binary_structure

from cosmic_conn.cr_pipeline.utils_img import erase_boundary_np

input_dir = '/home/cyxu/astro/Cosmic_CoNN_datasets/LCO_CR_dataset'
metrics = []

boundary = 100
s = generate_binary_structure(2, 2)

total_frames = 0
total_exp_time = 0  # seconds
total_ccd_area = 0  # cm^2
total_cr_count = 0
total_adu = 0
cr_in_ignore_mask = []
cr_num_in_ignore_mask = []

observed_cr_pixels_list = []
effective_pixels_list = []

for root, dirs, files in os.walk(os.path.join(input_dir)):

    if 'masked_fits' not in root:
        continue

    files = [f for f in files if not f[0] == '.']
    dirs[:] = [d for d in dirs if not d[0] == '.']

    for f in files:
        assert f.endswith('fits'), f'check {f}'

        with fits.open(os.path.join(root, f)) as hdul:
            # test multiple frames in one fits
            frmtotal = hdul[0].header['frmtotal']

            for j in range(frmtotal):
                print(f'frame {total_frames}')

                assert hdul[1 + j].header['EXTNAME'] == 'SCI'
                frame = hdul[1 + j].data.astype('float32')
                mask_gt = hdul[3 + frmtotal + j].data.astype('uint8')
                mask_ignore = hdul[3 + 3 + frmtotal + j].data.astype('uint8')

                exp_time = hdul[1].header['EXPTIME']
                gain = hdul[1].header['gain']

                # header keyword in meter, convert m to cm with * 100
                ccd_x = hdul[1].header['CCDXPIXE'] * 100
                ccd_y = hdul[1].header['CCDYPIXE'] * 100

                # removing 100 pixel boundary, consistent with ignore mask
                effective_pixels = (frame.shape[0] - 200) * \
                    (frame.shape[1] - 200)

                ccd_area = effective_pixels * ccd_y * ccd_x  # in cm^2

                # count individual CRs
                labelled_gt, num_gt = label(mask_gt, structure=s)

                # > 1 to remove good pixels (0) and boundary pixels (1)
                mask_ignore = erase_boundary_np(mask_ignore, boundary)
                mask_ignore = mask_ignore > 1

                # assuming uniform distribution of CR on the CCD
                # then the CRs overlapped with source share the same rate
                ignored_pixels_ratio = np.sum(mask_ignore) / effective_pixels
                cr_num_in_ignore_mask.append(ignored_pixels_ratio)

                # by CR pixel
                estimated_total_cr_pixels = np.sum(mask_gt) / \
                    (1 - ignored_pixels_ratio)

                # by CR count
                estimated_total_num_gt = num_gt / (1 - ignored_pixels_ratio)

                total_frames += 1
                total_exp_time += exp_time
                total_ccd_area += ccd_area  # cm^2
                total_cr_count += estimated_total_num_gt
                total_adu += np.sum(mask_gt * frame) * gain
                cr_in_ignore_mask.append(ignored_pixels_ratio)
                observed_cr_pixels_list.append(estimated_total_cr_pixels)
                effective_pixels_list.append(effective_pixels)

CR_flux = total_cr_count / total_exp_time / total_ccd_area
CR_energy = total_adu / total_exp_time / total_ccd_area

print(f'In {total_frames} frames, average CR flux = {CR_flux} CR/cm^2/s, CR energy = {CR_energy} e/cm^2/s')

# print(f'About {np.mean(cr_num_in_ignore_mask)}% (mean) {np.median(cr_num_in_ignore_mask)}% (median) of total CRs (number) affected by ignore mask.')

print(f'About {np.mean(cr_in_ignore_mask)} (mean) {np.median(cr_in_ignore_mask)} (median) of total CRs (pixels) affected by ignore mask.')

breakpoint()

print(
    f'total CR pixels / total image pixels / second = {np.sum(observed_cr_pixels_list) / np.sum(effective_pixels_list) / total_exp_time}')
