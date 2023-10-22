import numpy as np

from skimage import segmentation
from scipy import ndimage

import sep
from astropy.table import Table

from cosmic_conn.cr_pipeline.banzai_stats import robust_standard_deviation


def prune_nans_from_table(table):
    """
    From banzai
    https://github.com/LCOGT/banzai/blob/master/banzai/utils/array_utils.py
    """
    nan_in_row = np.zeros(len(table), dtype=bool)
    for col in table.colnames:
        nan_in_row |= np.isnan(table[col])
    return table[~nan_in_row]


def sep_source_mask(frames, valid_mask):
    """
    Following Banzai's SEP procedure with some modifications
    https://github.com/LCOGT/banzai/blob/master/banzai/photometry.py
    SEP manual
    https://www.astromatic.net/pubsvn/software/sextractor/trunk/doc/sextractor.pdf
    1. Extrat sources from the median frame, more reliable than SEP on individual frames
    2. Calculate each soruce's SEP stats, ellipse, window size
    3. Expand sources' rectangle window, acquire a local background mask
    4. Local background mask shuold be very close across frames, except for extreme PSF wings
    5. Calculate local background stats for each source and extract sources' mask from background
    6. Generate a source mask for each frame
    """

    # lower threshold as we work on the median frame
    threshold = 3.5
    min_area = 9

    # prep median frame for SEP source extractor
    median_frame = np.median(frames, axis=0)

    # increase limit as we work on the median frame
    ny, nx = median_frame.shape
    sep.set_extract_pixstack(int(nx * ny * 0.10))

    # subtract background
    bkg = sep.Background(median_frame, bw=32, bh=32, fw=3, fh=3)
    bkg_img = bkg.back()
    median_frame -= bkg_img

    # SEP applies a gaussian blur kernel
    # [[1,2,1], [2,4,2], [1,2,1]] by default
    try:
        # corner case when active sources pixel exceeds limit will crash
        sources = sep.extract(
            median_frame, threshold, err=bkg.globalrms, minarea=min_area
        )
    except:
        return None

    # Convert the detections into a table
    sources = Table(sources)

    if len(sources) > 3000:
        return None

    # We remove anything with a detection flag >= 8
    # This includes memory overflows and objects that are too close the edge
    sources = sources[sources["flag"] < 8]

    sources = prune_nans_from_table(sources)

    # Calculate the ellipticity and elongation
    sources["ellipticity"] = 1.0 - (sources["b"] / sources["a"])
    sources["elongation"] = sources["a"] / sources["b"]

    # Fix any value of theta that are invalid due to floating point rounding
    # -pi / 2 < theta < pi / 2
    sources["theta"][sources["theta"] > (np.pi / 2.0)] -= np.pi
    sources["theta"][sources["theta"] < (-np.pi / 2.0)] += np.pi

    # Calculate the FWHMs of the stars:
    fwhm = 2.0 * (np.log(2) * (sources["a"]
                               ** 2.0 + sources["b"] ** 2.0)) ** 0.5
    sources["fwhm"] = fwhm

    # small fwhm are often bad pixels/dust on CCD (consistent across frames)
    sources = sources[fwhm > 1.0]

    """
    Generate local mask for each source in each frame
    """
    sources_masks = []

    img_height = median_frame.shape[0]
    img_width = median_frame.shape[1]

    for frame in frames:
        sources_mask = np.zeros_like(median_frame, dtype="uint8")

        for i in range(len(sources)):
            src = sources[i]

            src_ellipse = np.zeros_like(median_frame, dtype="uint8")

            # soruce's ellipse based on the median frame
            sep.mask_ellipse(
                src_ellipse, src["x"], src["y"], src["a"], src["b"], src["theta"], r=2
            )

            # expand source window by 1.5x so more sky and PSF wings could be included
            x_expand = int((src["xmax"] - src["xmin"]) * 0.25)
            y_expand = int((src["ymax"] - src["ymin"]) * 0.25)

            # min, max to handle boundary cases
            ymin = max(0, src["ymin"] - y_expand)
            ymax = min(src["ymax"] + y_expand, img_height)
            xmin = max(0, src["xmin"] - x_expand)
            xmax = min(src["xmax"] + x_expand, img_width)

            # fail safe #0, if source too close to valid_mask boundary, pass
            # we got valid mask from reprojecting consecutive frames to same coordinates
            # so some frame will have missing information near boundary
            # valid_count = np.sum(valid_mask[ymin:ymax, xmin:xmax] == 0)
            # if valid_count > 0:
            #     continue

            src_window = np.zeros_like(median_frame, dtype="uint8")
            src_window[ymin:ymax, xmin:xmax] = 1

            # remove soruce ellipse from source window mask
            bkg_mask = src_window * (1 - src_ellipse)

            # fail safe #1, if ellipse larger than window, use ellipse directly
            if bkg_mask.sum() == 0.0:
                sources_mask[src_ellipse > 0] = 1
                continue

            # window crop background pixels from each frame
            bkg_frm = bkg_mask * frame

            # shift background and source pxiels values
            # valid bkg area will be above zero, all other area == 0
            zero_offset = np.min(bkg_frm) - 1
            src_frm = (frame - zero_offset) * src_window
            bkg_frm = (bkg_frm - zero_offset) * bkg_mask

            # if effective background usable, get local background stats
            # consider Lucy smoothing
            bkg_vec = bkg_frm[ymin:ymax, xmin:xmax].flatten()
            bkg_vec = bkg_vec[bkg_vec != 0]

            # use 2 sigma above local background median as the spread boundary
            # robust standard deviation ensures to reject PFS wings from stats
            # 2 sigma widen the source's extent slightly but includes more soft contour
            bkg_std = robust_standard_deviation(bkg_vec)
            bkg_upperbound = np.median(bkg_vec) + 2.0 * bkg_std

            src_peak_value = np.max(src_frm)
            # src_peak_value = max(src_peak, np.max(src_frm))

            # fail safe #2, if background upperbound higher than source peak
            if bkg_upperbound >= src_peak_value:
                sources_mask[src_ellipse > 0] = 1
                continue

            # effective pixel values range for the source
            tolerance = src_peak_value - bkg_upperbound

            # starting point for floodfill, from true peak pixel
            seed_point = np.unravel_index(src_frm.argmax(), src_frm.shape)
            # seed_point = (src['ypeak'], src['xpeak'])

            # flood filling is better than thresholding
            # as the mask expands from source center, CR in the source window won't be flagged
            src_mask = segmentation.flood(
                src_frm, seed_point, tolerance=tolerance
            ).astype("uint8")

            sources_mask[src_ellipse > 0] = 1
            sources_mask[src_mask > 0] = 1

        sources_masks.append(sources_mask)

    sources_masks = np.stack(sources_masks)

    # dilate CR by n pixel to avoid introducing artificl sharp edges
    struct = ndimage.generate_binary_structure(2, 2)
    sources_dilation = np.zeros_like(sources_masks)

    for i in range(sources_masks.shape[0]):
        sources_dilation[i] = ndimage.binary_dilation(
            sources_masks[i], structure=struct, iterations=2
        )

    return sources_dilation