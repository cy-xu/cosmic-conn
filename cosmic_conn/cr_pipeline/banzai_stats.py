# -*- coding: utf-8 -*-

import numpy as np

__author__ = 'cmccully'


def median(d, axis=None, mask=None):
    output_median = np.median(d, axis=axis)
    return output_median


def absolute_deviation(a, axis=None, mask=None):
    """
    Find the absolute deviation from the median of a numpy array. If an axis is provided,
    then the median is calculated along the given axis.
    If mask is included, elements in a that have a non-zero mask value will be ignored.

    Parameters
    ----------
    a : float32 numpy array
        Input array to find the absolute deviation from the median.

    axis : int (default is None)
           Index of the array to take the median
    mask : unit8 or boolean numpy array (default is None)
          Numpy array of bitmask values. Non-zero values are ignored when calculating the median.

    Returns
    -------
    absdev : float32 numpy array
        The absolute deviation from the median. If axis is None, then we return a single float.

    Notes
    -----
    If all of the elements in the array are masked (or all of the elements of the axis of interest
    are masked), the original array is returned.
    """

    a_median = median(a, axis=axis, mask=mask)
    if axis is not None:
        a_median = np.expand_dims(a_median, axis=axis)
    return np.abs(a - a_median)


def median_absolute_deviation(a, axis=None, abs_deviation=None, mask=None):
    if abs_deviation is None:
        abs_deviation = absolute_deviation(a, axis=axis, mask=mask)

    return median(abs_deviation, axis=axis, mask=mask)


def robust_standard_deviation(a, axis=None, abs_deviation=None, mask=None):
    return 1.4826 * median_absolute_deviation(a, axis=axis, abs_deviation=abs_deviation, mask=mask)


def sigma_clipped_mean(a, sigma, axis=None, mask=None, fill_value=0.0, inplace=False):
    """
    """
    abs_deviation = absolute_deviation(a, axis=axis, mask=mask)

    robust_std = robust_standard_deviation(
        a, axis=axis, abs_deviation=abs_deviation, mask=mask)

    if axis is not None:
        robust_std = np.expand_dims(robust_std, axis=axis)

    # Throw away any values that are N sigma from the median
    sigma_mask = abs_deviation > (sigma * robust_std)

    if mask is not None:
        sigma_mask = np.logical_or(sigma_mask, mask > 0)
    if inplace:
        mean_array = a
    else:
        mean_array = a.copy()

    mean_array[sigma_mask] = 0.0

    # Take the sigma clipped mean
    mean_values = mean_array.sum(axis=axis)

    n_good_pixels = np.logical_not(sigma_mask).sum(axis=axis)
    if axis is None:
        if n_good_pixels > 0:
            mean_values /= n_good_pixels
        else:
            mean_values = fill_value
    else:
        mean_values[n_good_pixels > 0] /= n_good_pixels[n_good_pixels > 0]
        mean_values[n_good_pixels == 0] = fill_value

    return mean_values
