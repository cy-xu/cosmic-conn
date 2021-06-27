# -*- coding: utf-8 -*-

"""
Defines command line arguments parser for CR labeling pipeline
CY Xu (cxu@ucsb.edu)
"""

import argparse


def argument_parser():
    """
    options for CR mask generation, default values are recommended, example command:
    `python stack_clean.py data/sample_path --sigma 5 --min_cr_size 3 --verbose`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="./data",
        help="path to banzai fits, banzai fits much be placed in banzai_frms directory",
    )

    parser.add_argument(
        "--min_exptime",
        type=float,
        default=100.0,
        help="minimum exposure time used to generate the CR mask",
    )

    parser.add_argument(
        "--snr_thres", type=float, default=5.0, help="initial SNR threshold"
    )

    parser.add_argument(
        "--snr_thres_low", type=float, default=2.5, help="initial SNR threshold"
    )

    parser.add_argument(
        "--dilation",
        type=int,
        default=1,
        help="minimum continuum CR size in pixels, used to reject very small CR that are possibily false positive",
    )

    parser.add_argument(
        "--flood_fill",
        action="store_true",
        help="use flood fill algorithm to expand peripheral pixels",
    )

    parser.add_argument(
        "--aligned",
        action="store_true",
        help="frames already aligned and saved in aligned_fits directroy",
    )

    parser.add_argument(
        "--min_cr_size",
        type=int,
        default=3,
        help="minimum continuum CR size in pixels, used to reject very small CR that are possibily false positive",
    )

    parser.add_argument(
        "--comment", default=None, help="remarks saved at end of export filename"
    )

    parser.add_argument(
        "--no_png", action="store_true", help="do NOT export png if flagged"
    )

    parser.add_argument(
        "--linear_model",
        action="store_true",
        help="do NOT use the linear CR model, simply the initial Sigma (5 by default)",
    )

    parser.add_argument(
        "--scrappy", action="store_true", help="generate CR from astro_scrappy"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="print more info for debug"
    )

    parser.add_argument(
        "--debug", action="store_true", help="print more info for debug"
    )

    parser.add_argument(
        "--nres", action="store_true", help="process NRES Spectrographs"
    )

    parser.add_argument(
        "--cpus", type=int, default=1, help="use multiple CPU cores for parallism"
    )

    return parser.parse_args()
