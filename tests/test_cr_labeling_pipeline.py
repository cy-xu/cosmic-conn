#!/usr/bin/env python

"""Tests for `cosmic_conn` package."""

import os
import shutil
import argparse

from astropy.io import fits


from cosmic_conn.data_utils import download_test_data
from cosmic_conn.cr_pipeline.utils_io import *
from cosmic_conn.reduce_cr import reduce_LCO_CR_mask

DATA_PATH = download_test_data()


def options():
    # init mock parameters for test
    opt = argparse.Namespace()
    opt.data = DATA_PATH
    opt.aligned = False
    opt.nres = False
    opt.snr_thres = 5
    opt.snr_thres_low = 2.5
    opt.dilation = 5
    opt.flood_fill = False
    opt.min_exptime = 99.0
    opt.min_cr_size = 2
    opt.cpus = 1
    opt.no_png = True
    opt.debug = False
    opt.comment = "test_case"

    log_file = os.path.join(opt.data, "CR_reduction_log.txt")
    opt.log_file = log_file

    return opt


class Test_CR_Labeling_Pipeline:
    def test_reproject(self):
        opt = options()
        assert os.path.exists(opt.data), "test_data directory doesn't exist"

        align_banzai(opt)

        for root, dirs, files in os.walk(opt.data):
            if root.endswith("aligned_fits"):
                for f in files:
                    with fits.open(os.path.join(root, f)) as hdu:
                        assert hdu[0].header["frmtotal"] > 0

    def test_cr_reduction(self):
        opt = options()

        for root, dirs, files in os.walk(opt.data):
            if root.endswith("aligned_fits"):
                for f in files:
                    reduce_LCO_CR_mask(f, root, opt)

        for root, dirs, files in os.walk(opt.data):
            if root.endswith("masked_fits"):
                for f in files:
                    with fits.open(os.path.join(root, f)) as hdu:
                        cr_mask = hdu["CR"].data
                        assert np.sum(cr_mask) > 0

        # clean up after test
        # shutil.rmtree(DATA_PATH)
