#!/usr/bin/env python

"""Tests for `cosmic_conn` package."""

import os
import shutil
import numpy as np
from astropy.io import fits

from cosmic_conn.inference_cr import init_model
from cosmic_conn.data_utils import download_test_data, is_fits_file


PREDICT_DIR = "cosmic_conn_output"
TEST_FILE = download_test_data()
DATA_PATH = os.path.join(TEST_FILE, "banzai_frms")


def inference_on_test_data(opt, model):
    for root, dirs, files in os.walk(opt.input):
        if PREDICT_DIR in root:
            continue

        files = [f for f in files if is_fits_file(f)]

        for f in files:
            with fits.open(os.path.join(root, f)) as hdul:
                frame = hdul["SCI"].data.astype("float32")
                pdt_cosmic = model.detect_cr(frame, ret_numpy=True)
                assert np.sum(pdt_cosmic) > 0.0
                assert np.max(pdt_cosmic) <= 1.0
                assert np.min(pdt_cosmic) >= 0.0

                hdul.append(fits.CompImageHDU(
                    pdt_cosmic, name=f"CR_probability"))

                # a new copy is saved in the prediction directory
                out_dir = os.path.join(root, PREDICT_DIR)
                os.makedirs(out_dir, exist_ok=True)

                fname = os.path.join(out_dir, f"CR_{f}")
                hdul.writeto(fname, overwrite=True)

    # clean up after test
    shutil.rmtree(DATA_PATH)


class Test_model_inference:
    def test_lco_imaging(self):
        model = init_model("ground_imaging")
        model.opt.input = DATA_PATH
        print(f"trained models loaded from {model.opt.load_model}")

        inference_on_test_data(model.opt, model)
