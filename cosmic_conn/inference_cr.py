# -*- coding: utf-8 -*-

"""
inference_cr.py: main entry file for console commands, app, and package import.
See documentation for usage.
CY Xu (cxu@ucsb.edu)
"""

import os
import sys
import time
import shutil
import logging
from tqdm import tqdm
import pretty_errors

import torch.backends.cudnn as cudnn
from astropy.io import fits

from cosmic_conn.dl_framework.cosmic_conn import Cosmic_CoNN
from cosmic_conn.dl_framework.options import ModelOptions
from cosmic_conn.data_utils import check_trained_models, console_arguments
from cosmic_conn.data_utils import parse_input

cudnn.enabled = True
cudnn.benchmark = True

PREDICT_DIR = "cosmic_conn_output"
TEMP_DIR = "instance_temp_storage"
MODEL_VERSON = "0.2.8"


def init_model(model, opt=None):
    """Models are initialized here by passing in the keyword.
    return a pytorch model.
    """
    model_dir = check_trained_models()

    if opt is None:
        opt = ModelOptions()

    # stdout for CLI users
    if opt.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # overwrite model if init_model is called directly
    opt.model = model

    # logging
    msg = f"Initializing Cosmic-CoNN CR detection {opt.model} model...\n"
    logging.info(msg)

    for k, v in opt.__dict__.items():
        logging.debug(f"{k}: {v}")

    # model path
    if opt.model == "ground_imaging":
        opt.load_model = f"{model_dir}/cosmic-conn_ground_imaging_v{MODEL_VERSON}.pth"

    elif opt.model == "NRES":
        opt.load_model = f"{model_dir}/cosmic-conn_LCO_NRES_v{MODEL_VERSON}.pth"

    elif opt.model == "HST_ACS_WFC":
        opt.norm = "batch"
        opt.load_model = f"{model_dir}/cosmic-conn_HST_ACS_WFC_v{MODEL_VERSON}.pth"

    elif opt.model.endswith("tar") or opt.model.endswith("pth"):
        opt.load_model = opt.model

    else:
        msg = "-m [ground_imaging | NRES | HST_ACS_WFC | model_file(.pth/.tar)]"
        logging.error(msg)
        raise ValueError(msg)

    # init model instance
    cr_model = Cosmic_CoNN()
    cr_model.initialize(opt)
    cr_model.eval()

    msg = f"{opt.model} model initialized.\n"
    logging.info(msg)

    return cr_model


def detect_FITS(model):
    # get arguments from user
    options = model.opt
    all_fits = parse_input(options.input, PREDICT_DIR)
    ext = model.opt.ext

    # start batch detection
    for i in tqdm(range(len(all_fits))):
        f = all_fits[i]

        with fits.open(f) as hdul:
            image = None
            tic = time.perf_counter()

            try:
                if ext != 'SCI':
                    msg = f"Reading data from hdul[{ext}]"
                    # if user asigned extension, try it first
                    image = hdul[ext].data.astype("float32")

                elif hdul[0].data is not None:
                    msg = "Reading data from hdul[0]"
                    image = hdul[0].data.astype("float32")

                elif hdul[1].data is not None:
                    msg = "Reading data from hdul[1]"
                    image = hdul[1].data.astype("float32")

                else:
                    msg = f"Reading data from hdul[{ext}]"
                    image = hdul[ext].data.astype("float32")

                logging.info(msg)

            except:
                msg = f"No valid data found in extention 0, 1 or {ext}, \
                    to specify extension name: -e SCI."
                logging.error(msg)
                raise ValueError(msg)

            # detection
            cr_probability = model.detect_cr(image, ret_numpy=True)

            hdul.append(fits.CompImageHDU(
                cr_probability, name="CR_probability"))

            # a new copy is saved in the prediction directory
            out_dir = os.path.join(os.path.split(f)[0], PREDICT_DIR)
            os.makedirs(out_dir, exist_ok=True)

            out_name = os.path.join(out_dir, f"CR_{os.path.basename(f)}")
            hdul.writeto(out_name, overwrite=True)

            toc = time.perf_counter()
            msg = f"Detection of a {image.shape} image took {round(toc-tic, 2)}s."
            msg1 = f"Result saved as {os.path.basename(out_name)}"
            logging.info(msg)
            logging.info(msg1)

    msg = f"\nDone. The CR probability map is appended to a new FITS copy and saved in {PREDICT_DIR}"
    logging.info(msg)

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    return 0


def detect_image(cr_model, image, ret_numpy=True):
    cr_mask = cr_model.detect_cr(image.astype("float32"), ret_numpy=ret_numpy)
    return cr_mask


def CLI_entry_point():
    # entry point form CLI detection or web app
    opt = console_arguments()

    # print out logs in CLI, otherwise only logging
    opt.verbose = True

    cr_model = init_model(opt.model, opt)

    if opt.app:
        # launch web-app
        from cosmic_conn.web_app import app        
        app.main(cr_model, opt)
    else:
        # batch processing FITS
        detect_FITS(cr_model)
