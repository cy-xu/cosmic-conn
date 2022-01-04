# -*- coding: utf-8 -*-

"""
Tools to support consoles commands and app.
CY Xu (cxu@ucsb.edu)
"""

import os
import argparse
import glob
import requests
import zipfile
from pathlib import Path
from cosmic_conn.dl_framework.options import ModelOptions

PYTEST_DATA_URL = "https://sites.cs.ucsb.edu/~cy.xu/cosmic_conn/Cosmic-CoNN_test_data.zip"
MODEL_URL = "https://sites.cs.ucsb.edu/~cy.xu/cosmic_conn/trained_models.zip"

EXTENSIONS = ["fits", "fz"]

def is_fits_file(filename):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in EXTENSIONS)

def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                # if chunk:
                f.write(chunk)
    return local_filename


def download_test_data():
    """ download three banzai reduced FITS for tests
    """
    test_data_path = os.path.join(os.getcwd(), "Cosmic-CoNN_test_data")
    print(f"Downloading Cosmic-CoNN test data to {test_data_path}")
    file_name = download_file(PYTEST_DATA_URL)

    with zipfile.ZipFile(file_name, "r") as zip_ref:
        zip_ref.extractall(test_data_path)

    assert os.path.exists(test_data_path)
    os.remove(file_name)

    return test_data_path


def download_trained_models(model_path):
    print(f"Downloading trained Cosmic-CoNN models to {model_path}")
    model_name = download_file(MODEL_URL)

    with zipfile.ZipFile(model_name, "r") as zip_ref:
        zip_ref.extractall(model_path)

    assert os.path.exists(model_path)
    os.remove(model_name)

    return model_path


def check_trained_models():
    parent_dir = str(Path(__file__).parent.absolute())
    model_dir = os.path.join(parent_dir, "trained_models")
    models = glob.glob(model_dir + "/*.pth")
    if len(models) < 3:
        raise ValueError("models not found in trained_models directory.")
    return model_dir


def console_arguments():
    """Parse console arguments for cosmic_conn."""
    parser = argparse.ArgumentParser(
        description="Cosmic-CoNN console commands to detect CRs "
        "in FITS file(s). CR prediction will be attached "
        "as a FITS extention. A new copy of the FITS "
        "file is saved in cosmic_conn_output."
    )
    parser.add_argument(
        "-a", "--app", action="store_true", help="launch the CR "
        "detector web app"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="./",
        help="input FITS file or directory for CR detection. "
        "Passing a directory will batch detect all FITS files,"
        "including subdirectories.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ground_imaging",
        help="use one of the models: ground_imaging | NRES | HST_ACS_WFC",
    )
    parser.add_argument(
        "-e",
        "--ext",
        type=str,
        default="SCI",
        help="read data from this hdul extension, SCI by default."
    )
    parser.add_argument(
        "-c",
        "--crop",
        type=int,
        default=1024,
        help="slice the image to stamps of this size, 1024 by default."
        "Set to 0 for full image detection, large memory required."
    )

    opt = parser.parse_args()
    opt = vars(opt)

    # swtich to custom class from argparse dict
    model_options = ModelOptions()
    for k, v in opt.items():
        model_options[k] = v

    return model_options


def parse_input(input, PREDICT_DIR):
    all_fits = []

    # sanity check
    assert os.path.exists(input), (
        f"{input} does not exist. Usage: cosmic-conn --help"
    )

    # a single file or files in directory?
    if os.path.isfile(input):
        all_fits.append(input)

    elif os.path.isdir(input):
        # walk through all subdirectories to find all FITS
        for root, dirs, files in os.walk(input):
            if PREDICT_DIR in root:
                continue

            files = [f for f in files if not f[0] == "."]
            dirs[:] = [d for d in dirs if not d[0] == "."]

            # check extention for FITS file
            files = [f for f in files if is_fits_file(f)]
            files = [os.path.join(root, f) for f in files]

            all_fits += files

        print(f"{len(all_fits)} FITS found in {input}")

    assert len(all_fits) > 0, (
        (f"Invalid input {input}. Use -i INPUT to specify "
            "FITS file or a directory path")
    )

    return all_fits


# if __name__ == "__main__":
#     download_test_data()
#     check_trained_models()
