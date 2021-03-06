# -*- coding: utf-8 -*-

"""
Main entry point for the CR labeling pipeline.
`$ bash scripts/reduce_lco_cr.sh` is an example script.
CY Xu (cxu@ucsb.edu)
"""

import os
from functools import partial
import multiprocessing as mp
import astropy.io.fits as fits

try:
    from cosmic_conn.cr_pipeline.lco_cr_reduction import CR_reduction
    from cosmic_conn.cr_pipeline.nres_cr_reduction import NRES_CR_detector
    from cosmic_conn.cr_pipeline.utils_io import align_banzai, is_fits_file
    from cosmic_conn.cr_pipeline.options import argument_parser
except ImportError:
    raise ImportError(
        "Please run `pip install cosmic-conn[develop]` to install all packages for development."
        )


def reduce_LCO_CR_mask(fname, root, opt):
    """
    Takes an hdul of at least 3 consecutive observations from the same night to
    generate a binary CR mask based on a give Signal to Noise Ratio threshold and (optional) a linear CR model

    Outputs a multi-extention fits with 3 SCI frames:
        hdul[1:1+3] are BANZAI reduced SCI frames
        hdul[4] is the valid_mask flags valid CCD area after projection
        hdul[5] is SEP soruce table
        hdul[6:6+3] are CR masks
        hdul[9:9+3] are ignore masks
    """
    fits_path = os.path.join(root, fname)

    try:
        hdul = fits.open(fits_path, memmap=True, lazy_load_hdus=True)
    except:
        with open(opt.log_file, "a") as log:
            msg = f"issue reading {fits_path}\n\n"
            log.write(msg)
            print(msg)

    # use parent directory, avoid duplicates
    opt.root = os.path.split(root)[0]

    exp_time = hdul["SCI"].header["exptime"]

    if exp_time >= opt.min_exptime:
        # init a CR_detector object
        detector = CR_reduction(hdul.copy(), opt)
        ret = detector.generate_CR_masks()

        # if return a string, the detectioned failed and reason is included in the string
        if isinstance(ret, str):
            with open(opt.log_file, "a") as log:
                msg = f"{fname} skipped, reason: {ret}\n\n"
                log.write(msg)
                print(msg)
    else:
        print(f"{fname} skipped, reason: exposure time {exp_time}")

    # free memory
    hdul.close()


def reduce_NRES_CR(fname, root, opt):
    """
    Outputs a combined fits, if the input have 3 frames:
        hdul[0:3] are aligned banzai frames
        hdul[3:6] are corresponding CR masks
    """
    fits_path = os.path.join(root, fname)

    try:
        hdul = fits.open(fits_path, memmap=True, lazy_load_hdus=True)
    except:
        with open(opt.log_file, "a") as log:
            msg = f"issue reading {fits_path}\n"
            log.write(msg)
            print(msg)

    # use parent directory, avoid duplicates
    opt.root = os.path.split(root)[0]

    detector = NRES_CR_detector(hdul.copy(), opt)
    ret = detector.generate_CR_masks()

    # if return a string, the detectioned failed and reason is included in the string
    if isinstance(ret, str):
        with open(opt.log_file, "a") as log:
            msg = f"{fname} skipped, reason: {ret}\n"
            log.write(msg)
            print(msg)

    # free memory
    hdul.close()


if __name__ == "__main__":
    opt = argument_parser()

    # group consecutive ovservations and reproject/align frames
    if not opt.aligned:
        align_banzai(opt)
    """
    after reprojection, read each aligned fits to reduce the CR mask,
    the mask will be appended after the aligned fits and saved as
    a new fits for training, expected final file structure:
        data/banzai_frms
        data/aligned_fits
        data/masked_fits
    """

    assert os.path.exists(opt.data), f"{opt.data} doesn't exist."

    # logging
    log_file = os.path.join(opt.data, "CR_reduction_log.txt")
    opt.log_file = log_file

    # collect already processed to resume work
    processed = []
    for root, dirs, files in os.walk(opt.data):

        # pass fits already detected
        if root.endswith("masked_fits"):
            for f in files:
                if is_fits_file(f):
                    processed.append(f[:30])

        # pass fits already rejected
        if "rejected" in root:
            for f in files:
                processed.append(f[:30])

    # multithread processing
    for root, dirs, files in os.walk(opt.data):
        if root.endswith("aligned_fits"):

            with open(log_file, "a") as log:
                log.write(
                    f"directory: {root}\n"
                    f"SNR thresholds {opt.snr_thres}, {opt.snr_thres_low}\n"
                    f"dilation {opt.dilation}, minimum CR size {opt.min_cr_size}\n"
                    f"{opt.comment}\n\n"
                )

            # avoid temp files generated by Mac's Finder
            files = [f for f in files if not f[0] == "."]
            dirs[:] = [d for d in dirs if not d[0] == "."]
            files.sort()

            files = [f for f in files if is_fits_file(f)]

            files = [f for f in files if f[:30] not in processed]

            assert len(files) > 0, f"found no valid fits files, check {root}"

            if opt.nres:
                for f in files:
                    reduce_NRES_CR(f, root, opt)
                continue

            # use multiple CPU cores to speed up
            if opt.cpus > 1:
                with mp.Pool(processes=opt.cpus) as pool:
                    detect_partial = partial(
                        reduce_LCO_CR_mask, root=root, opt=opt)
                    pool.map(detect_partial, files)
            else:
                for f in files:
                    reduce_LCO_CR_mask(f, root, opt)
