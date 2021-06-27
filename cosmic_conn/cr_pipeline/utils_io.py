# -*- coding: utf-8 -*-

"""
Help functions for data IO
CY Xu (cxu@ucsb.edu)
"""

import os
import numpy as np

from astropy.io import fits
from reproject import reproject_interp

from cosmic_conn.cr_pipeline.utils_img import acquire_valid_area, save_as_png, zscale_stack_cutoff

EXTENSIONS = ["fits", "fz"]


def is_fits_file(filename):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in EXTENSIONS)


def pop_from_list(list, index):
    new_list = []
    for i in range(len(list)):
        if i not in index:
            new_list.append(list[i])
    return new_list


def group_nres_sequences(root, fnames):
    fnames.sort()
    grouped_sequences = []

    # otherwise group frames of the same requst
    sequence = []

    i = 0
    while i < len(fnames):
        i_path = os.path.join(root, fnames[i])

        try:
            with fits.open(i_path) as hdu:
                hdr = hdu["SPECTRUM"].header

                sequence_propid = hdr["PROPID"]
                sequence_exptime = hdr["EXPTIME"]
                sequence_request = hdr["REQNUM"]

                date = hdr["DATE"]
                object_name = hdr["OBJECT"] if hdr["OBJECT"] != "N/A" else "NA"
        except:
            # os.remove(i_path)
            print(f"{i_path} is corrupted and removed")
            i += 1
            continue

        # inner loop of consecutive frames
        end_idx = min(len(fnames), i + 3)
        sequence = []
        bkgs = []

        for j in range(i, end_idx):

            # read current frame and header
            curr_fname = fnames[j]
            path = os.path.join(root, curr_fname)
            print(curr_fname)

            with fits.open(path) as hdu:
                hdr = hdu["SPECTRUM"].header

                proposal_id = hdr["PROPID"]
                request = hdr["REQNUM"]
                exptime = hdr["EXPTIME"]

            # exposure time are not always exact the same between frames
            # 5% difference allowed but avoid large difference in exposure time
            if np.abs(exptime - sequence_exptime) > 0.05 * sequence_exptime:
                print(f"exposure time not matching\n")
                continue

            if request == "UNSPECIFIED":
                print(f"request unspecified\n")
                continue

            # check if following frames are from the same request (in case of missing files)
            if request != sequence_request:
                print(f"request not matching\n")
                continue

            if proposal_id != sequence_propid:
                print(f"proposal id not matching\n")
                continue

            # evertthing checks out, add to list
            sequence.append(curr_fname)

        # dump the sequence to the main list after inner loop
        if len(sequence) == 3:
            grouped_sequences.append(sequence)

            print(
                f"request {sequence_request} took {len(sequence)} frames of {object_name}"
            )
            print(f"on {date[:10]}, exp_time {sequence_exptime}\n")

            # hdus are the "clean" single frames replaced pixels from BPM
            aligned_hdul = new_hdul()

            for f in sequence:
                f_path = os.path.join(root, f)
                with fits.open(f_path) as hdu:
                    extension = hdu["SPECTRUM"]
                    extension.header["maxlin"] = 30000
                    extension.header["saturate"] = 37500
                    extension.header["filename"] = f

                    aligned_hdul.append(extension.copy())

            # update primary header
            aligned_hdul[0].header["frmtotal"] = len(sequence)

            # save stacked frames in one fits
            root_parent = os.path.split(root)[0]
            # stacked_root = os.path.join(root_parent, 'aligned_nres')
            save_as_fits(aligned_hdul, sequence[0], root_parent)

        # move forward frame_count steps in the outer loop
        # avoid infinite loop, skip ahead one frame
        if len(sequence) == 0:
            i += 1
        else:
            i += len(sequence)

        if i >= len(fnames):
            break

    return grouped_sequences


def group_consecutive_sequences(root, fnames, min_exptime):
    fnames.sort()
    grouped_sequences = []

    # otherwise group frames of the same requst
    sequence = []

    i = 0
    while i < len(fnames):
        i_path = os.path.join(root, fnames[i])

        try:
            with fits.open(i_path) as hdu:
                hdr = hdu["SCI"].header

                sequence_frmtotal = hdr["FRMTOTAL"]
                sequence_exptime = hdr["EXPTIME"]
                sequence_request = hdr["REQNUM"]
                sequence_molecule = hdr["MOLNUM"]

                date = hdr["DATE"]
                object_name = hdr["OBJECT"] if hdr["OBJECT"] != "N/A" else "NA"

                # some fits misses WCSERR key
                try:
                    wcserr = hdr["WCSERR"]
                except:
                    wcserr = 1
        except:
            os.remove(i_path)
            print(f"{i_path} is corrupted and removed")
            i += 1
            continue

        # keep only consecutive sequence
        if sequence_frmtotal < 3 or sequence_frmtotal > 12:
            os.remove(i_path)
            print(f"{fnames[i]} removed, frame count {sequence_frmtotal}\n")
            i += 1
            continue

        # reject short-exposed frames
        elif sequence_exptime < min_exptime:
            i += 1
            print(f"exposure less than {min_exptime}")
            continue

        # reject WCS unresolved frames
        elif wcserr != 0:
            os.remove(i_path)
            i += 1
            print("wcs not resolved")
            continue

        else:
            # 0m4 telescope shutter tend to open before telescope stablize
            # leading to blurry first exposure, skip 1st frame if possible
            if sequence_frmtotal > 3 and fnames[i][3:6] == "0m4":
                i += 1

            # inner loop of consecutive frames
            end_idx = min(len(fnames), i + 3)
            sequence = []
            bkgs = []

            for j in range(i, end_idx):

                # read current frame and header
                curr_fname = fnames[j]
                path = os.path.join(root, curr_fname)
                print(curr_fname)

                with fits.open(path) as hdu:
                    hdr = hdu["SCI"].header
                    request = hdr["REQNUM"]
                    exptime = hdr["EXPTIME"]
                    molecule = hdr["MOLNUM"]

                    try:
                        bkgs.append(float(hdr["L1MEDIAN"]))
                    except:
                        os.remove(path)
                        fnames.pop(j)
                        print(f"L1MEDIAN not available\n")
                        continue

                # exposure time are not always exact the same between frames
                # 5% difference allowed but avoid large difference in exposure time
                if np.abs(exptime - sequence_exptime) > 0.05 * sequence_exptime:
                    print(f"exposure time not matching\n")
                    continue

                # check if following frames are from the same request (in case of missing files)
                if request != sequence_request:
                    print(f"request not matching\n")
                    continue

                # same request could split into different molecules, should avoid
                if molecule != sequence_molecule:
                    print(
                        f"molecule {molecule}, sequence molecule {sequence_molecule}\n"
                    )
                    continue

                # evertthing checks out, add to list
                sequence.append(curr_fname)

                # dump the sequence to the main list after inner loop
                if len(sequence) == 3:

                    # test sample standard deviation between frames
                    # s = sqrt((1/(N-1)sum(x_i-mu)^2))
                    # for numpy, ddof makes it unbiased std

                    bkgs = np.array(bkgs) - np.median(bkgs)
                    std = np.std(bkgs, ddof=1)

                    if std < 5.0:
                        grouped_sequences.append(sequence)

                        print(
                            f"request {round(float(sequence_request))}",
                            f"took {len(sequence)} frames of {object_name}",
                            f"on {date[:10]}, exp_time {sequence_exptime},",
                            f"molecule {sequence_molecule}\n",
                        )

                        # reset after 3 frames
                        # sequence_copy = copy.deepcopy(sequence)
                    else:
                        print(curr_fname)
                        print(f"bkg variation {std} too too high\n")

            # move forward frame_count steps in the outer loop
            # avoid infinite loop, skip ahead one frame
            if len(sequence) == 0:
                i += 1
            else:
                i += len(sequence)

        if i >= len(fnames):
            break

    return grouped_sequences


def extract_fname(file_path):
    filename = os.path.split(file_path)[-1]
    if filename.endswith(".fz"):
        return filename[:-8]
    elif filename.endswith(".fitz"):
        return filename[:-5]
    else:
        return os.path.splitext(filename)[0]


def new_hdul():
    hdr = fits.Header()
    hdr["aligned"] = False
    hdr["frmtotal"] = 0
    empty_primary = fits.PrimaryHDU(header=hdr)
    return fits.HDUList([empty_primary])


def save_file_name_to_header(hdul, fpath):
    filename = extract_fname(fpath)
    hdul["SCI"].header["filename"] = filename
    return hdul


def fix_bad_ccd_saturated_pixels(hdul):
    mask = hdul["BPM"].data
    # bad pixels
    median = np.median(hdul["SCI"].data)
    hdul["SCI"].data[mask == 1] = median
    hdul["SCI"].data[mask == 3] = median

    # over-saturated pixels
    saturate = hdul["SCI"].header["saturate"]
    upper_bound = 0.9 * saturate
    hdul["SCI"].data[mask == 2] = upper_bound

    return hdul


def remove_nan(frame):
    # replace Nan value with 0
    nan_index = np.isnan(frame)
    frame[nan_index] = 0.0
    return frame


def copy_WCS(reference_hdu, query_hdu):
    # after projection, replace WCS to the target WCS
    ref_hdr = reference_hdu.header
    keys = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

    for k in keys:
        query_hdu.header[k] = ref_hdr[k]

    return query_hdu


def reproject_to_same_WCS(fits_paths, root, no_png=False):
    """
    Takes N consecutive frames as input, project to middle frame's WCS
    remove bad pixels from banzai's mask, combine to a single new fits  
    return a combined fits, where masked_fits[1:N+1] are reprojected frames 
    """
    # Use middle image as the target for alignment
    N_frames = len(fits_paths)
    reference_idx = N_frames // 2

    # hdus are the "clean" single frames replaced pixels from BPM
    hdus = []

    for i in range(len(fits_paths)):

        path = fits_paths[i]
        # pre-process and copy the frame
        hdul = fits.open(path, lazy_load_hdus=False)
        hdul = fix_bad_ccd_saturated_pixels(hdul)
        hdul = save_file_name_to_header(hdul, path)
        hdus.append(hdul["SCI"].copy())

        # save CAT extension, the extracted sources
        if i == reference_idx:
            cat_extension = hdul["CAT"].copy()

        hdul.close()

    reference_hdu = hdus[reference_idx]
    aligned_hdul = new_hdul()
    valid_masks = []

    for i in range(N_frames):
        hdus[i].ver = i
        query_hdu = hdus[i]

        if i == reference_idx:
            # append the reference frame directly, no reprojection needed
            aligned_hdul.append(query_hdu)

        else:
            # project query_hdu to WCS of the reference_hdu (the middle frame)
            aligned_array, footprint = reproject_interp(
                query_hdu, reference_hdu.header, order="nearest-neighbor"
            )

            aligned_array = remove_nan(aligned_array)
            query_hdu.data = np.float32(aligned_array)

            query_hdu = copy_WCS(reference_hdu, query_hdu)
            aligned_hdul.append(query_hdu)

            # save footprint mask for future use
            valid_masks.append(np.array(footprint > 0).astype("uint8"))

    # from all reprojection footprint masks, get common valid area across all frames
    valid_mask = acquire_valid_area(np.stack(valid_masks))

    aligned_hdul.append(fits.CompImageHDU(
        valid_mask, name="valid_mask", uint=True))

    # append CAT extention, SEP extracted sources
    aligned_hdul.append(cat_extension)

    # update primary header
    aligned_hdul[0].header["frmtotal"] = N_frames

    # save png preview to quickly review/remove bad frames
    if not no_png:
        contrast = 0.1
        hdu_vis = np.stack([hdu.data for hdu in aligned_hdul[1:-1]])
        hdu_vis, _ = zscale_stack_cutoff(
            hdu_vis, nsamples=10_000, contrast=contrast)

        out_path = os.path.join(os.path.split(root)[0], "aligned_png")

        for i in range(N_frames):
            fname = extract_fname(fits_paths[i])
            save_as_png(hdu_vis[i], out_path, fname, file_type="noisy")

    return aligned_hdul


def align_banzai(opt):
    """
    Read banzai processed frames, reproject a consecutive sequence of frames 
    to the same WCS, and combine multiple fits into a single fits

    expected directory structure:
        data/banzai_frms
        data/telescope_a/banzai_frms
        data/telescope_b/banzai_frms

    new fits will be saved to 'aligned_fits' directory in next to the source banzai_frms
    """
    for root, dirs, files in os.walk(opt.data):

        # avoid temp files generated by Mac's Finder
        files = [f for f in files if not f[0] == "."]
        dirs[:] = [d for d in dirs if not d[0] == "."]

        fitz_only = [fit for fit in files if is_fits_file(fit)]
        fitz_only.sort()

        if not root.endswith("banzai_frms"):
            continue
        print(f"{len(fitz_only)} fits files found in {root}\n")

        # group consecutive observations of same object from the same night
        if opt.nres:
            grouped_sequences = group_nres_sequences(root, fitz_only)

        else:
            grouped_sequences = group_consecutive_sequences(
                root, fitz_only, opt.min_exptime
            )

            for i in range(len(grouped_sequences)):

                # each group is N (N>=3) frames of consecutive observations
                # reproject all frames to align with the middle frame's WCS
                fits_paths = [os.path.join(root, f)
                              for f in grouped_sequences[i]]

                aligned_hdul = reproject_to_same_WCS(
                    fits_paths, root, opt.no_png)

                # save aligned frames in one fits
                root_parent = os.path.split(root)[0]
                save_as_fits(
                    aligned_hdul, grouped_sequences[i][0], root_parent)


def save_as_fits(aligned_fits, fname, dir, mask=False):
    # new file name same as the first file of the sequence
    # currently fixed to merge three consecutive frames
    frame_count = 3

    if mask:
        new_path = os.path.join(dir, "masked_fits")
        new_filename = f"{remove_ext(fname)}_{frame_count}frms_masks.fits"
    else:
        new_path = os.path.join(dir, "aligned_fits")
        new_filename = f"{remove_ext(fname)}_{frame_count}frms_aligned.fits"

    # save aligned fits to sub-directory
    mkdir(new_path)
    out_path = os.path.join(new_path, new_filename)

    aligned_fits.writeto(out_path, overwrite=True)


def read_aligned_fits(directory):
    file_names, roots = [], [], []

    for root, dirs, files in os.walk(directory):
        if root.endswith("aligned_fits"):
            # avoid temp files generated by Mac's Finder
            file_names = [f for f in files if not f[0] == "."]
            dirs[:] = [d for d in dirs if not d[0] == "."]

            file_names.sort()
            roots.append(root)

    return roots, file_names


def remove_outlier_frm(frames, outlier_idx, update_hdul=False):
    if update_hdul:
        # for hdul update, need to remove frame and valid mask
        total = frames[0].header["frmtotal"]

        # hdul[0] is primary frame, skip
        hdul_idx = [x + 1 for x in outlier_idx]
        frames = np.delete(frames, hdul_idx, axis=0)

        # update the count of total frames in headers
        for i in range(len(frames)):
            frames[i].header["frmtotal"] = total - len(outlier_idx)

        frames = fits.HDUList(list(frames))

    else:
        frames = np.delete(frames, outlier_idx, axis=0)

    return frames


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_ext(filename):
    if filename.endswith(".fz"):
        return filename[:-8]
    elif filename.endswith(".fitz"):
        return filename[:-5]
    else:
        return os.path.splitext(filename)[0]


def save_fits(array, dir, filename, file_type):
    fname = filename + file_type + ".fitz"
    out_path = os.path.join(dir, fname)
    hud = fits.PrimaryHDU(array)
    hud.writeto(out_path)
    print(f"{file_type} saved to {out_path}")


def save_as_npy(array, dir, filename, file_type):
    mkdir(dir)
    path = os.path.join(dir, filename + "_" + file_type)
    if file_type == "mask":
        array = array.astype("uint8")
    np.save(path, array, allow_pickle=True)


def save_array_png(
    noisy, mask, cleaned, ignore_mask, root, fnames, save_np, save_png, header, comment
):
    # save frames as numpy arrays
    if save_np:
        npy_dir = os.path.join(root, "train_npy")
        mkdir(npy_dir)

        filename = os.path.join(npy_dir, fnames[0])
        np.savez_compressed(filename, noisy=noisy, mask=mask)

    if save_png:
        png_dir = comment if comment is not None else "train_set_png"
        png_dir = "rejected_frms" if "reject" in comment else png_dir

        img_dir = os.path.join(root, png_dir)

        obj = header["OBJECT"]
        exptime = header["EXPTIME"]
        rdnoise = header["RDNOISE"]
        put_text = f"object:{obj} / exposure:{exptime} / RD_noise:{rdnoise}"

        noisy_vis, cut_offs = zscale_stack_cutoff(
            noisy, nsamples=10_000, contrast=0.1)

        if cleaned is not None:
            clean_vis, _ = zscale_stack_cutoff(
                cleaned, nsamples=10_000, contrast=0.1, cut_offs=cut_offs
            )

        for i in range(noisy.shape[0]):
            fname = fnames[i] + "_" + \
                comment if comment is not None else fnames[i]

            save_as_png(noisy_vis[i], img_dir, fname, "noisy", zscale=False)

            if mask is not None:
                save_as_png(mask[i], img_dir, fname, "mask")

            if cleaned is not None:
                save_as_png(clean_vis[i], img_dir, fname,
                            "cleaned", zscale=False)

        # save single sources mask
        if ignore_mask is not None:
            for j in range(ignore_mask.shape[0]):
                save_as_png(
                    ignore_mask[j], img_dir, fnames[0] +
                    "_ignore_mask", "ignore_mask",
                )


def save_fits_with_CR(opt, hdul, gains, cr_mask, ignore_mask, filenames, debug_frames):
    N_frames = cr_mask.shape[0]

    if opt.debug:
        mask_high, mask_low, cr_signal_to_noise = debug_frames

    for j in range(N_frames):
        # update frame read_noise with new
        hdul[j + 1].header["gain"] = gains[j].item()

        # append CR mask
        hdul.append(
            fits.CompImageHDU(
                cr_mask[j], header=hdul[j + 1].header, name="CR", ver=j, uint=True
            )
        )

        # append debugging frames
    if opt.debug:
        for j in range(N_frames):
            hdul.append(
                fits.CompImageHDU(
                    mask_high[j],
                    header=hdul[j + 1].header,
                    name="mask_high",
                    ver=j,
                    uint=True,
                )
            )
        for j in range(N_frames):
            hdul.append(
                fits.CompImageHDU(
                    cr_signal_to_noise[j],
                    header=hdul[j + 1].header,
                    name="CR_SNR",
                    ver=j,
                )
            )

    # append source mask
    for j in range(ignore_mask.shape[0]):
        hdul.append(
            fits.CompImageHDU(
                ignore_mask[j], header=hdul[j + 1].header, name="ignore", uint=True
            )
        )

    # save as new fits with masks, inside "masked_fits" directory
    save_as_fits(hdul, filenames[0], opt.root, mask=True)


def hdul_to_array(fits):
    # read frames from hdul to ndarrays
    frmtotal = fits[0].header["frmtotal"]

    frames = np.stack([fits[i].data for i in range(1, frmtotal + 1)])

    # valid mask is the common valid area across all frames
    valid_mask = fits[-2].data
    frames *= valid_mask

    filenames = [fits[i].header["filename"] for i in range(1, frmtotal + 1)]
    gains = [fits[i].header["gain"] for i in range(1, frmtotal + 1)]

    return frames, valid_mask, gains, filenames
