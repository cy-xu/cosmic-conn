# -*- coding: utf-8 -*-

"""
Helper functions for model evaluation and figure plotting.
CY Xu (cxu@ucsb.edu)
"""

import os
import torch
import numpy as np
import random
from astropy.io import fits

from skimage.morphology import dilation, square
from sklearn.metrics import precision_recall_curve, roc_curve

from cosmic_conn.dl_framework.cosmic_conn import Cosmic_CoNN
from cosmic_conn.dl_framework.utils_ml import modulus_boundary_crop, clean_large, tensor2np
from cosmic_conn.cr_pipeline.utils_img import save_as_png, erase_boundary_np, subtract_sky


# modified from https://gist.github.com/JDWarner/6730747
def calculate_dice_score(prediction, reference, threshold=0.5, dilate_1px=False):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    eps = 1
    pdt = np.asarray(prediction > threshold).astype(np.bool)
    ref = np.asarray(reference > threshold).astype(np.bool)

    if dilate_1px:
        pdt = dilation(pdt, square(3))

    if pdt.shape != ref.shape:
        raise ValueError(
            "Shape mismatch: prediction and reference must have the same shape."
        )

    # Compute Dice coefficient
    intersection = np.logical_and(pdt, ref)

    return 2.0 * intersection.sum() / (pdt.sum() + ref.sum() + eps)


def maskMetric(PD, GT):
    if len(PD.shape) == 2:
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(GT.shape[0]):
        P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return np.array([TP, TN, FP, FN])


def maskMetric_DL(prediction, reference, thresholds=None, dialate_1px=False):
    if thresholds is None:
        thresholds = np.logspace(0, 10, num=50, base=2) / (2 ** 10)
        thresholds = np.pad(thresholds, (1, 0), "constant",
                            constant_values=(0.0, 1.0))

    metrics = np.zeros([len(thresholds), 4])

    selem = square(3)

    assert (
        prediction.shape == reference.shape
    ), f"Shapes {prediction.shape} and {reference.shape} mismatch"

    ref = reference.astype("bool")

    for i, thres in enumerate(thresholds):
        TP, TN, FP, FN = 0, 0, 0, 0
        pdt = prediction > thres

        if dialate_1px:
            pdt = dilation(pdt, selem)

        TP = (pdt[ref == 1] == 1).sum()
        TN = (pdt[ref == 0] == 0).sum()
        FP = (pdt[ref == 0] == 1).sum()
        FN = (pdt[ref == 1] == 0).sum()

        metrics[i] = np.array(np.array([TP, TN, FP, FN]))

    return metrics, thresholds


def HST_evaluate_and_save_masks(model, dset, model_name, limit, metrics_path, data_type):
    total_samples = min(limit, len(dset))

    dice_thres = np.linspace(0, 1, 51)
    dice_scores = np.zeros((len(dice_thres), 1))

    predicted_masks = []
    label_masks = []

    for i in range(total_samples):
        raw, mask, badmask, clean, filename = dset[i]

        raw = np.squeeze(raw)
        mask = np.squeeze(mask)
        badmask = np.squeeze(badmask)

        # random rotation and mirror flip
        rotation = random.randint(0, 3)
        flip = random.randint(0, 2)

        # rotate the same patch of sky for data augmentation
        raw = np.rot90(raw, rotation, axes=(-2, -1))
        mask = np.rot90(mask, rotation, axes=(-2, -1))
        badmask = np.rot90(badmask, rotation, axes=(-2, -1))

        # flip for data augmentation
        if flip == 0:
            raw = np.flipud(raw)
            mask = np.flipud(mask)
            badmask = np.flipud(badmask)
        if flip == 1:
            raw = np.fliplr(raw)
            mask = np.fliplr(mask)
            badmask = np.fliplr(badmask)

        if "deepCR" == model_name:
            pdt_mask = model.clean(raw, binary=False, inpaint=False)
        else:
            pdt_mask = model.detect_cr(raw.astype("float32"), ret_numpy=True)

        # ignore mask correction
        pdt_mask = pdt_mask * (1 - badmask)
        mask = mask * (1 - badmask)

        predicted_masks.append(pdt_mask)
        label_masks.append(mask)

        for j, thres in enumerate(dice_thres):
            dice_scores[j] += calculate_dice_score(
                pdt_mask, mask, threshold=thres
            )

        print(f'processed {i}/{total_samples}...')

    print(f"{total_samples} {data_type} images evaluted")
    dice_scores /= total_samples

    # use correct plotting steps
    predicted_masks = np.array(predicted_masks).ravel()
    label_masks = np.array(label_masks).ravel()
    print(f"calculate metrics ...")

    fpr, tpr, thres = roc_curve(label_masks, predicted_masks)
    precision, recall, thresholds = precision_recall_curve(
        label_masks, predicted_masks)

    os.makedirs(os.path.split(metrics_path)[0], exist_ok=True)

    np.savez_compressed(
        metrics_path,
        fpr=fpr,
        tpr=tpr,
        thres=thres,
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        dice_scores=dice_scores,
        dice_thres=dice_thres
    )

    print(f"evaluated metrics saved to {metrics_path}")

    return [fpr, tpr, thres, precision, recall, thresholds, dice_scores, dice_thres]


# @profile
def LCO_evaluate_and_save_masks(model, valid_set, name=None, opt=None):
    model.eval()

    patch = 1024
    boundary = 0
    total_fits = len(valid_set)

    dice_thres = np.linspace(0, 1, 51)
    dice_scores = np.zeros((len(dice_thres), 1))

    predicted_masks = np.array([], dtype="float32")
    label_masks = np.array([], dtype="uint8")
    # dice_scores = []
    # dice_scores_dlt = []
    vis_dir = os.path.join(opt.expr_dir, name + "_Result_Vis")

    with torch.no_grad():
        for i, f in enumerate(valid_set):
            print(f"{i}/{total_fits} fits tested")

            with fits.open(f) as hdul:
                subset = range(3)
                frames = [hdul[1 + i].data for i in subset]
                mask_ref = [hdul[3 + 3 + i].data for i in subset]
                mask_ignore = [hdul[3 + 3 + 3 + i].data for i in subset]
                filenames = [hdul[i + 1].header["FILENAME"] for i in subset]

            for j in range(len(frames)):

                frm = modulus_boundary_crop(frames[j])
                msk_ref = modulus_boundary_crop(mask_ref[j])
                msk_ignore = modulus_boundary_crop(mask_ignore[j])
                fname = filenames[j]

                if opt.rotation:
                    # random rotation and mirror flip
                    rotation = random.randint(0, 3)
                    flip = random.randint(0, 2)

                    # rotate the same patch of sky for data augmentation
                    frm = np.rot90(frm, rotation, axes=(-2, -1))
                    msk_ref = np.rot90(msk_ref, rotation, axes=(-2, -1))
                    msk_ignore = np.rot90(msk_ignore, rotation, axes=(-2, -1))

                    # flip for data augmentation
                    if flip == 0:
                        frm = np.flipud(frm)
                        msk_ref = np.flipud(msk_ref)
                        msk_ignore = np.flipud(msk_ignore)
                    if flip == 1:
                        frm = np.fliplr(frm)
                        msk_ref = np.fliplr(msk_ref)
                        msk_ignore = np.fliplr(msk_ignore)

                # over-lapping crop and evaluate
                # msk_pdt = clean_large(
                #     frm, model, patch=patch, overlap=boundary, ret_numpy=True
                # )

                msk_pdt = model.detect_cr(frm, ret_numpy=True)

                # without ignore mask will lead to degrated test performance for this data set
                msk_ignore = (msk_ignore > 0).astype("float32")
                msk_pdt = msk_pdt * (1 - msk_ignore)
                msk_ref = msk_ref * (1 - msk_ignore)

                for t, thres in enumerate(dice_thres):
                    dice_scores[t] += calculate_dice_score(
                        msk_pdt, msk_ref, threshold=thres
                    )

                predicted_masks = np.append(predicted_masks, msk_pdt.ravel())
                label_masks = np.append(
                    label_masks, msk_ref.ravel().astype("uint8"))

                if not opt.vis:
                    continue

                save_as_png(
                    frm, vis_dir, fname, file_type="2_noisy", zscale=True, grid=True,
                )
                save_as_png(
                    msk_pdt, vis_dir, fname, file_type="3_mask_pdt", grid=True,
                )
                save_as_png(
                    msk_ref, vis_dir, fname, file_type="4_mask_ref", grid=True,
                )
                save_as_png(
                    msk_ignore, vis_dir, fname, file_type="5_mask_ignore", grid=True,
                )

    total = total_fits * len(frames)
    dice_scores /= total

    return [predicted_masks, label_masks, dice_scores, dice_thres]


# @profile
def NRES_evaluate_and_save_masks(model, valid_set):
    model.eval()

    boundary = 100
    sample_total = len(valid_set) * 3

    dice_thres = np.linspace(0, 1, 51)
    dice_scores = np.zeros((len(dice_thres), 1))

    predicted_masks = np.array([], dtype="float32")
    label_masks = np.array([], dtype="uint8")

    with torch.no_grad():
        for fit in valid_set:
            print(fit)

            frames = []
            mask_ignore = []
            mask_gt = []
            filenames = []
            hdul = fits.open(fit)

            for i in range(3):
                frames.append(hdul[i + 1].data)
                mask_gt.append(hdul[i + 1 + 3].data)
                mask_ignore.append(hdul[i + 1 + 3 + 3].data)
                filenames.append(hdul[i + 1].header["filename"])

            hdul.close()

            frames = np.stack(frames).astype("float32")
            mask_ignore = np.stack(mask_ignore).astype("uint8")

            overscan = frames[:, :, -35:]
            frames -= np.median(overscan, axis=(1, 2), keepdims=True)

            for j in range(frames.shape[0]):

                # try whole image detections
                msk_pdt = model.detect_cr(frames[j], ret_numpy=True)

                # remove boundary for consistency
                msk_pdt = erase_boundary_np(msk_pdt, boundary)
                msk_ref = erase_boundary_np(mask_gt[j], boundary)

                msk_pdt = msk_pdt * (1 - mask_ignore[j])
                msk_ref = msk_ref * (1 - mask_ignore[j])

                for t, thres in enumerate(dice_thres):
                    dice_scores[t] += calculate_dice_score(
                        msk_pdt, msk_ref, threshold=thres
                    )

                predicted_masks = np.append(predicted_masks, msk_pdt.ravel())
                label_masks = np.append(
                    label_masks, msk_ref.ravel().astype("uint8"))

                # img_path = f"./temp_NRES_vis/test_set"
                # os.makedirs(img_path, exist_ok=True)
                # fname = filenames[j]
                # save_as_png(msk_pdt, img_path, fname, "mask_pdt", grid=True)
                # save_as_png(msk_ref, img_path, fname, "mask_ref", grid=True)
                # save_as_png(frames[j], img_path, fname,
                #             "noisy", zscale=True, grid=True)

    dice_scores /= sample_total

    return [predicted_masks, label_masks, dice_scores, dice_thres]


def prep_valid_data_numpy(data, source):
    if source == "lco":
        frames, masks, masks_ignore, filenames, read_noise = data
    else:
        frames, masks, masks_ignore, medians, filenames = data

    frms, h, w = frames.shape

    if isinstance(filenames[0], list):
        filenames = [item for sublist in filenames for item in sublist]

    # dump to gpu not
    frames = torch.tensor(frames).view(-1, h, w)
    masks = torch.tensor(masks).view(-1, h, w)
    masks_ignore = torch.tensor(masks_ignore).view(-1, h, w)
    # in case of HST data, source mask is bad_mask

    frames = subtract_sky(frames, remove_negative=False)

    if source == "lco":
        medians = torch.median(frames, dim=-3)[0].view(-1, h, w)
    else:
        medians = torch.tensor(medians).view(-1, h, w)

    frames = tensor2np(frames)
    masks = tensor2np(masks)
    medians = tensor2np(medians)
    masks_ignore = tensor2np(masks_ignore)

    return frames, masks, medians, masks_ignore, filenames, read_noise


def evaluate_LCO_model_metrics(opt, model_name, valid_set, nres=False):

    # read or calculate comparision resutls
    if nres:
        metrics_path = f"{opt.expr_dir}/metrics_{opt.out_name}_NRES.npz"
    else:
        metrics_path = f"{opt.expr_dir}/metrics_{opt.out_name}.npz"

    if os.path.isfile(metrics_path):

        metrics = np.load(metrics_path, allow_pickle=True)

        precision = metrics["precision"]
        recall = metrics["recall"]
        thresholds = metrics["thresholds"]

        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        thres = metrics["thres"]

        dice_scores = metrics["dice_scores"]
        dice_thres = metrics["dice_thres"]

        print(f"loaded {model_name} evaluated metrics")

        return [
            precision,
            recall,
            thresholds,
            fpr,
            tpr,
            thres,
            dice_scores,
            dice_thres,
        ]

    else:
        # init models
        model = Cosmic_CoNN()
        model.initialize(opt)

        print(f"evaluating frames with {model_name}")

        if nres:
            pdt_metrics = NRES_evaluate_and_save_masks(model, valid_set)

        else:
            # calculate metrics for two variants of deepCR-mask
            pdt_metrics = LCO_evaluate_and_save_masks(
                model, valid_set, name=model_name, opt=opt
            )

        predicted_masks, label_masks = pdt_metrics[0], pdt_metrics[1]
        dice_scores, dice_thres = pdt_metrics[2], pdt_metrics[3]

    precision, recall, thresholds = precision_recall_curve(
        label_masks, predicted_masks)
    fpr, tpr, thres = roc_curve(label_masks, predicted_masks)

    np.savez_compressed(
        metrics_path,
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        fpr=fpr,
        tpr=tpr,
        thres=thres,
        dice_scores=dice_scores,
        dice_thres=dice_thres,
    )

    print(f"{model_name} evaluated metrics saved")

    return [precision, recall, thresholds, fpr, tpr, thres, dice_scores, dice_thres]


def evaluate_Gemini_metrics(opt, model_name, valid_set, rotation=False):
    # read or calculate comparision resutls
    metrics_path = f"{opt.expr_dir}/gemini_metrics_{opt.out_name}.npz"
    img_path = os.path.join(opt.expr_dir, f"pngs_{opt.suffix}")

    if os.path.isfile(metrics_path):

        metrics = np.load(metrics_path, allow_pickle=True)

        precision = metrics["precision"]
        recall = metrics["recall"]
        thresholds = metrics["thresholds"]

        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        thres = metrics["thres"]

        dice_scores = metrics["dice_scores"]
        dice_thres = metrics["dice_thres"]

        print(f"loaded {model_name} evaluated metrics")

        return [precision, recall, thresholds, fpr, tpr, thres, dice_scores, dice_thres]

    else:
        # init models
        model = Cosmic_CoNN()
        model.initialize(opt)
        model.eval()

        print(f"evaluating frames with {model_name}")
        # calculate metrics for two variants of deepCR-mask

        total_fits = len(valid_set)
        dice_thres = np.linspace(0, 1, 51)
        dice_scores = np.zeros((len(dice_thres), 1))

        thresholds = np.logspace(0, 32, num=100, base=2) / (2 ** 32)
        thresholds = np.pad(thresholds, (1, 0), "constant",
                            constant_values=(0.0, 1.0))

        metrics = np.zeros([len(thresholds), 4])
        selem = square(3)

        predicted_masks = np.array([], dtype="float32")
        label_masks = np.array([], dtype="uint8")

        with torch.no_grad():
            for i, f in enumerate(valid_set):
                print(f"{i}/{total_fits} fits tested")

                with fits.open(f) as hdul:
                    frame = np.array(hdul[1].data, dtype="float32")
                    mask_ref = np.array(hdul[3].data == 8, dtype="float32")
                    mask_ignore = np.array(hdul[3].data > 0, dtype="float32")
                    mask_object = np.array(hdul[4].data > 0, dtype="float32")
                    filename = hdul[0].header["ORIGNAME"]

                # Gemini ignore mask contain multiple classes, post process
                mask_ignore -= mask_ref
                mask_ignore = np.logical_or(mask_object, mask_ignore)

                # dilate the ignroe mask by 1px so the detector gaps are removed
                # inverse the ignore mask for computation
                mask_ignore = 1 - dilation(mask_ignore, selem)

                if "1x1" in opt.comment:
                    effective_area = [[600, 3900, 1400, 4900]]
                else:
                    effective_area = [[400, 1900, 800, 2500]]

                # Split frame into three effective areas and evaluate separately
                for i, a in enumerate(effective_area):

                    frm = frame[a[0]: a[1], a[2]: a[3]]
                    ref = mask_ref[a[0]: a[1], a[2]: a[3]]
                    ign = mask_ignore[a[0]: a[1], a[2]: a[3]]
                    fname = filename + f"_{i}"

                    if opt.clean_large:
                        # over-lapping crop and evaluate
                        pdt = clean_large(
                            frm,
                            model,
                            patch=opt.patch,
                            overlap=opt.overlap,
                            ret_numpy=True,
                        )
                    else:
                        pdt = model.detect_cr(frm, ret_numpy=True)

                    # apply ignore mask as GMOS reduction recipe did
                    ign = erase_boundary_np(ign, boundary_width=64)
                    pdt *= ign
                    ref *= ign

                    for t, thres in enumerate(dice_thres):
                        dice_scores[t] += calculate_dice_score(
                            pdt, ref, threshold=thres
                        )

                    predicted_masks = np.append(predicted_masks, pdt.ravel())
                    label_masks = np.append(
                        label_masks, ref.ravel().astype("uint8"))

                    if not opt.vis:
                        continue

                    save_as_png(pdt, img_path, fname, "mask_pdt", grid=True)
                    save_as_png(ref, img_path, fname, "mask_ref", grid=True)
                    save_as_png(ign, img_path, fname, "mask_ignore", grid=True)
                    save_as_png(frm, img_path, fname, "noisy",
                                zscale=True, grid=True)

    dice_scores /= total_fits

    precision, recall, thresholds = precision_recall_curve(
        label_masks, predicted_masks)
    fpr, tpr, thres = roc_curve(label_masks, predicted_masks)

    np.savez_compressed(
        metrics_path,
        precision=precision,
        recall=recall,
        thresholds=thresholds,
        fpr=fpr,
        tpr=tpr,
        thres=thres,
        dice_scores=dice_scores,
        dice_thres=dice_thres
    )

    print(f"{model_name} evaluated metrics saved")

    return [precision, recall, thresholds, fpr, tpr, thres, dice_scores, dice_thres]


def medmask(image, mask):
    clean = np.copy(image)
    xmax = image.shape[0]
    ymax = image.shape[1]
    medianImage = np.median(image)
    good = image * (1 - mask)
    pos = np.where(mask)
    for i in range(len(pos[0])):
        x = pos[0][i]
        y = pos[1][i]
        img = good[
            max(0, x - 2): min(x + 3, xmax + 1), max(0, y - 2): min(y + 3, ymax + 1)
        ]
        if img.sum() != 0:
            clean[x, y] = np.median(img[img != 0])
        else:
            clean[x, y] = medianImage
    return clean
