# -*- coding: utf-8 -*-

"""
Evaluation code for HST imaging data
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import os
from deepCR import deepCR

from cosmic_conn.inference_cr import init_model
from cosmic_conn.evaluation.utils import HST_evaluate_and_save_masks
from cosmic_conn.dl_framework.dataloader import read_HST_patches, HST_dataset


def test_model(model, dataset, limit, out_dir, data_type, model_name):
    metrics_path = out_dir + "_" + data_type + ".npz"

    if os.path.isfile(metrics_path):
        metrics = np.load(metrics_path, allow_pickle=True)

        fpr = metrics["fpr"]
        tpr = metrics["tpr"]
        thres = metrics["thres"]
        dice_scores = metrics["dice_scores"]
        dice_thres = metrics["dice_thres"]

        metrics = [fpr, tpr, thres, dice_scores, dice_thres]

        print(f"loaded {model_name, data_type} evaluated metrics")

    else:
        print(f"evaluate {model_name, data_type}")

        metrics = HST_evaluate_and_save_masks(
            model,
            dataset,
            model_name,
            limit=limit,
            metrics_path=metrics_path,
            data_type=data_type,
        )

    dice_scores = metrics[-2]
    dice_thres = metrics[-1]
    dice_idx = np.argmax(dice_scores)
    dice_best = dice_scores[dice_idx]
    dice_threshold = dice_thres[dice_idx]

    print(
        f"{model_name} {data_type} Dice score: {round(dice_best.item(), 5)} at {dice_threshold}\n"
    )

    return metrics


def plot_ROC(
    plt, deepcr_metrics, cosmic_conn_metrics, roc_ylim, title, ylabel=False,
):

    cosmic_FPR = cosmic_conn_metrics[0] * 100
    cosmic_TPR = cosmic_conn_metrics[1] * 100

    # print for benchmark table
    tpr_01 = cosmic_TPR[(cosmic_FPR < 0.1).sum()]
    tpr_001 = cosmic_TPR[(cosmic_FPR < 0.01).sum()]

    print(
        f"Cosmic-CoNN {title} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}"
    )

    plt.plot(
        cosmic_FPR,
        cosmic_TPR,
        color="tab:orange",
        linestyle="--",
        label="Cosmic-CoNN",
        linewidth=1.5,
    )

    deepcr_FPR = deepcr_metrics[0] * 100
    deepcr_TPR = deepcr_metrics[1] * 100

    # print for benchmark table
    tpr_01 = deepcr_TPR[(deepcr_FPR < 0.1).sum()]
    tpr_001 = deepcr_TPR[(deepcr_FPR < 0.01).sum()]

    print(
        f"deepCR {title} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}\n"
    )

    plt.plot(
        deepcr_FPR,
        deepcr_TPR,
        color="tab:blue",
        linestyle="--",
        label="deepCR",
        linewidth=1.5,
    )

    plt.legend(loc=4)
    plt.xscale("log")
    plt.xlim(*roc_xlim)
    plt.xticks(ticks=[10e-3, 10e-2, 10e-1, 1, 10, 100])

    plt.ylim(*roc_ylim)
    plt.grid()

    plt.title(title)
    plt.xlabel("false positive rate [%]")

    if ylabel:
        plt.ylabel("true positive rate [%]")

    # print the precision-recall stats

    cosmic_precision = cosmic_conn_metrics[3] * 100
    cosmic_recall = cosmic_conn_metrics[4] * 100

    deepcr_precision = deepcr_metrics[3] * 100
    deepcr_recall = deepcr_metrics[4] * 100

    # Cosmic-CoNN
    thres_idx = np.sum(cosmic_recall < 95) - 1
    thres_recall = round(cosmic_recall[thres_idx], 2)
    thres_precision = round(cosmic_precision[thres_idx].item(), 2)
    thres_precision = (cosmic_precision[thres_idx].item() +
                       cosmic_precision[thres_idx+1].item())/2
    thres_precision = round(thres_precision, 2)
    print(
        f"Cosmic-CoNN recall 95, precision {thres_precision}.\n")

    # deepCR
    thres_idx = np.sum(deepcr_recall < 95) - 1
    thres_recall = round(deepcr_recall[thres_idx], 2)
    # average between 94 and 96
    thres_precision = (deepcr_precision[thres_idx].item() +
                       deepcr_precision[thres_idx+1].item())/2
    thres_precision = round(thres_precision, 2)
    print(
        f"deepCR recall 95, precision {thres_precision}.\n")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    print("Creating ROC curves for HST test data")

    # use entire test set from deepCR for evaluation
    limit = 840

    # currently used in paper
    checkpoint = "2021_03_27_18_44_HST_Cosmic-Conn_BCE"
    epoch = 780

    coscmic_metrics_dir = (
        f"checkpoints_paper_final/{checkpoint}/HST_data_evaluation_epoch{epoch}_{limit}"
    )

    cosmic_metrics = os.path.join(
        coscmic_metrics_dir, f"HST_cosmic-conn_metrics_{limit}"
    )

    # init Cosmic-CoNN model
    load_model = f"checkpoints_paper_final/{checkpoint}/models/cr_checkpoint_{epoch}.pth.tar"
    cosmic_conn_model, opt = init_model(load_model)

    # init deepCR model
    deepCR_2_32 = deepCR(mask='ACS-WFC-F606W-2-32',
                         inpaint='ACS-WFC-F606W-2-32', device='GPU')

    deepcr_metrics_dir = f"/home/cyxu/astro/Cosmic_CoNN_datasets/HST_deepCR/deepCR_evaluation"
    deepcr_metrics = os.path.join(
        deepcr_metrics_dir, f"HST_deepCR_metrics_{limit}")

    # create new data loader for each HST category
    print(f"loading DeepCR HST test set")
    all_paths = read_HST_patches(
        "/home/cyxu/astro/Cosmic_CoNN_datasets/HST_deepCR/unpickled_testset")

    ex_paths = [f for f in all_paths if "EX" in os.path.basename(f)]
    gc_paths = [f for f in all_paths if "GC" in os.path.basename(f)]
    gal_paths = [f for f in all_paths if "GAL" in os.path.basename(f)]

    dset_test_EX = HST_dataset(
        ex_paths, mini_batch=1, max_size=limit, mode="evaluate")
    dset_test_GC = HST_dataset(
        gc_paths, mini_batch=1, max_size=limit, mode="evaluate")
    dset_test_GAL = HST_dataset(
        gal_paths, mini_batch=1, max_size=limit, mode="evaluate"
    )

    datasets = [dset_test_EX, dset_test_GC, dset_test_GAL]

    # deepCR model trained on HST

    """
    ROC Curve
    True Positive Rate (Sensitivity) = True Positives / (True Positives + False Negatives)
    False Positive Rate (Specificity) = False Positives / (False Positives + True Negatives)
    """

    plt.figure(figsize=(12, 4))
    plt.rcParams.update({"font.size": 12})

    roc_xlim = [1e-3, 100]
    roc_ylim = [70, 100]

    # EX ROC
    plt.subplot(131)
    deepcr_EX = test_model(
        deepCR_2_32, dset_test_EX, limit, deepcr_metrics, "EX", "deepCR"
    )
    cosmic_EX = test_model(
        cosmic_conn_model, dset_test_EX, limit, cosmic_metrics, "EX", "Cosmic-CoNN"
    )
    plot_ROC(plt, deepcr_EX, cosmic_EX, roc_ylim, "extragalactic field", True)

    # GC ROC
    plt.subplot(132)
    deepcr_GC = test_model(
        deepCR_2_32, dset_test_GC, limit, deepcr_metrics, "GC", "deepCR"
    )
    cosmic_GC = test_model(
        cosmic_conn_model, dset_test_GC, limit, cosmic_metrics, "GC", "Cosmic-CoNN"
    )
    plot_ROC(plt, deepcr_GC, cosmic_GC, roc_ylim, "globular cluster", False)

    # GAL ROC
    plt.subplot(133)
    roc_ylim = [40, 100]
    deepcr_GAL = test_model(
        deepCR_2_32, dset_test_GAL, limit, deepcr_metrics, "GAL", "deepCR"
    )
    cosmic_GAL = test_model(
        cosmic_conn_model, dset_test_GAL, limit, cosmic_metrics, "GAL", "Cosmic-CoNN"
    )
    plot_ROC(plt, deepcr_GAL, cosmic_GAL, roc_ylim, "resolved galaxy", False)

    plt.tight_layout()

    figure_path = f"paper_PDFs/HST_ROC_PR_{limit}_{checkpoint}_epoch{epoch}.pdf"

    plt.savefig(figure_path, bbox_inches="tight")

    print(f"saved to {figure_path}")
