# -*- coding: utf-8 -*-

"""
Evaluation code for LCO NRES spectroscopic data
"""

import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from cosmic_conn.evaluation.utils import evaluate_LCO_model_metrics
from cosmic_conn.dl_framework.options import TrainingOptions
from cosmic_conn.dl_framework.dataloader import find_all_fits


def plot_PR(plt, eval_metrics):
    """
    Precision-Recall Curve

    Positive Predictive Power (Precision) = True Positives / (True Positives + False Positives)
    Recall (Sensitivity) = True Positives / (True Positives + False Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, f1_score]
    """

    colors = ["tab:blue", "tab:orange", "tab:red",
              "tab:green", "tab:pink", "k", "g"]
    c_id = 0

    for key, value in eval_metrics.items():
        label = key
        metric = value
        color = colors[c_id]
        linestyle = "--"
        linewidth = 1.5

        precision = metric[0] * 100
        recall = metric[1] * 100

        plt.plot(
            recall,
            precision,
            color,
            linestyle=linestyle,
            label=label,
            linewidth=linewidth,
        )

        dice_scores = metric[-2]
        dice_thres = metric[-1]
        dice_idx = np.argmax(dice_scores)
        dice_threshold = dice_thres[dice_idx]

        # PR threshold list from 0 to 1
        thres_idx = np.sum(recall > 95) - 1
        thres_recall = round(recall[thres_idx], 2)
        thres_precision = round(precision[thres_idx], 2)

        print(
            f"threshold {dice_threshold} recalls {thres_recall} at precision {thres_precision}.\n"
        )

        plt.plot(
            recall[thres_idx],
            precision[thres_idx],
            "o",
            color=color,
            label=f"recall {thres_recall}, precision {thres_precision}",
        )

        c_id += 1

    plt.legend(loc=4)

    plt.xlim(0, 104)
    plt.ylim(0, 104)
    plt.xlabel("recall [%]")
    plt.ylabel("precision [%]")


def plot_ROC(plt, metrics_ROC):
    linewidth = 1.5
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
    c_id = 0

    for key, value in metrics_ROC.items():
        label = key
        metric = value
        color = colors[c_id]

        fpr = metric[3] * 100
        tpr = metric[4] * 100

        plt.plot(
            fpr, tpr, color=color, linestyle="solid", label=label, linewidth=linewidth
        )

        # print for benchmark table
        tpr_01 = tpr[(fpr < 0.1).sum()]
        tpr_001 = tpr[(fpr < 0.01).sum()]

        print(
            f"{label} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}\n"
        )

        dice_scores = metric[-2]
        dice_thres = metric[-1]
        dice_idx = np.argmax(dice_scores)
        dice_best = dice_scores[dice_idx]
        dice_threshold = dice_thres[dice_idx]

        print(f"{label} Dice score: {round(dice_best.item(), 5)} at {dice_threshold}")

        c_id += 1

    plt.legend(loc=4)

    roc_xlim = [1e-3, 100]
    roc_ylim = [90, 100]

    plt.xscale("log")
    plt.xlim(*roc_xlim)
    plt.ylim(*roc_ylim)

    plt.xlabel("false positive rate [%] (log10)")
    plt.ylabel("true positive rate [%]")


def plot_ROC_PR_figures(metrics_ROC, metrics_PR):
    """
    ROC Curve

    True Positive Rate (Sensitivity) = True Positives / (True Positives + False Negatives)
    False Positive Rate (Specificity) = False Positives / (False Positives + True Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, dice_score]
    """

    # generate figure
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams.update({"font.size": 12})

    plt.figure(figsize=(12, 6))
    # plt.suptitle('Receiver Operating Characteristic', fontsize=14)

    ########### Figure Three, ROC zoomed in for DL models ##############
    plt.subplot(121)
    plt.grid(True)
    plt.title(f"Spectroscopic Data ROC")
    plot_ROC(plt, metrics_ROC)

    ########### Figure Two, ROC with different Sigfrac ##############
    plt.subplot(122)
    plt.grid(True)
    plt.title("Precision-Recall")
    plot_PR(plt, metrics_PR)

    # plt.tight_layout()
    plt.savefig(
        f"paper_PDFs/NRES_ROC_PR_figures_{opt.suffix}.pdf", bbox_inches="tight"
    )
    print(f"ROC figure saved to NRES_ROC_PR_figures")
    print()


if __name__ == "__main__":

    PR_eval_metrics = {}
    ROC_eval_metrics = {}

    opt = TrainingOptions()
    opt.initialize()
    opt.opt.mode = "inference"
    opt = opt.parse()

    # shared config
    opt.data = "/home/cyxu/astro/Cosmic_CoNN_datasets/LCO_NRES/test_set"
    opt.valid_crop = 0
    opt.validRatio = 1.0
    opt.vis = False
    ignore_mask = True

    if not opt.random_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # load test data
    valid_data = find_all_fits(opt, min_exposure=0.0)
    valid_data = random.sample(valid_data, int(
        opt.validRatio * len(valid_data)))
    valid_set = [f[0] for f in valid_data]
    print(f"{len(valid_set)} x 3 NRES images used for testing, total {len(valid_data)}")

    ####################################

    # used in paper
    opt.suffix = "gn_7760"
    models = {
        "2021_05_05_00_25_NRES_Cosmic-Conn_GN": [
            "group",
            7760,
            "gn_7760",
        ],
    }

    for key, value in models.items():
        opt.norm = value[0]
        train_path = f"checkpoints/{key}"

        # dir name
        opt.out_name = f"{key}_epoch_{value[1]}"

        # model path
        opt.load_model = os.path.join(
            train_path, "models", f"cr_checkpoint_{value[1]}.pth.tar"
        )

        # eval result output
        opt.expr_dir = os.path.join(
            train_path, f"NRES_data_ROC_PR_epoch_{value[1]}")
        os.makedirs(opt.expr_dir, exist_ok=True)

        metrics = evaluate_LCO_model_metrics(opt, key, valid_set, nres=True)
        PR_eval_metrics[key] = metrics
        ROC_eval_metrics[key] = metrics

        print()

    ####################################
    # Plotting scripts

    plot_ROC_PR_figures(ROC_eval_metrics, PR_eval_metrics)
