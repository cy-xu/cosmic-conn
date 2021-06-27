# -*- coding: utf-8 -*-

"""
Evaluation code for Gemini GMOS-N/S imaging data
"""

import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from cosmic_conn.evaluation.utils import evaluate_Gemini_metrics
from cosmic_conn.dl_framework.options import TrainingOptions
from cosmic_conn.dl_framework.dataloader import find_all_fits


def test_model(a_name, best_epoch, valid_data, opt):
    train_path = f"checkpoints_paper_final/{a_name}"

    # Title
    opt.out_name = f"{a_name}_epoch_{best_epoch}_{opt.comment}"

    opt.load_model = os.path.join(
        train_path, "models", f"cr_checkpoint_{best_epoch}.pth.tar"
    )
    opt.expr_dir = os.path.join(
        train_path, f"Gemini_data_ROC_PR_epoch{best_epoch}")
    os.makedirs(opt.expr_dir, exist_ok=True)

    metrics = evaluate_Gemini_metrics(opt, a_name, valid_data)
    print()

    return metrics


def plot_PR(plt, eval_metrics, pr_labels, line_style, marker, y_label=False):
    """
    Precision-Recall Curve

    Positive Predictive Power (Precision) = True Positives / (True Positives + False Positives)
    Recall (Sensitivity) = True Positives / (True Positives + False Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, f1_score]
    """

    # generate figure
    c_id = 0
    l_id = 0
    colors = ["tab:orange", "tab:blue", "tab:orange",
              "tab:blue", "tab:orange", "tab:blue"]

    for key, value in eval_metrics.items():
        label = pr_labels[l_id]
        metric = value
        linestyle = line_style
        linewidth = 1.5

        precision = metric[0] * 100
        recall = metric[1] * 100

        plt.plot(
            recall,
            precision,
            colors[c_id],
            linestyle=linestyle,
            label=label,
            linewidth=linewidth,
        )

        dice_scores = metric[-2]
        dice_thres = metric[-1]
        dice_idx = np.argmax(dice_scores)
        dice_best = dice_scores[dice_idx]
        dice_threshold = dice_thres[dice_idx]

        print(
            f"{label} Dice score: {round(dice_best.item(), 5)} at {dice_threshold}.\n")

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
            marker,
            color=colors[c_id],
            label=f"recall {thres_recall}, precision {thres_precision}",
        )

        c_id += 1
        l_id += 1

    # plt.legend(loc="lower left", bbox_to_anchor=(0.02, 0.01), prop={"size": 11})
    plt.legend(loc="lower left", prop={"size": 12})

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("recall [%]")

    if y_label:
        plt.ylabel("precision [%]")

    plt.xticks(ticks=[0, 20, 40, 60, 80, 100])

    print()


def plot_ROC_PR_figures(metrics_ROC_1x1, metrics_ROC_2x2):

    # generate figure
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams.update({"font.size": 12})
    plt.figure(figsize=(13, 4))

    # figure one, ROC
    """
    ROC Curve

    True Positive Rate (Sensitivity) = True Positives / (True Positives + False Negatives)
    False Positive Rate (Specificity) = False Positives / (False Positives + True Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, dice_score]
    """

    plt.subplot(131)
    plt.grid(True)
    plt.title("(a) ROC (80-100% TPR)")
    linewidth = 1.5
    c_id = 0
    l_id = 0

    labels = ['(1x1) Cosmic-CoNN', '(1x1) deepCR',
              '(2x2) Cosmic-CoNN', '(2x2) deepCR']

    # 1x1 binning
    for label, metric in metrics_ROC_1x1.items():
        color = colors[c_id]
        linestyle = "--"

        fpr = metric[3] * 100
        tpr = metric[4] * 100

        plt.plot(
            fpr, tpr, color=color, linestyle=linestyle, label=labels[l_id], linewidth=linewidth
        )

        # print for benchmark table
        tpr_01 = tpr[(fpr < 0.1).sum()]
        tpr_001 = tpr[(fpr < 0.01).sum()]

        print(
            f"{labels[l_id]} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}"
        )

        # find out best threshold and its dice score
        dice_scores = metric[-2]
        dice_thres = metric[-1]
        dice_idx = np.argmax(dice_scores)
        dice_best = dice_scores[dice_idx]
        dice_threshold = dice_thres[dice_idx]

        print(
            f"{labels[l_id]} Dice score: {round(dice_best.item(), 5)} at {dice_threshold}\n")

        c_id += 1
        l_id += 1

    # 2x2 binning
    c_id = 0
    for label, metric in metrics_ROC_2x2.items():
        color = colors[c_id]
        linestyle = "dotted"

        fpr = metric[3] * 100
        tpr = metric[4] * 100

        plt.plot(
            fpr, tpr, color=color, linestyle=linestyle, label=labels[l_id], linewidth=linewidth
        )

        # print for benchmark table
        tpr_01 = tpr[(fpr < 0.1).sum()]
        tpr_001 = tpr[(fpr < 0.01).sum()]

        print(
            f"{labels[l_id]} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}"
        )

        # find out best threshold and its dice score
        dice_scores = metric[-2]
        dice_thres = metric[-1]
        dice_idx = np.argmax(dice_scores)
        dice_best = dice_scores[dice_idx]
        dice_threshold = dice_thres[dice_idx]

        print(
            f"{labels[l_id]} Dice score: {round(dice_best.item(), 5)} at {dice_threshold}\n")

        c_id += 1
        l_id += 1

    # plt.legend(loc="lower right", bbox_to_anchor=(0.98, 0.01), prop={"size": 11})
    plt.legend(loc="lower right", prop={"size": 12})

    roc_xlim = [1e-3, 100]
    roc_ylim = [80, 100]

    plt.xscale("log")
    plt.xlim(*roc_xlim)
    plt.ylim(*roc_ylim)

    plt.xticks(ticks=[10e-3, 10e-2, 10e-1, 1, 10, 100])
    plt.yticks(ticks=[80, 85, 90, 95, 100])

    plt.xlabel("FPR [%]")
    plt.ylabel("TPR [%]")

    # figure two

    plt.subplot(132)
    plt.grid(True)
    plt.title("(b) Precision-Recall (1x1 binning)")

    pr_labels = ['Cosmic-CoNN', 'deepCR']
    plot_PR(plt, metrics_ROC_1x1, pr_labels, '--', 'o', True)

    # figure three

    plt.subplot(133)
    plt.grid(True)
    plt.title("(c) Precision-Recall (2x2 binning)")

    plot_PR(plt, metrics_ROC_2x2, pr_labels, 'dotted', '^')

    plt.tight_layout()
    plt.savefig(
        f"paper_PDFs/Gemini_ROC_PR_figures_{opt.suffix}.pdf", bbox_inches="tight",
    )
    print(f"saved to Gemini_ROC_PR_figures_{opt.suffix}.pdf")


if __name__ == "__main__":

    metrics_PR_2x2 = {}
    metrics_ROC_DL_2x2 = {}

    metrics_PR_1x1 = {}
    metrics_ROC_DL_1x1 = {}

    metrics_ROC_LAC = {}
    colors = ["tab:orange", "tab:blue", "tab:green",
              "tab:pink", "tab:red", "k", "g"]

    opt = TrainingOptions()
    opt.initialize()
    opt.opt.mode = "inference"
    opt = opt.parse()

    # shared config
    opt.validRatio = 1.0
    opt.vis = False
    ignore_mask = True
    threshold = 0.01

    opt.clean_large = False
    opt.patch = 1024
    opt.overlap = 0

    if not opt.random_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # load test data
    opt.data = "/home/cyxu/astro/Cosmic_ConNN_datasets/GEMINI_testset/masked_fits/2x2_binning"

    valid_data_2x2 = find_all_fits(opt, min_exposure=0.0)
    valid_data_2x2, _ = zip(*valid_data_2x2)
    valid_data_2x2 = random.sample(
        valid_data_2x2, int(opt.validRatio * len(valid_data_2x2))
    )

    # load test data
    opt.data = "/home/cyxu/astro/Cosmic_ConNN_datasets/GEMINI_testset/masked_fits/1x1_binning"

    valid_data_1x1 = find_all_fits(opt, min_exposure=0.0)
    valid_data_1x1, _ = zip(*valid_data_1x1)
    valid_data_1x1 = random.sample(
        valid_data_1x1, int(opt.validRatio * len(valid_data_1x1))
    )

    opt.suffix = "for_paper"
    # used in paper
    models = {
        "2021_03_14_16_36_LCO_Cosmic-Conn_1e3continue": [32, "group", 10040, "Cosmic-CoNN"],
        "2021_03_14_16_42_LCO_deepCR_continue": [32, "batch", 5370, "deepCR"],
    }

    for key, value in models.items():
        opt.norm = value[1]
        opt.hidden = value[0]

        opt.comment = "1x1_test"
        metrics_1x1 = test_model(key, value[2], valid_data_1x1, opt)
        metrics_ROC_DL_1x1[value[3]] = metrics_1x1

        opt.comment = "2x2_test"
        metrics_2x2 = test_model(key, value[2], valid_data_2x2, opt)
        metrics_ROC_DL_2x2[value[3]] = metrics_2x2

    # Plotting scripts
    plot_ROC_PR_figures(metrics_ROC_DL_1x1, metrics_ROC_DL_2x2)

print("the end")
