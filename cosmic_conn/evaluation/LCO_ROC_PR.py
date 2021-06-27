# -*- coding: utf-8 -*-

"""
Evaluation code for LCO imaging data
"""

import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from cosmic_conn.evaluation.utils import evaluate_LCO_model_metrics
from cosmic_conn.dl_framework.options import TrainingOptions
from cosmic_conn.dl_framework.dataloader import find_all_fits


def plot_ROC(plt, metrics_DL, metrics_LAC, y_label=False, zoom_in=False):
    linewidth = 1.5
    c_id = 0

    for label, metric in metrics_DL.items():
        color = colors[c_id]
        linestyle = "--"

        fpr = metric[3] * 100
        tpr = metric[4] * 100

        plt.plot(
            fpr, tpr, color=color, linestyle=linestyle, label=label, linewidth=linewidth
        )

        # print for benchmark table
        tpr_01 = tpr[(fpr < 0.1).sum()]
        tpr_001 = tpr[(fpr < 0.01).sum()]

        print(
            f"{label} FPR 0.01: TPR {round(tpr_001, 5)}; FPR 0.1: TPR {round(tpr_01, 5)}"
        )

        c_id += 1

    if metrics_LAC is not None:
        # Astro-SCRAPPY ROC
        for label, metric in metrics_LAC.items():
            TP, TN, FP, FN = metric[:, 0], metric[:,
                                                  1], metric[:, 2], metric[:, 3]
            TPR = TP / (TP + FN) * 100
            FPR = FP / (FP + TN) * 100

            # append extreme cases to connect complete ROC curves
            TPR = [100] + TPR.tolist() + [0]
            FPR = [100] + FPR.tolist() + [0]

            # a interpolate smoothing is not necessary at the moment
            # pr_f = interp1d(np.linspace(1, 0, len(TPR)), TPR)
            # TPR = pr_f(np.linspace(0, 1, len(fpr))) * 100
            # pr_f = interp1d(np.linspace(1, 0, len(FPR)), FPR)
            # FPR = pr_f(np.linspace(0, 1, len(fpr))) * 100

            label = str(f"Astro-SCRAPPY ({label})")

            plt.plot(
                FPR,
                TPR,
                color=colors[c_id],
                linestyle="--",
                label=label,
                linewidth=linewidth,
            )

            # export for benchmark table
            tpr_01 = TPR[(np.array(FPR) < 0.1).sum()]
            tpr_001 = TPR[(np.array(FPR) < 0.01).sum()]
            print(
                f"{label}: 0.01 FPR, TPR {round(tpr_001, 5)}, 0.1 FPR, TPR {round(tpr_01, 5)}"
            )

            c_id += 1

    # plt.legend(loc="lower right", bbox_to_anchor=(0.98, 0.01), prop={"size": 11})
    plt.legend(loc="lower right", prop={"size": 11})

    roc_xlim = [1e-3, 100]
    roc_ylim = [99, 100] if zoom_in else [60, 100]

    plt.xscale("log")
    plt.xlim(*roc_xlim)
    plt.ylim(*roc_ylim)

    if zoom_in:
        plt.yticks(ticks=[99, 99.2, 99.4, 99.6, 99.8, 100])
    else:
        plt.yticks(ticks=[60, 70, 80, 90, 100])

    plt.xticks(ticks=[10e-3, 10e-2, 10e-1, 1, 10, 100])
    plt.xlabel("false-positive rate (FPR) [%]")

    if y_label:
        plt.ylabel("true-positive rate (TPR) [%]")

    print()


def plot_PR(plt, eval_metrics):
    """
    Precision-Recall Curve

    Positive Predictive Power (Precision) = True Positives / (True Positives + False Positives)
    Recall (Sensitivity) = True Positives / (True Positives + False Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, f1_score]
    """

    # generate figure
    c_id = 0
    colors = ["tab:orange", "tab:blue", "tab:orange", "tab:blue"]

    for key, value in eval_metrics.items():
        label = key
        metric = value
        linestyle = "--"
        # linewidth = 2. if '256' in label else 1.5
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
            "o",
            color=colors[c_id],
            label=f"recall {thres_recall}, precision {thres_precision}",
        )

        # Second recall thershold at 99%
        thres_idx = np.sum(recall > 99) - 1
        thres_recall = round(recall[thres_idx], 2)
        thres_precision = round(precision[thres_idx], 2)

        print(
            f"threshold {dice_threshold} recalls {thres_recall} at precision {thres_precision}.\n"
        )

        plt.plot(
            recall[thres_idx],
            precision[thres_idx],
            "^",
            color=colors[c_id],
            label=f"recall {thres_recall}, precision {thres_precision}",
        )

        c_id += 1

    # plt.legend(loc="lower left", bbox_to_anchor=(0.02, 0.01), prop={"size": 11})
    plt.legend(loc="lower left", prop={"size": 12})

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("recall [%]")
    plt.ylabel("precision [%]")

    plt.xticks(ticks=[0, 20, 40, 60, 80, 100])

    print()


def plot_ROC_PR_figures(metrics_ROC, metrics_ROC_LAC, metrics_PR, comment):
    """
    ROC Curve

    True Positive Rate (Sensitivity) = True Positives / (True Positives + False Negatives)
    False Positive Rate (Specificity) = False Positives / (False Positives + True Negatives)

    metrics = [precision, recall, thresholds, fpr, tpr, thres, dice_score]
    """

    # generate figure
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams.update({"font.size": 12})
    plt.figure(figsize=(13, 4))

    # figure one, ROC
    plt.subplot(131)
    plt.grid(True)
    plt.title(f"(a) ROC")
    plot_ROC(plt, metrics_ROC, metrics_ROC_LAC, True)

    # figure two, ROC zoomed in
    plt.subplot(132)
    plt.grid(True)
    plt.title(f"(b) ROC (99-100% TPR)")
    plot_ROC(plt, metrics_ROC, metrics_ROC_LAC, False, True)

    # figure three, precision-recall
    plt.subplot(133)
    plt.grid(True)
    plt.title("(c) Precision-Recall")
    plot_PR(plt, metrics_PR)

    plt.tight_layout()
    plt.savefig(
        f"paper_PDFs/LCO_ROC_PR_figures_{comment}.pdf", bbox_inches="tight")

    print(f"saved to LCO_ROC_PR_figures_{comment}.pdf\n")


if __name__ == "__main__":

    metrics_PR = {}
    metrics_ROC_DL = {}
    metrics_ROC_LAC = {}
    colors = ["tab:orange", "tab:blue", "tab:red",
              "tab:green", "tab:pink", "k", "g"]

    opt = TrainingOptions()
    opt.initialize()
    opt.opt.mode = "inference"
    opt = opt.parse()

    # shared config
    opt.data = "/home/cyxu/astro/Cosmic_ConNN_datasets/LCO_CR_dataset/test_set"
    opt.valid_crop = 0
    opt.validRatio = 1.0
    opt.vis = False
    opt.rotation = False

    if not opt.random_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # Used in paper
    opt.comment = "for_paper"
    models = {
        "2021_03_14_16_36_LCO_Cosmic-Conn_1e3continue": [32, "group", 10040, "Cosmic-CoNN"],
        "2021_03_14_16_42_LCO_deepCR_continue": [32, "batch", 5370, "deepCR"],
    }

    # Ablation study
    opt.validRatio = 0.3

    # load test data
    valid_data = find_all_fits(opt, min_exposure=0.0)
    valid_data, _ = zip(*valid_data)
    valid_data = random.sample(valid_data, int(
        opt.validRatio * len(valid_data)))

    for key, value in models.items():
        opt.norm = value[1]
        opt.hidden = value[0]

        train_path = f"checkpoints_paper_final/{key}"

        # Title
        opt.out_name = f"{key}_epoch_{value[2]}"
        opt.expr_dir = os.path.join(
            train_path, f"LCO_data_ROC_PR_epoch{value[2]}")
        os.makedirs(opt.expr_dir, exist_ok=True)

        opt.load_model = os.path.join(
            train_path, "models", f"cr_checkpoint_{value[2]}.pth.tar"
        )

        metrics = evaluate_LCO_model_metrics(opt, key, valid_data)

        metrics_PR[value[3]] = metrics
        metrics_ROC_DL[value[3]] = metrics

        print()

    # load evaluation results from Astro-SCRAPPY
    scrappy_results = os.path.join(opt.data, "astro_scrappy_evaluation")

    path_0m4 = os.path.join(
        scrappy_results,
        "metrics_astro_scrappy_0m4_objlim0.5_sigfrac0.1.npz",
    )
    path_1m0 = os.path.join(
        scrappy_results,
        "metrics_astro_scrappy_1m0_objlim2.0_sigfrac0.1.npz",
    )
    path_2m0 = os.path.join(
        scrappy_results,
        "metrics_astro_scrappy_2m0_objlim2.0_sigfrac0.1.npz",
    )

    metrics_ROC_LAC["0.4m"] = np.load(path_0m4, allow_pickle=True)["metrics"]
    metrics_ROC_LAC["1.0m"] = np.load(path_1m0, allow_pickle=True)["metrics"]
    metrics_ROC_LAC["2.0m"] = np.load(path_2m0, allow_pickle=True)["metrics"]

    ####################################
    # Plotting scripts

    plot_ROC_PR_figures(metrics_ROC_DL, metrics_ROC_LAC,
                        metrics_PR, opt.comment)
