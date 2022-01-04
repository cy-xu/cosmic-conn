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


def plot_theshold_completeness(plt, metrics, preferred_thres):
    """
    # the order of metrics:
    # metrics = [precision, recall, thresholds, fpr, tpr, thres, dice_scores, dice_thres]

    Completeness (Recall) = True Positives / (True Positives + False Negatives)    
    """

    linewidth = 1.5
    c_id = 0

    for label, metric in metrics.items():
        color = colors[c_id]
        linestyle = "--"

        thresholds = metric[2]
        completeness = metric[1] * 100

        # completeness has one extra value than thresholds
        completeness = completeness[:-1]

        plt.plot(
            thresholds, completeness, color=color, linestyle=linestyle, label=label, linewidth=linewidth
        )

        for thres in preferred_thres:
            c_id += 1

            # PR threshold list from 0 to 1
            thres_idx = np.sum(thresholds < thres) - 1
            print(f'thres {thres}, completeness {completeness[thres_idx]}')

            plt.plot(
                thres,
                completeness[thres_idx],
                "o",
                color=colors[c_id],
                label=f"t={thres}, completeness={round(completeness[thres_idx], 2)}",
            )

    plt.legend(loc="lower left")

    plt.xscale("log")
    # plt.yscale("log")
    plt.xlim(left=1e-5)
    # plt.xlim(right=1.1)
    # plt.xlim(0.01, 1.1)
    plt.ylim(top=105)

    plt.xticks(ticks=[1e-5, 1e-3, 1e-1, 1])

    plt.xlabel("threshold (0.0 ≤ t ≤ 1.0)")
    plt.ylabel("completeness / recall [%]")

    print()


def plot_threshold_false_discovery_rate(plt, eval_metrics, preferred_thres):
    """
    False Discovery Rate (FDR) = False Positives / (True Positives + False Positives)
    # we don't have this value but FDR = 1 - precision, which we know

    # the order of metrics:
    # metrics = [precision, recall, thresholds, fpr, tpr, thres, dice_scores, dice_thres]
    """

    # generate figure
    c_id = 0

    for key, value in eval_metrics.items():
        label = key
        metric = value
        linestyle = "--"
        linewidth = 1.5

        thresholds = metric[2]

        # precision has one extra value than thresholds
        precision = metric[0]
        precision = precision[:-1]

        false_discovery_rate = (1 - precision) * 100

        plt.plot(
            thresholds,
            false_discovery_rate,
            colors[c_id],
            linestyle=linestyle,
            label=label,
            linewidth=linewidth,
        )

        for thres in preferred_thres:
            c_id += 1
            # PR threshold list from 0 to 1
            thres_idx = np.sum(thresholds < thres) - 1

            plt.plot(
                thres,
                false_discovery_rate[thres_idx],
                "o",
                color=colors[c_id],
                label=f"t={thres}, FDR={round(false_discovery_rate[thres_idx], 2)}",
            )

    plt.legend(loc="lower left")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1e-7)
    plt.ylim(bottom=1e-3)
    # plt.ylim(top=120)

    plt.xticks(ticks=[1e-7, 1e-5, 1e-3, 1e-1, 1])

    plt.xlabel("threshold (0.0 ≤ t ≤ 1.0)")
    plt.ylabel("false discovery rate [%]")

    print()


def plot_figures(metrics, preferred_thres, comment):
    # generate figure
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams.update({"font.size": 12})
    # plt.figure(figsize=(13, 4))
    plt.figure(figsize=(9, 4))

    # figure one, ROC
    plt.subplot(121)
    plt.grid(True)
    plt.title(f"Thesholds to Completeness")
    plot_theshold_completeness(plt, metrics, preferred_thres)

    # figure two, ROC zoomed in
    plt.subplot(122)
    plt.grid(True)
    plt.title(f"Thresholds to False Discovery Rate (FDR)")
    plot_threshold_false_discovery_rate(plt, metrics, preferred_thres)

    plt.tight_layout()
    plt.savefig(
        f"paper_PDFs/LCO_banzai_figures_{comment}.pdf", bbox_inches="tight")

    print(f"saved to LCO_banzai_figures_{comment}.pdf\n")


if __name__ == "__main__":

    metrics_cos = {}
    metrics_ROC_DL = {}
    metrics_ROC_LAC = {}
    colors = ["tab:orange", "tab:red", "tab:blue",
              "tab:pink", "tab:green", "k", "g"]

    opt = TrainingOptions()
    opt.initialize()
    opt.opt.mode = "inference"
    opt = opt.parse()

    # shared config
    opt.data = "/home/cyxu/astro/Cosmic_CoNN_datasets/LCO_CR_dataset/test_set"
    opt.valid_crop = 2000
    opt.validRatio = 1.0
    opt.vis = False
    opt.rotation = False

    # plot a particular threshold on the curves
    preferred_thres = [0.1, 0.5, 0.9]

    if not opt.random_seed:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    # Used in paper
    opt.comment = "for_banzai_docs"
    models = {
        "2021_03_14_16_36_LCO_Cosmic-Conn_1e3continue": [32, "group", 10040, "Cosmic-CoNN"],
    }

    # load test data
    valid_data = find_all_fits(opt, min_exposure=0.0)
    valid_data, _ = zip(*valid_data)
    valid_data = random.sample(valid_data, int(
        opt.validRatio * len(valid_data)))

    for key, value in models.items():
        opt.norm = value[1]
        opt.hidden = value[0]

        train_path = f"checkpoints_paper_final/{key}"
        train_path = f"checkpoints_revision/{key}"
        train_path = f"checkpoints/{key}"

        # Title
        opt.out_name = f"{key}_epoch_{value[2]}_2k_Crop"
        opt.expr_dir = os.path.join(
            train_path, f"LCO_data_ROC_PR_epoch{value[2]}")
        os.makedirs(opt.expr_dir, exist_ok=True)

        opt.load_model = os.path.join(
            train_path, "models", f"cr_checkpoint_{value[2]}.pth.tar"
        )

        metrics = evaluate_LCO_model_metrics(opt, key, valid_data)

        metrics_cos[value[3]] = metrics

    ####################################
    # Plotting scripts

    plot_figures(metrics_cos, preferred_thres, opt.comment)
