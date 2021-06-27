# -*- coding: utf-8 -*-

"""
find the more generic model
"""

import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt

from cosmic_conn.evaluation.utils import evaluate_Gemini_metrics, evaluate_LCO_model_metrics
from cosmic_conn.dl_framework.options import TrainingOptions
from cosmic_conn.dl_framework.dataloader import find_all_fits


def key_metrics(m_lco, m_a, m_b):
    key_values = {}
    metrics_list = [m_lco, m_a, m_b]

    for i in range(3):
        metrics = metrics_list[i]
        precision = metrics[0] * 100
        recall = metrics[1] * 100

        for thres in [95, 99]:
            thres_idx = np.sum(recall > thres) - 1
            thres_recall = recall[thres_idx]
            thres_precision = round(precision[thres_idx], 5)
            print(
                f'at {int(thres_recall)}, precision {round(thres_precision, 3)}')

            key_values.setdefault(thres, 0)

            if i == 0:
                key_values[thres] += 2 * thres_precision
            else:
                key_values[thres] += thres_precision

    key_values[95] /= 4.0
    key_values[99] /= 4.0

    precision_avg = round((key_values[95]+key_values[99]) / 2.0, 5)
    key_values['average'] = precision_avg

    message = f"average precision {precision_avg}. \
        95-99: {round(key_values[95],3), round(key_values[99],3)}"
    print(message)

    return message, key_values


if __name__ == "__main__":

    opt = TrainingOptions()
    opt.initialize()
    opt.opt.mode = "inference"
    opt = opt.parse()

    # shared config
    opt.validNum = 20
    ignore_mask = True
    threshold = 0.01

    opt.patch = 1024
    opt.overlap = 0
    opt.clean_large = False
    opt.vis = False
    opt.rotation = False

    if not opt.random_seed:
        torch.manual_seed(1)
        np.random.seed(1)
        random.seed(1)

    # load LCO test data
    opt.data = "/home/cyxu/astro/Cosmic_CoNN_datasets/LCO_CR_dataset/test_set"

    valid_data = find_all_fits(opt, min_exposure=0.0)
    valid_data, _ = zip(*valid_data)
    valid_data = random.sample(valid_data, opt.validNum)

    # load Gemini test data
    opt.data = "/home/cyxu/astro/Cosmic_CoNN_datasets/GEMINI_testset/masked_fits/2x2_binning"

    valid_data_2x2 = find_all_fits(opt, min_exposure=0.0)
    valid_data_2x2, _ = zip(*valid_data_2x2)

    # load Gemini test data
    opt.data = "/home/cyxu/astro/Cosmic_CoNN_datasets/GEMINI_testset/masked_fits/1x1_binning"

    valid_data_1x1 = find_all_fits(opt, min_exposure=0.0)
    valid_data_1x1, _ = zip(*valid_data_1x1)

    # candidates

    opt.hidden = 32
    target_model = "2021_03_14_16_36_LCO_Cosmic-Conn_1e3continue"
    candidates = [8210, 10040, 11040, 13320, 14860, 15180, 17260, 20360]

    opt.hidden = 32
    opt.norm = 'batch'
    target_model = "2021_03_14_16_42_LCO_deepCR_continue"
    candidates = [3040, 3540, 4020, 4530, 5060, 5370, 5670, 6040]

    # locate all the models to test
    models = []
    model_names = []
    train_path = f"checkpoints/{target_model}"

    for root, dirs, files in os.walk(train_path):
        for f in files:
            if f.endswith('tar'):
                strings = f.split('_')
                epoch = [int(s) for s in strings[-1].split('.') if s.isdigit()]
                # if epoch[0] > opt.starting_epoch:
                if epoch[0] in candidates:
                    models.append([os.path.join(root, f), f])

    # test all the models

    for m_path, m_name in models:

        # load the model
        opt.load_model = m_path

        opt.expr_dir = os.path.join(train_path, 'model_search', m_name)
        os.makedirs(opt.expr_dir, exist_ok=True)

        opt.out_name = m_name+'_1x1'
        metrics_11 = evaluate_Gemini_metrics(
            opt, m_name+'_1x1', valid_data_1x1)

        opt.out_name = m_name+'_2x2'
        metrics_22 = evaluate_Gemini_metrics(
            opt, m_name+'_2x2', valid_data_2x2)

        opt.out_name = m_name+'_lco'
        metrics_lco = evaluate_LCO_model_metrics(
            opt, m_name+'_lco', valid_data)

        msg, key_11 = key_metrics(metrics_lco, metrics_11, metrics_22)

        # logging
        log_path = os.path.join(train_path, 'model_search.txt')
        with open(log_path, "a") as log_file:
            log_file.write(f"{m_name}: {msg}\n")

print("the end")
