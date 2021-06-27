# -*- coding: utf-8 -*-

"""
Plot the training evaluation figure for paper
CY Xu (cxu@ucsb.edu)
"""

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def extract_csv(path, max_epoch, scale_correction=False):
    """because Tensorboard export CSV is not continuous/complete,
    we create a new array in order to plot the curves correctly
    """
    array = genfromtxt(path, delimiter=",", dtype="float32")
    # remove Nan in first row
    array = array[1:]
    # new None array
    sparce_array = np.full(max_epoch+1, None)
    # init loss = 1
    sparce_array[0] = 1

    # fill sparce array with availabe data
    for line in array:
        epoch = int(line[1])
        loss = line[2]

        if scale_correction:
            # 1024 model sampled 70 samples per epoch,
            # while it shuold be 210 to have equal updates per epoch
            # as 256 models, which used full dataset (Appendix A)
            epoch = epoch // 3

        if epoch > max_epoch:
            break

        sparce_array[epoch] = loss

    return sparce_array.astype('float32')


save_path = "paper_utils/training_evaluation.pdf"

# March 32 channel batch
cosmic_conn_1024 = "paper_utils/tensorboard_logs/run-2021_03_14_16_36_LCO_Cosmic-Conn_1e3continue-tag-loss_valid_cr_loss.csv"
cosmic_conn_BN = "paper_utils/tensorboard_logs/run-2021_06_04_17_45_LCO_seed0_Cosmic-CoNN_BN-tag-loss_valid_cr_loss.csv"
cosmic_conn_256 = "paper_utils/tensorboard_logs/run-2021_03_14_16_47_LCO_Cosmic-CoNN_256px-tag-loss_valid_cr_loss.csv"
deepCR_256 = "paper_utils/tensorboard_logs/run-2021_03_14_16_42_LCO_deepCR_continue-tag-loss_valid_cr_loss.csv"


max_epoch = 5000
epochs = np.linspace(0, max_epoch, max_epoch + 1)

cosmic_conn_1024 = extract_csv(cosmic_conn_1024, max_epoch, True)
cosmic_conn_256 = extract_csv(cosmic_conn_256, max_epoch)
cosmic_conn_BN = extract_csv(cosmic_conn_BN, max_epoch)  # correctly scaled
deepCR_256 = extract_csv(deepCR_256, max_epoch)

# plotting

plt.rcParams.update({"font.size": 12})
f = plt.figure(figsize=(12, 4))
ax = f.add_subplot()
# fig, ax = plt.subplots()
width = 0.8
linewidth = 1.5

mask = np.isfinite(cosmic_conn_1024)
ax.plot(
    epochs[mask],
    cosmic_conn_1024[mask],
    color="tab:orange",
    label="(1024px) Cosmic-CoNN",
    linewidth=linewidth,
    linestyle="-",
)

mask = np.isfinite(cosmic_conn_BN)
ax.plot(
    epochs[mask],
    cosmic_conn_BN[mask],
    color="tab:orange",
    label="(1024px) Cosmic-CoNN w/ BN",
    linewidth=linewidth,
    linestyle="--",
)

mask = np.isfinite(cosmic_conn_256)
ax.plot(
    epochs[mask],
    cosmic_conn_256[mask],
    color="tab:blue",
    label="(256px) Cosmic-CoNN",
    linewidth=linewidth,
    linestyle="-",
)

mask = np.isfinite(deepCR_256)
ax.plot(
    epochs[mask],
    deepCR_256[mask],
    color="tab:blue",
    label="(256px) deepCR",
    linewidth=linewidth,
    linestyle="--",
)

ax.set_ylabel("1 - Dice score")
ax.set_xlabel("epochs (defined in Appendix C)")

ax.legend()

ax.set_yscale('log')
ax.set_yticks([0.1, 0.2, 0.3, 0.5, 0.8, 1.0])
ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())


plt.grid(True)
plt.xlim(0, epochs[-1])
min = 0.08
plt.ylim(min, 1)


plt.savefig(save_path, bbox_inches="tight")
