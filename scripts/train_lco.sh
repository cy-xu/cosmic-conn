#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
--data /home/cyxu/astro/Cosmic_ConNN_datasets/LCO_CR_dataset/train_set \
--max_train_size 70 \
--validRatio 0.2 \
--max_valid_size 0 \
--model lco \
--loss median_bce \
--imbalance_alpha 100 \
--batch 10 \
--crop 1024 \
--eval_epoch 200 \
--hidden 32 \
--conv_type unet \
--down_type maxpool \
--up_type deconv \
--norm group \
--n_group 8 \
--gn_channel 0 \
--lr 1e-3 \
--epoch 30000 \
--milestones 30000 \
--valid_crop 2000 \
--validate_freq 1 \
--seed 0 \
--comment LCO_Cosmic-Conn_test


# --data data/LCO_CR_dataset \
# --continue_train 2021_02_27_11_47_LCO_Cosmic-Conn \
# --continue_epoch 9000 \
# --random_seed \
