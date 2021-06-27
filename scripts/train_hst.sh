#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cosmic_conn/train.py \
--data data/HST_deepCR \
--max_train_size 0 \
--validRatio 0.2 \
--max_valid_size 0 \
--model lco \
--loss bce \
--imbalance_alpha 10 \
--batch 160 \
--crop 256 \
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
--validate_freq 10 \
--seed 0 \
--comment HST_Cosmic-CoNN
