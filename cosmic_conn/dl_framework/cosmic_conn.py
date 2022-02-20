"""
Main file to define the Cosmic-CoNN pytorch class.
CY Xu (cxu@ucsb.edu)
"""

import os
import math
import random
import logging

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


# model import
from cosmic_conn.dl_framework.utils_ml import (
    clean_large,
    memory_check,
    modulus_boundary_crop,
    subtract_sky,
    median_weighted_bce,
    tensor2np,
    DiceLoss,
    remove_nan
)
from cosmic_conn.dl_framework.unet import UNet_module
from cosmic_conn.cr_pipeline.utils_img import save_as_png


class Cosmic_CoNN(nn.Module):
    def __init__(self):
        super(Cosmic_CoNN, self).__init__()

    def name(self):
        return "Unsupervised_CR"

    def initialize(self, opt):
        self.opt = opt
        self.best_name = "epoch_0"
        self.last_best_model = None
        self.best_valid_loss = float("inf")
        self.vis = []

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("...GPU found, yeah!")
            
        else:
            self.device = torch.device("cpu")
            logging.info("...GPU or CUDA not detected, using CPU (slower). ")
            logging.info("...training on CPU is not recommended.")

        self.model_dir = os.path.join(opt.expr_dir, "models")
        self.valid_dir = os.path.join(opt.expr_dir, "validation")

        if not opt.load_model:
            os.makedirs(self.valid_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)

        self.build_models(opt)

    def build_models(self, opt):
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_dtype(torch.float32)

        # group norm parameters
        norm_setting = [opt.n_group, opt.gn_channel, opt.no_affine]

        # the network is defined here
        self.network = UNet_module(
                n_channels=1,
                n_classes=1,
                hidden=opt.hidden,
                norm=opt.norm,
                norm_setting=norm_setting,
                conv_type=opt.conv_type,
                down_type=opt.down_type,
                up_type=opt.up_type,
                deeper=opt.deeper,
            )

        self.network.to(self.device)

        # count trainlearnableable parameteres
        # model_parameters = filter(lambda p: p.requires_grad, self.network.parameters())
        # total_params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f"Total learnable parameters {total_params}")

        if opt.mode == "train":

            # init common loss functions
            if opt.loss == "bce":
                self.criterion = nn.BCELoss()

            elif opt.loss == "median_bce":
                self.criterion = median_weighted_bce
                self.bce = nn.BCELoss()

            elif opt.loss == "mse":
                self.criterion = nn.MSELoss()

            elif opt.loss == "dice":
                self.criterion = DiceLoss

            else:
                raise ValueError('Invalid loss function, choose from \
                                bce, median_bce, mse, dice')

            # optimizer
            params = self.network.parameters()

            self.optimizer = optim.Adam(params, lr=opt.lr)
            # self.optimizer = optim.SGD(self.network.parameters(), lr=opt.lr)

            # init network
            self.apply(self.weights_init("kaiming"))

            # Multi Step Scheduler
            if opt.milestones[0] == '0':
                milestones = []
            else:
                milestones = [int(i) for i in opt.milestones]

            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=0.1
            )

            # only import tensorboard for training
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    "Please run `pip install cosmic-conn[develop]` to install all packages for development."
                )

        # Inference or continue training

        if opt.mode == "inference":
            # inference only

            if len(opt.load_model) == 0:
                raise ValueError("Please specify the path to load model file -m.")

            checkpoint = torch.load(opt.load_model, map_location=self.device)

            if opt.load_model.endswith("pth"):
                self.network.load_state_dict(checkpoint)
            else:
                self.network.load_state_dict(checkpoint["state_dict_mask"])

            # use available memory to determine detection method
            # self.full_image_detection = memory_check(self.device)

        elif opt.mode == "train" and opt.continue_train:
            # initialize with previously saved model, but new optimizer
            if opt.load_model:
                checkpoint_path = opt.load_model

            elif opt.continue_epoch > 0:
                checkpoint_path = os.path.join(
                    self.model_dir, f"cr_checkpoint_{opt.continue_epoch}.pth.tar"
                )

            else:
                raise ValueError(
                    "To continue traininng, must provid checkpoint directory in \
                    opt.continue_train or model path in opt.load_model"
                )

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if opt.load_model:
                if opt.load_model.endswith("pth"):
                    self.network.load_state_dict(checkpoint)
                else:
                    self.network.load_state_dict(checkpoint["state_dict_mask"])

            # continue with previously saved model and optimizer
            elif opt.continue_epoch > 0:
                self.network.load_state_dict(checkpoint["state_dict_mask"])
                self.optimizer.load_state_dict(checkpoint["optimizer_mask"])

                # fast forward scheduler to adjust LR rate properly
                for i in range(opt.continue_epoch):
                    self.lr_scheduler.step()
                print(f'LR scheduler fastforward to {opt.continue_epoch}')

            self.writer = SummaryWriter(
                log_dir=str(opt.expr_dir),
                filename_suffix=f"continue_{opt.continue_epoch}",
                purge_step=opt.continue_epoch + 1,
            )

        else:
            # start new training
            self.writer = SummaryWriter(log_dir=str(opt.expr_dir))

    def forward(self, noisy):
        return self.network(noisy)

    def detect_full_image(self, image):
        # try full image detecction first
        mask = torch.zeros_like(image)
        frame_ = modulus_boundary_crop(image, modulo=16)
        tensor = self.dim_shift(frame_).to(self.device)
        tensor -= tensor.median()
        pdt = self.network(tensor).squeeze()
        mask[: pdt.shape[0], : pdt.shape[1]] = pdt
        return mask

    def detect_image_stamps(self, image, crop=1024):
        # if not enough memory, detect in smaller stamps
        mask = None

        # by defualt we use smaller stamp size as memory safeguard
        stamp_sizes = [1024, 512, 256] if crop==1024 else [crop]

        for stamp in stamp_sizes:
            try:
                torch.cuda.empty_cache()

                mask = clean_large(
                    image, self.network, patch=stamp, overlap=0
                )
            except:
                logging.warning(f"...{stamp}x{stamp} stamp won't fit into memory.")

            if mask is not None:
                break
        
        if mask is None:
            msg = "...detection failed. Memory too small?"
            logging.error(msg)
            raise ValueError(msg)

        return mask

    # @torch.jit.export
    def detect_cr(self, image, ret_numpy=True):
        # replace NaN with 0.0 if exist
        image = remove_nan(image)

        # numpy array -> tensor
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image).float()

        # no gradient saved during inference
        with torch.no_grad():
            if self.opt.crop == 0:
                # full image detection as user requested
                mask = self.detect_full_image(image)
            else:
                # slice image into smaller stamps
                mask = self.detect_image_stamps(image, self.opt.crop)

        if ret_numpy:
            return tensor2np(mask)
        else:
            return mask

    # @profile
    def update_model(self, data, epoch):
        self.epoch = epoch
        self.optimizer.zero_grad()

        # special case, train in eval() for batch norm
        if self.opt.norm == "batch" and epoch > self.opt.eval_epoch:
            self.eval()
        else:
            self.train()

        # dynamically calculate the imbalance alpha for median weighted BCE
        if epoch <= 1:
            self.imbalance_alpha = 0.0
        else:
            self.imbalance_alpha = min(epoch / self.opt.imbalance_alpha, 1.0)

        # forward and backward propogate one sample at a time, no update
        train_loss = self.forward_pass(data)

        train_loss.backward()
        self.optimizer.step()

        return train_loss

    def forward_pass(self, data):

        # mask_src is bad_mask when it's HST data
        noisy, mask_ref, medians, mask_ignore, filenames = self.prep_data(data)

        mask_pdt = None

        # 1) learn reference mask
        mask_pdt = self.network(noisy)

        # original deepCR implementation ignored gradients from ignore mask
        if "hst" in filenames[0]:
            mask_pdt = mask_pdt * (1 - mask_ignore)
            mask_ref = mask_ref * (1 - mask_ignore)

        if self.opt.loss == "median_bce":

            if self.imbalance_alpha < 1.0:
                loss_ref = self.criterion(
                    mask_pdt, mask_ref, medians, self.imbalance_alpha
                )
            else:
                loss_ref = self.bce(mask_pdt, mask_ref)

        else:
            loss_ref = self.criterion(mask_pdt, mask_ref)

        # for debug, reivew training images
        # self.vis.append([
        #     tensor2np(noisy.squeeze()[0]),
        #     tensor2np(medians.squeeze()[0]),
        #     tensor2np(mask_ref.squeeze()[0]),
        #     tensor2np(mask_pdt.squeeze()[0]), filenames[0]
        # ])

        return loss_ref

    def prep_data(self, data, validate=False):
        """
        data input shape: (batch, patch, frm, h, w)
        validate input shape: (1, frm, h, w)
        output shape (batch * patch, 1, h, w)
        """
        if len(data) == 5:
            frames, cr_masks, ignore_masks, medians, filenames = data
        else:
            # ignore_mask is not used for training for LCO data
            frames, cr_masks, ignore_masks, filenames = data

        # send to gpu for faster median frame calculation
        frames = frames.to(self.device)

        if isinstance(filenames[0], list) or isinstance(filenames[0], tuple):
            filenames = [item for sublist in filenames for item in sublist]

        frames = subtract_sky(frames, remove_negative=False)

        # HST data, no need for following steps
        if "hst" in filenames[0]:
            batch, frm, h, w = frames.shape
            patch = 1
            # reshape to 4 dimensions
            frames = frames.view(-1, 1, h, w)
            cr_masks = cr_masks.view(-1, 1, h, w)

            ignore_masks = ignore_masks.view(-1, 1, h, w)
            medians = medians.view(-1, 1, h, w)

        elif "lco" in filenames[0] or "nres" in filenames[0]:
            # extra processing needed fro LCO data
            batch, frm, h, w = frames.shape

            # calculate median on GPU is faster, -3 is the frame axis in both cases
            medians = torch.median(frames, dim=-3)[0].view(-1, 1, h, w)

            if validate:
                frames = frames.view(-1, 1, h, w)
                cr_masks = cr_masks.view(-1, 1, h, w)
                ignore_masks = ignore_masks.view(-1, 1, h, w)
                ignore_masks[ignore_masks > 0.0] = 1.0

                if len(filenames) != frames.shape[0]:
                    filenames = filenames * frames.shape[0]

            else:
                # extract same frame for both image and cr_mask
                frames, cr_masks = self.extract_random_frame(frames, cr_masks)
                ignore_masks = None

        cr_masks = cr_masks.to(self.device)
        medians = medians.to(self.device)

        if ignore_masks is not None:
            ignore_masks = ignore_masks.to(self.device)

        return [frames, cr_masks, medians, ignore_masks, filenames]

    def extract_random_frame(self, frames, cr_masks):
        # randomly pick one frame from a sequence of three frames
        b, c, h, w = frames.shape
        idx = random.randint(0, c - 1)

        frames = [frames[i, idx] for i in range(b)]
        cr_masks = [cr_masks[i, idx] for i in range(b)]

        frames = torch.stack(frames).view(-1, 1, h, w)
        cr_masks = torch.stack(cr_masks).view(-1, 1, h, w)

        return frames, cr_masks

    # @profile
    def validate(self, valid_loader, epoch, source):
        loss_dict = {}
        loss_key = f"loss/valid_cr_loss_{source}"
        self.eval()
        self.epoch_dir = os.path.join(self.valid_dir, f"epoch_{epoch}")

        # DICE evaluation
        with torch.no_grad():
            # load each fits file
            for i, data in enumerate(valid_loader):

                frames, cr_masks, medians, ignore_masks, filenames = self.prep_data(
                    data, validate=True
                )

                b, c, h, w = frames.shape
                pdt_masks = torch.zeros_like(frames)

                for j in range(b):
                    pdt = self.detect_image_stamps(frames[j].squeeze())
                    pdt_masks[j] = pdt.unsqueeze(dim=0)

                # pdt_masks = self.detect_image_stamps(frames)

                # to correct the training set bias for neutral evaluation
                pdt_masks = pdt_masks * (1 - ignore_masks)
                cr_masks = cr_masks * (1 - ignore_masks)

                # validation loss might be different from training
                # so different training loss could be evaluated at the same standard
                cr_loss = DiceLoss(pdt_masks, cr_masks)

                loss_dict[loss_key] = loss_dict.get(loss_key, 0)\
                    + cr_loss.item()

            # save prediction png for preview
            for j in range(b):
                median = medians.squeeze(
                )[j] if source == "hst" else medians.squeeze()

                self.vis.append(
                    [
                        tensor2np(frames.squeeze()[j]),
                        tensor2np(median),
                        tensor2np(cr_masks.squeeze()[j]),
                        tensor2np(pdt_masks.squeeze()[j]),
                        filenames[j],
                    ]
                )

        # normalize to 1.
        loss_dict[loss_key] = loss_dict[loss_key] / len(valid_loader)

        # sum all losses for a mixed model
        loss_dict["loss/valid_cr_loss"] = (
            loss_dict.get("loss/valid_cr_loss", 0) + loss_dict[loss_key]
        )

        self.loss_dict = loss_dict
        return loss_dict

    def save_checkpoint(self, epoch):

        # save best checkpoing for continue training
        state = {
            "epoch": epoch,
            "state_dict_mask": self.network.state_dict(),
            "optimizer_mask": self.optimizer.state_dict(),
        }

        # save best model for inference
        model_path = f"{self.model_dir}/cr_checkpoint_{epoch}.pth.tar"
        torch.save(state, model_path)

        return model_path

    def get_current_visuals(self, epoch):
        os.makedirs(self.epoch_dir, exist_ok=True)
        root = self.epoch_dir

        for vis in self.vis:
            frm, med, msk_ref, msk_pdt, fname = vis
            text = self.opt.comment if "lco" in fname else None

            _, z1z2 = save_as_png(
                med, root, fname, file_type="1_median", zscale=True, grid=True,
            )

            save_as_png(
                frm,
                root,
                fname,
                file_type="2_noisy",
                zscale=True,
                z1z2=z1z2,
                grid=True,
            )

            save_as_png(msk_ref, root, fname,
                        file_type="4_mask_ref", grid=True)

            save_as_png(msk_pdt, root, fname,
                        file_type="5_mask_pdt", grid=True)

        # reset global vis list after all files saved
        self.vis = []

    def update_learning_rate(self, epoch):

        # log LR first, real current epoch LR
        lr_mask = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("para/lr_mask", lr_mask, epoch)

        self.lr_scheduler.step()

    def weights_init(self, init_type="gaussian"):
        def init_fun(m):
            classname = m.__class__.__name__
            if (
                classname.find("Conv") == 0 or classname.find("Linear") == 0
            ) and hasattr(m, "weight"):
                # print m.__class__.__name__
                if init_type == "gaussian":
                    init.normal_(m.weight.data, 0.0, 0.02)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif init_type == "orthogonal":
                    init.orthogonal_(m.weight.data, gain=math.sqrt(2))
                elif init_type == "default":
                    pass
                else:
                    assert 0, "Unsupported initialization: {}".format(
                        init_type)
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        return init_fun

    # tensorboard logging
    def log_value(self, key, value, epoch):
        self.writer.add_scalar(key, value, epoch)

    def log_dict(self, loss_dict, epoch):
        for key, value in loss_dict.items():
            self.writer.add_scalar(key, value, epoch)

    def dim_shift(self, tensor):
        tensor = tensor.squeeze()

        if len(tensor.shape) == 4:
            return tensor
        elif len(tensor.shape) == 3:
            return tensor.unsqueeze(dim=1)
        elif len(tensor.shape) == 2:
            return tensor.unsqueeze(dim=0).unsqueeze(dim=0)
