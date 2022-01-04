"""
Cosmic-CoNN model training options.
CY Xu (cxu@ucsb.edu)
"""

import argparse
import os
import datetime

class ModelOptions():
    def __init__(self):
        # init default model options
        
        # training and model settings
        self.__dict__["data"] = "./"
        self.__dict__["max_train_size"] = 0
        self.__dict__["mode"] = "inference"
        self.__dict__["model"] = "lco"
        self.__dict__["loss"] = 0
        self.__dict__["imbalance_alpha"] = 100.0
        self.__dict__["norm"] = "group"
        self.__dict__["no_affine"] = False
        self.__dict__["n_group"] = 8
        self.__dict__["gn_channel"] = 0
        self.__dict__["conv_type"] = "unet"
        self.__dict__["up_type"] = "deconv"
        self.__dict__["down_type"] = "maxpool"
        self.__dict__["deeper"] = False
        self.__dict__["crop"] = 1024
        self.__dict__["batch"] = 4
        self.__dict__["hidden"] = 32
        self.__dict__["lr"] = 0.001
        self.__dict__["eval_epoch"] = 10
        self.__dict__["min_exposure"] = 0.0
        self.__dict__["epoch"] = 3000
        self.__dict__["milestones"] = ['0']
        self.__dict__["seed"] = 0
        self.__dict__["random_seed"] = False
        self.__dict__["comment"] = ""

        # validation/test settings
        self.__dict__["load_model"] = ""
        self.__dict__["validate_freq"] = 1
        self.__dict__["validRatio"] = 0.2
        self.__dict__["max_valid_size"] = 0
        self.__dict__["valid_crop"] = 2000

        # continue training
        self.__dict__["continue_train"] = ""
        self.__dict__["continue_epoch"] = 0

        self.__dict__["expr_dir"] = "./"
        self.__dict__["log_file"] = ""
        self.__dict__["prefix"] = ""

        # inference settings
        self.__dict__["app"] = False
        self.__dict__["input"] = "./"
        self.__dict__["ext"] = "SCI"
        self.__dict__["verbose"] = False


    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()


class TrainingOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # training and model settings
        self.parser.add_argument(
            "--data",
            default=os.path.join(os.getcwd(), "data"),
            help="path to data directory",
        )

        self.parser.add_argument(
            "--max_train_size",
            type=int,
            default=0,
            help="the # of samples randomly draw in each epoch, 0 uses entire dataset",
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="train",
            help="train | inference, inference mode does not create checkpoint or log",
        )
        self.parser.add_argument(
            "--model",
            type=str,
            default="lco",
            help="lco | hst | nres, dataset specific dataloader",
        )
        self.parser.add_argument(
            "--loss",
            type=str,
            default="median_bce",
            help="bce | median_bce | dice | mse, loss function used for training",
        )
        self.parser.add_argument(
            "--imbalance_alpha",
            type=float,
            default=100.0,
            help="number of iterations for the Median Weighted BCE Loss to linearly increase the lower bound alpha to 1. See paper for detail.",
        )
        self.parser.add_argument(
            "--norm",
            type=str,
            default="group",
            help="batch | group | instance, feature normalization method",
        )
        self.parser.add_argument(
            "--no_affine",
            action="store_true",
            help="do not use affine transformation in normalization",
        )
        self.parser.add_argument(
            "--n_group", type=int, default=8, help="fixed group number for group normalization "
        )
        self.parser.add_argument(
            "--gn_channel",
            type=int,
            default=0,
            help="fixed channel number, >0 will override n_group, 0 uses fixed group number",
        )
        self.parser.add_argument(
            "--conv_type", type=str, default="unet", help="unet | resnet, types for convolution module"
        )
        self.parser.add_argument(
            "--up_type", type=str, default="deconv", help="deconv | upscale, types for deconvolution module"
        )
        self.parser.add_argument(
            "--down_type",
            type=str,
            default="maxpool",
            help="maxpool | avgpool | stride, types for the pooling layer",
        )
        self.parser.add_argument(
            "--deeper",
            action="store_true",
            help="deeper network, one more downsample and upsample layer",
        )
        self.parser.add_argument(
            "--crop", type=int, default=1024, help="training input stamp size"
        )
        self.parser.add_argument(
            "--batch",
            type=int,
            default=4,
            help="training batch size",
        )
        self.parser.add_argument(
            "--hidden", type=int, default=32, help="channel # of first conv layer"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.001, help="learning rate, 0.001 by default"
        )
        self.parser.add_argument(
            "--eval_epoch", type=int, default=10, help="# of phase0 training epochs, only applies to batch normalization"
        )
        self.parser.add_argument(
            "--min_exposure",
            type=float,
            default=0.0,
            help="minimum exposure time when sampling training data",
        )

        self.parser.add_argument(
            "--epoch", type=int, default=3000, help="total training epochs")
        self.parser.add_argument(
            "--milestones",
            nargs="+",
            default=['0'],
            help="milesstones to reduce the learning rate, e.g. '1000 2000 3000', '0' keeps LR constant",
        )
        self.parser.add_argument(
            "--seed", type=int, default=0, help="assign manual seed")
        self.parser.add_argument(
            "--random_seed", action="store_true", help="it will initialize the model multiple times to find a best random seed if flagged"
        )
        self.parser.add_argument(
            "--comment", default="", help="comment is appended to the checkpoint directory name")

        # validation/test settings
        self.parser.add_argument(
            "--load_model", default="", help="path to load a model for inference model")
        self.parser.add_argument(
            "--validate_freq", type=int, default=1, help="number per epochs to perform model validation")
        self.parser.add_argument(
            "--validRatio", type=float, default=0.2, help="the ratio of training data reserved for validation, 0.2 by default")
        self.parser.add_argument(
            "--max_valid_size", type=int, default=0, help="the number of sample reserved for validation, >0 will override validRatio")
        self.parser.add_argument(
            "--valid_crop", type=int, default=2000, help="stamp size for the center-cropping during validation")

        # continue training
        self.parser.add_argument(
            "--continue_train", type=str, default="", help="to continue a previous training, provide the checkpoint directory name"
        )
        self.parser.add_argument(
            "--continue_epoch", type=int, default=0, help="the number of epoch to continue")

        self.initialized = True
        self.opt = self.parser.parse_args()

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt.expr_dir, self.opt.prefix = self.mkCheckpointDir(self.opt)

        args = vars(self.opt)

        if self.opt.mode == "train":

            print("------------ Options -------------")
            for k, v in args.items():
                print("%s: %s" % (str(k), str(v)))
            print("-------------- End ----------------")

            if self.opt.continue_train:
                file_name = os.path.join(
                    self.opt.expr_dir,
                    f"opt_train_continue_{self.opt.continue_epoch}.txt",
                )

            if not (self.opt.continue_train or self.opt.load_model):
                # save to the disk
                file_name = os.path.join(self.opt.expr_dir, "opt_train.txt")

            with open(file_name, "wt") as opt_file:
                opt_file.write("------------ Options -------------\n")
                for k, v in args.items():
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")

            self.opt.log_file = file_name

        # swtich to custom class from argparse dict
        model_options = ModelOptions()
        for k, v in args.items():
            model_options[k] = v

        return model_options


    def mkCheckpointDir(self, opt):
        suffix = opt.comment
        continue_train = opt.continue_train

        if continue_train:
            directory = continue_train
            path = f"./checkpoints/{continue_train}"

        else:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y_%m_%d_%H_%M")
            directory = str(timestamp) + "_" + suffix
            path = f"./checkpoints/{directory}"

            if self.opt.mode == "train":
                os.makedirs(path, exist_ok=True)

        return path, directory
