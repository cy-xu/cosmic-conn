# -*- coding: utf-8 -*-

"""
Main entry point for the Cosmic-CoNN deep learning model training.
`$ bash scripts/train_lco.sh` is an example script.
CY Xu (cxu@ucsb.edu)
"""

import sys
import random
import time
import numpy as np

import torch
import torch.backends.cudnn as cudnn

try:
    from cosmic_conn.dl_framework.options import TrainingOptions
    from cosmic_conn.dl_framework.dataloader import CreateDataLoader
    from cosmic_conn.dl_framework.cosmic_conn import Cosmic_CoNN
except ImportError:
    raise ImportError(
        "Please run `pip install cosmic-conn[develop]` to install all packages for development."
        )


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def search_best_seed(opt, loss_key, candidates=30):
    loss_best = float("inf")
    model_best_path = None
    seed_best = 0

    _, _, valid_loader = CreateDataLoader(opt)

    for i in range(candidates):
        randseed = np.random.randint(0, 2 ** 32)
        setup_seed(randseed)

        model = Cosmic_CoNN()
        model.initialize(opt)
        loss_dict = model.validate(valid_loader, epoch=0, source=opt.model)

        if loss_dict[loss_key] <= loss_best:
            loss_best = loss_dict[loss_key]
            model_best_path = model.save_checkpoint(epoch=0)
            seed_best = randseed
            print("best model")

        del model
        print(f"randseed {randseed}, loss {loss_dict[loss_key]}")

    opt.load_model = model_best_path

    return opt, seed_best


def main():
    # get training arguments/options
    train_options = TrainingOptions()
    opt = train_options.parse()
    loss_key = "loss/valid_cr_loss"

    # deterministic flags for model reproducibility
    cudnn.deterministic = True
    cudnn.benchmark = False

    if opt.random_seed:
        # https://github.com/pytorch/pytorch/issues/12873
        # cudnn.enabled = True
        # cudnn.benchmark = True
        opt, randseed = search_best_seed(opt, loss_key, 30)
    else:
        randseed = opt.seed

    # seed everything before instances are created
    setup_seed(randseed)

    data_loader, train_set, valid_data_loader = CreateDataLoader(opt)
    model = Cosmic_CoNN()
    model.initialize(opt)

    # logging
    with open(opt.log_file, "a") as log_file:
        now = time.strftime("%c")
        log_file.write(f"random seed: {randseed}")
        log_file.write(
            f"\n================ Training Loss ({now}) ================\n")

    # epoch loop, continue_epoch is 0 by default
    for e in range(0, opt.epoch + 1, 1):
        tic = time.perf_counter()
        train_loss = 0.0
        loss_dict = {}

        # load NEXT subset of data to memory (parallel)
        train_set.sample_subset()

        # repeat sampling process so data dataloader
        # have same randomness when continue from a previous job
        if e <= opt.continue_epoch:
            continue
        else:
            log_file = open(opt.log_file, "a")
            log_file.write(f"epoch {e}\n")
            print(f"\nepoch {e}")

        # sample loop
        for i, data in enumerate(data_loader):
            loss = model.update_model(data, e)
            train_loss += loss.item()

        train_loss /= len(data_loader)
        toc = time.perf_counter()

        # logging
        model.log_value("loss/train_loss", train_loss, e)
        log = f"train loss {round(train_loss, 10)}, time {round(toc-tic)}s"
        log_file.write(log + "\n")
        print(log)

        # validation
        if e % opt.validate_freq == 0:
            loss_dict = model.validate(valid_data_loader, e, source=opt.model)
            model.log_dict(loss_dict, e)

            # best validation loss and save model
            if loss_dict[loss_key] <= model.best_valid_loss:
                model.best_valid_loss = loss_dict[loss_key]
                model.get_current_visuals(e)
                # self.test_model(epoch)

                log = f"new best model at epoch {e}"
                log_file.write(log + "\n")
                print(log)

            # save checkpoint every epoch
            model.best_model_path = model.save_checkpoint(e)

            # logging
            valid_loss = round(loss_dict[loss_key], 10)
            log = f"valid loss {valid_loss}, time {round(time.perf_counter()-toc)}s"
            log_file.write(log + "\n")
            print(log)

        # update LR at end of each epoch
        model.update_learning_rate(e)
        log_file.write("\n")
        log_file.close()
        print()

    sys.exit("Finish training")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
