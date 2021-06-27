"""
Defines dataloader for CR datasets of various types of data.
CY Xu (cxu@ucsb.edu)
"""

import os
import random
import numpy as np

from astropy.io import fits
from astropy.io.fits import getval
from torch.utils.data import Dataset, DataLoader

from cosmic_conn.dl_framework.utils_ml import (
    _worker_init_fn,
    center_crop_npy,
    erase_boundary_np,
    modulus_boundary_crop,
)


def is_fits_file(filename):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in ["fits", "fz"])


def find_all_fits(opt, min_exposure=100.0):
    # key_word = 'banzai_frms' if opt.load_model else 'masked_fits'
    paths, file_names = [], []
    skipped = 0

    for root, dirs, files in os.walk(opt.data):
        if "masked_fits" in root and len(files) > 0:
            files = [f for f in files if not f[0] == "."]
            dirs[:] = [d for d in dirs if not d[0] == "."]
            files.sort()
            for f in files:
                p = os.path.join(root, f)
                if is_fits_file(p):
                    try:
                        try:
                            exp_time = getval(p, "EXPTIME", 1)
                        except:
                            # Gemini header
                            exp_time = getval(p, "EXPTIME", 0)
                        if exp_time > min_exposure:
                            paths.append(p)
                            file_names.append(f)
                        else:
                            skipped += 1
                    except:
                        print(f"problem reading fits: {p}")
                        # os.remove(p)

    path_name_pairs = list(zip(paths, file_names))
    print(f"{skipped} fits skipped for exposure time shorter than {min_exposure}")

    return path_name_pairs


# @profile
def CreateDataLoader(opt):

    if opt.model in ["mixed", "lco", "nres"]:
        print(f"loading LCO data...")
        all_samples = find_all_fits(opt, min_exposure=opt.min_exposure)
        assert len(all_samples) > 0, \
            f"No valid training data found in {opt.data}"

    if opt.model in ["mixed", "hst"]:
        print(f"loading HST data...")
        all_samples = read_HST_patches(opt.data)

    if opt.max_valid_size > 0:
        valid_size = \
            min(opt.max_valid_size, int(opt.validRatio * len(all_samples)))
    else:
        valid_size = int(opt.validRatio * len(all_samples))

    train_size = len(all_samples) - valid_size

    # in-place shuffle
    random.shuffle(all_samples)

    train_samples = all_samples[:train_size]
    valid_samples = all_samples[train_size:]

    if valid_size < 1:
        valid_samples = train_samples

    # when creating dataload only for evaluations
    if train_size > 0:

        if opt.model == "lco":
            train_set = LCO_Dataset(opt, train_samples)

        elif opt.model == "nres":
            train_set = LCO_NRES_Dataset(opt, train_samples)

        # created HST and LCO dataest and a fused dataset
        elif opt.model == "hst":
            train_set = HST_dataset(
                train_samples,
                mini_batch=1,
                augment=5,
                max_size=opt.max_train_size,
                mode="train",
            )

        else:
            raise ValueError(f"uknown training model {opt.model}")

        # create training dataloader
        train_data_loader = DataLoader(
            dataset=train_set,
            batch_size=opt.batch,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
        )

    else:
        train_data_loader = None

    # Create validation set and data loader
    if opt.model == "lco":
        valid_set = LCO_Dataset(opt, valid_samples, validate=True)

    elif opt.model == "hst":
        valid_set = HST_dataset(
            valid_samples,
            mini_batch=64,
            augment=None,
            max_size=opt.max_valid_size,
            mode="validate",
        )

    elif opt.model == "nres":
        valid_set = LCO_NRES_Dataset(opt, valid_samples, validate=True)

    else:
        raise ValueError(f"uknown training model {opt.model}")

    if valid_set:
        valid_batch = 16 if opt.model == "hst" else 1

        valid_data_loader = DataLoader(
            dataset=valid_set,
            batch_size=valid_batch,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

    return train_data_loader, train_set, valid_data_loader


def read_HST_patches(path):
    all_paths = []

    for root, dirs, files in os.walk(path):

        files = [f for f in files if not f[0] == "."]
        files = [f for f in files if f.endswith("npz")]
        files.sort()

        for f in files:
            all_paths.append(os.path.join(root, f))

    assert len(all_paths) > 0, f"no HST image found, check {path}"

    return all_paths


# modified by CY
def identical_random_crop(array, size, crop_indices=None):
    h, w = array.shape
    rangew = (w - size) // 2 if w > size else 0
    rangeh = (h - size) // 2 if h > size else 0

    if crop_indices is None:
        offsetw = np.random.randint(0, rangew)
        offseth = np.random.randint(0, rangeh)
    else:
        offsetw = crop_indices[0]
        offseth = crop_indices[1]

    offsetw = 0 if rangew == 0 else offsetw
    offseth = 0 if rangeh == 0 else offseth

    cropped = array[offseth: offseth + size, offsetw: offsetw + size]

    if cropped.shape[-1] == 0:
        return array, [offsetw, offseth]
    else:
        return cropped, [offsetw, offseth]


def stack_identical_random_crop(
    ndarrays, crop_size, crop_indices=None, rotation=None, flip=None
):

    # if no crop_indices provided, create new from mask
    if crop_indices is None:
        patch, crop_indices = identical_random_crop(
            ndarrays[0], crop_size, None)

    # batch random crop frames and masks
    identical_crops = []

    for i in range(ndarrays.shape[0]):
        identical_crops.append(
            identical_random_crop(ndarrays[i], crop_size, crop_indices)[0]
        )

    # stack the same patch from different exposures
    cropped_patch = np.stack(identical_crops)

    # rotate the same patch of sky for data augmentation
    if rotation is not None:
        cropped_patch = np.rot90(cropped_patch, rotation, axes=(-2, -1))

    # flip for data augmentation
    if flip is not None:
        if flip == 0:
            cropped_patch = np.flipud(cropped_patch)
        if flip == 1:
            cropped_patch = np.fliplr(cropped_patch)

    return cropped_patch, crop_indices


def subtact_sky_remove_extreme(noisy_stack):
    # subtract sky and remove extreme low value from entire stack
    noisy_stack -= np.median(noisy_stack)
    lower_bound = np.mean(noisy_stack) - 3.5 * np.std(noisy_stack)
    noisy_stack[noisy_stack < lower_bound] = lower_bound
    return noisy_stack


def sample_cr_frames(frames, cr_frames, min_layers=0, max_layers=9):
    """
    return a randomly cropped CR frames overlay same shaep as frames
    """
    cr_overlay = np.zeros_like(frames)

    if max_layers == 0:
        return cr_overlay

    shape = frames.shape

    if len(shape) == 4:
        n_frames = shape[1]
        crop_shape = shape[-1]
    elif len(shape) == 3:
        n_frames = shape[0]
        crop_shape = shape[-1]
    else:
        raise ValueError("check input frames shape")

    for i in range(n_frames):
        # superimpose random layers of CR for each image
        n_layers = np.random.randint(min_layers, max_layers)
        cr_layers = np.random.choice(cr_frames, n_layers, replace=False)

        curr_overlay = np.zeros([crop_shape, crop_shape], dtype="float32")

        for j in range(n_layers):
            cr_layer = np.load(cr_layers[j], allow_pickle=True)
            # faster array operation without memory map...
            cr_layer = cr_layer["arr_0"]

            cr_crop, _ = identical_random_crop(cr_layer, crop_shape)

            # random rotate for data augmentation
            r = random.randint(0, 3)
            cr_crop = np.rot90(cr_crop, r, axes=(-2, -1))

            # random flip for data augmentation
            f = random.randint(0, 2)
            if f == 0:
                cr_crop = np.flipud(cr_crop)
            if f == 1:
                cr_crop = np.fliplr(cr_crop)

            curr_overlay += cr_crop

        if len(shape) == 4:
            cr_overlay[0, i] = curr_overlay

        if len(shape) == 3:
            cr_overlay[i] = curr_overlay

    return cr_overlay


def superimpose_cr(
    frames, cr_masks, cr_frames, min_layers=0, max_layers=9, source_mask=None
):
    """
    Superimpsoe frames and CR masks with identical cr_frames crops
    """
    cr_overlay = sample_cr_frames(frames, cr_frames, min_layers, max_layers)

    if source_mask is not None:
        cr_overlay = cr_overlay * source_mask

    frames += cr_overlay
    cr_masks = (cr_masks + cr_overlay > 0).astype("uint8")

    return frames, cr_masks


# LCO data helper function


def maximum_center_crop(frame_stack, crop_size):
    """
    crop non-overlapping patches from full frame for validation
    boundary removed for improve robustness
    :return:
    """
    assert (
        len(frame_stack.shape) == 3
    ), f"expecting input as a stack of 2D arrays [n, h, w]"

    n, h, w = frame_stack.shape
    cropped_patches = []

    if crop_size == 0:
        if h % 16 != 0 or w % 16 != 0:

            for i in range(n):
                cropped_patches.append(
                    modulus_boundary_crop(frame_stack[i], modulo=16)
                )

            return np.stack(cropped_patches)

        else:
            return frame_stack

    else:
        if h < crop_size or w < crop_size:
            crop_size = min(h, w)

        for i in range(n):
            cropped_patches.append(
                center_crop_npy(frame_stack[i], crop_size=crop_size)
            )

        return np.stack(cropped_patches)


def sample_frames_from_hdul(key, full_set, validate, opt):
    """
    LCO training data fits extension (example of 3 frames)
    No.         Name                    Format
    0           PRIMARY
    1-3         SCI                     float32
    4           VALID_MASK              uint8
    5           CAT (SEP from banzai)
    6-8         CR_MASK                 uint8
    9-11        SOURCE_MASK             uint8
    """
    file_path = full_set[key]
    hdul = fits.open(file_path, lazy_load_hdus=False)

    # randomly sample 3 frames from a larger stack to save and maintain consistent GPU ram
    frmtotal = hdul[0].header["frmtotal"]

    subset = random.sample(range(frmtotal), 3)

    # skip hdul[0] primary frame
    frames = [hdul[1 + i].data for i in subset]

    # skip primary, valid_mask
    cr_masks = [hdul[3 + frmtotal + i].data for i in subset]

    frames = np.stack(frames, axis=0)
    cr_masks = np.stack(cr_masks, axis=0)

    if validate:
        ignore_masks = [hdul[3 + frmtotal + frmtotal + i].data for i in subset]
        ignore_masks = np.stack(ignore_masks, axis=0)

        headers = [dict(hdul[i + 1].header) for i in subset]
        file_names = ["lco_" + hdr["FILENAME"] for hdr in headers]

    hdul.close()

    # special handling certain telescopes with boundary issues
    # remove 100px around the boundary in image and mask
    edge_width = 100
    frames = erase_boundary_np(frames, edge_width)
    cr_masks = erase_boundary_np(cr_masks, edge_width)

    if validate:
        ignore_masks = erase_boundary_np(ignore_masks, edge_width)

        # center crop for validation data
        frames = maximum_center_crop(frames, opt.valid_crop)
        cr_masks = maximum_center_crop(cr_masks, opt.valid_crop)
        ignore_masks = maximum_center_crop(ignore_masks, opt.valid_crop)

    else:
        # generate identical rotation and flip seeds so both
        # frame and mask are rotated identically
        rotation = random.randint(0, 3)
        flip = random.randint(0, 2)

        # used for deepCR which did not use such augmentation
        if "deepcr" in opt.comment.lower():
            rotation = None
            flip = None

        # only crop training images
        frames, crop_indices = stack_identical_random_crop(
            frames, opt.crop, None, rotation, flip
        )

        cr_masks, _ = stack_identical_random_crop(
            cr_masks, opt.crop, crop_indices, rotation, flip
        )

        file_names = None
        ignore_masks = None

    return frames, cr_masks, ignore_masks, file_names


class LCO_Dataset(Dataset):
    def __init__(self, opt, samples, validate=False):
        self.opt = opt
        self.crop = opt.crop
        self.validate = validate

        # self.subset_paths, self.subset_fnames = zip(*samples)
        self.full_set = {x[1]: x[0] for x in samples}
        self.unused_set = list(self.full_set.keys())

        # self.full_set = samples
        # self.full_set_copy = samples.copy()

        # init and count actual frames
        # self.paths, self.file_names = zip(*self.full_set)
        self.dataset_size = 0

        for item in self.full_set.values():
            with fits.open(item) as hdul:
                self.dataset_size += hdul[0].header["frmtotal"]

        assert self.dataset_size > 0, f"No valid training data found in {self.opt.data}"

        # acquire subset size
        if validate:
            self.subset_size = len(self.full_set)
            print(
                f"Validation set total: {self.dataset_size} frames from {len(samples)} fits\n "
            )
        else:
            if self.opt.max_train_size > 0 and self.opt.max_train_size < len(
                self.full_set
            ):
                print(f"sampling {self.opt.max_train_size} fits each epoch")
                self.subset_size = self.opt.max_train_size
            else:
                print(f"shuffling full data set each epoch")
                self.subset_size = len(self.full_set)

            print(
                f"Training set total: {self.dataset_size} frames from {len(samples)} fits\n "
                f"{opt.crop}x{opt.crop} patch(es) for each fits, batch size {opt.batch} "
            )

        self.sample_subset()

    def __len__(self):
        return self.subset_size

    def sample_subset(self):
        if len(self.unused_set) < self.subset_size:
            self.unused_set = list(self.full_set.keys())

        # sample without duplicates
        self.subset_keys = np.random.choice(
            self.unused_set, self.subset_size, replace=False
        )

        # remove used files to ensure all fiels are used in one epoch
        for s in self.subset_keys:
            if s in self.unused_set:
                self.unused_set.remove(s)

    def __getitem__(self, index):
        # key is the file name
        key = self.subset_keys[index]

        frames, cr_masks, ignore_masks, file_names = sample_frames_from_hdul(
            key, self.full_set, self.validate, self.opt
        )

        cr_masks = cr_masks.astype("float32")
        frames = frames.astype("float32")

        if self.validate:
            ignore_masks = ignore_masks.astype("float32")
        else:
            file_names = "lco_" + key
            ignore_masks = cr_masks

        return frames, cr_masks, ignore_masks, file_names


class HST_dataset(Dataset):
    def __init__(
        self,
        paths=None,
        mini_batch=16,
        augment=None,
        random_nmask=True,
        aug_sky=[0, 0],
        max_size=0,
        mode="train",
    ):
        """Initializa the HSTdata class.

        Parameters:
            file: filename including directory
        """

        # 16x256x256 = 1x1024x1024, so each iteration has equal impact on the model
        self.mini_batch = mini_batch

        self.paths = paths
        self.augment = augment
        self.random_nmask = random_nmask
        self.aug_sky = aug_sky
        self.mode = mode

        # about 60 256x256 images equals one 2k by 2k iamge
        if mode == "evaluate":
            if max_size > 0 and max_size < len(self.paths):
                self.paths = self.paths[:max_size]

        if mode == "train" or mode == "validate":
            if max_size > 0 and max_size < len(self.paths):
                self.paths = random.sample(self.paths, max_size)

        self.dataset_size = len(self.paths)

    def __len__(self):
        return self.dataset_size

    def sample_subset(self):
        return 1

    def __getitem__(self, index):
        # since HST patches are only 256, draw more sample per batch
        if self.mode == "train":
            samples = random.sample(range(self.dataset_size), self.mini_batch)
        else:
            # then not in train mode
            samples = [index]

        raws, cleans, masks, badmasks = [], [], [], []
        gains, filenames = [], []

        for idx in samples:
            path = self.paths[idx]
            patch_npz = np.load(path)

            patch = {}
            for key in ["raw", "clean", "crmask", "badmask", "exp", "gain", "sky"]:
                patch[key] = patch_npz[key]

            exp, gain, sky = patch["exp"], patch["gain"], patch["sky"]

            # generate identical rotation and flip seeds so both
            # frame and mask are rotated identically
            rotation = random.randint(0, 3)
            flip = random.randint(0, 2)

            for key in ["raw", "clean", "crmask", "badmask"]:
                patch[key] = np.rot90(patch[key], rotation, axes=(-2, -1))

                # no flipping if flip == 2
                if flip == 0:
                    patch[key] = np.flipud(patch[key])
                if flip == 1:
                    patch[key] = np.fliplr(patch[key])

            raws.append(patch["raw"].astype("float32"))
            cleans.append(patch["clean"].astype("float32"))
            masks.append(patch["crmask"].astype("float32"))
            badmasks.append(patch["badmask"].astype("float32"))

            filenames.append("hst_" + os.path.basename(path)[:-4])
            gains.append(gain)

        # a = self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])
        # a = a * sky

        # if self.augment is not None:
        #     if self.random_nmask:
        #         size = np.random.randint(1, self.augment)
        #     else:
        #         size = self.augment
        #     index = np.random.randint(self.dataset_size, size=size)
        #     index = np.concatenate([index, np.array([i])])
        #     mask = self.crmask[index].sum(axis=0) > 0
        #     badmask = self.badmask[i].astype(bool) + self.crmask[i].astype(bool)

        raws = np.stack(raws)
        cleans = np.stack(cleans)
        masks = np.stack(masks)
        badmasks = np.stack(badmasks)

        return raws, masks, badmasks, cleans, filenames


class LCO_NRES_Dataset(Dataset):
    def __init__(self, opt, samples, validate=False):
        self.opt = opt
        self.crop = opt.crop
        self.full_set = samples
        self.validate = validate

        # init and count actual frames
        self.paths, self.file_names = zip(*samples)

        # count total frames
        self.dataset_size = 0
        for i in range(len(samples)):
            with fits.open(self.paths[i]) as hdul:
                self.dataset_size += hdul[0].header["frmtotal"]

        assert self.dataset_size > 0, f"No valid training data found in {self.opt.data}"

        if validate:
            print(
                f"Validation set: {self.dataset_size} frames from {len(samples)} fits"
            )
        else:
            print(
                f"Training set total: {self.dataset_size} frames from {len(samples)} fits\n "
                f"{opt.crop}x{opt.crop} patch(es) for each fits, batch size {opt.batch}"
            )

        # sample actual epoch subset
        self.sample_subset()

    def __len__(self):
        return len(self.paths)

    def sample_subset(self):
        # shuffle all data each epoch
        subset = random.sample(self.full_set, len(self.full_set))
        self.paths, self.file_names = zip(*subset)

    def sample_nres_frames(self, index):
        """
        LCO NRES training fits extension (example of 3 frames)
        No.         Name                    Format
        0           PRIMARY
        1-3         SCI                     int16
        4-6         CR_MASK                 uint8
        7-9         SOURCE_MASK             uint8
        """
        hdul = fits.open(self.paths[index], lazy_load_hdus=False)

        # randomly sample 3 frames from a larger stack to save and maintain consistent GPU ram
        frmtotal = hdul[0].header["frmtotal"]

        indices = random.sample(range(frmtotal), 3)

        # skip hdul[0] primary frame
        frames = [hdul[1 + i].data for i in indices]
        cr_masks = [hdul[1 + frmtotal + i].data for i in indices]

        # ignore mask is not used during training, only in validation
        if self.validate:
            ignore_masks = [
                hdul[1 + frmtotal + frmtotal + i].data for i in indices
                ]
            ignore_stack = np.stack(ignore_masks, axis=0)
        else:
            ignore_stack = None

        headers = [dict(hdul[i + 1].header) for i in indices]
        file_names = ["nres_" + hdr["FILENAME"] for hdr in headers]

        hdul.close()

        frames = np.stack(frames, axis=0)
        cr_masks = np.stack(cr_masks, axis=0)

        return frames, cr_masks, ignore_stack, file_names

    def __getitem__(self, index):
        frames, cr_masks, ignore_masks, file_names = \
            self.sample_nres_frames(index)

        filename = "nres_" + self.file_names[index]

        # special handling certain telescopes with boundary issues
        # remove 100px around the boundary in image and mask
        edge_width = 100

        frames = erase_boundary_np(frames, edge_width)
        cr_masks = erase_boundary_np(cr_masks, edge_width)

        # No random rotation for NRES data
        rotation = None
        flip = None

        # NRES validation happens on full frame, no cropping
        frames = frames.astype("float32")
        cr_masks = cr_masks.astype("float32")

        # ignore mask is only used during validation
        if self.validate:
            ignore_masks = erase_boundary_np(ignore_masks, edge_width)
            ignore_masks = ignore_masks.astype("float32")

            frames = maximum_center_crop(frames, self.opt.valid_crop)
            cr_masks = maximum_center_crop(cr_masks, self.opt.valid_crop)
            ignore_masks = \
                maximum_center_crop(ignore_masks, self.opt.valid_crop)

        else:
            # only crop training data
            cr_masks, crop_indices = stack_identical_random_crop(
                cr_masks, self.opt.crop, None, rotation, flip
            )

            frames, _ = stack_identical_random_crop(
                frames, self.opt.crop, crop_indices, rotation, flip
            )

            # placeholder, not used
            ignore_masks = cr_masks

        return frames, cr_masks, ignore_masks, filename


class MixedTrainSet(Dataset):
    def __init__(self, opt, LCO_dataset, HST_dataset):
        self.opt = opt
        self.batch_size = opt.batch
        self.batch_counter = 0
        self.die = 0

        self.lco_set = LCO_dataset
        self.hst_set = HST_dataset

        self.lco_len = len(LCO_dataset)
        self.hst_len = len(HST_dataset)

        print(
            f"creating fused dataset, {len(LCO_dataset)} fits from LCO and {len(HST_dataset)} images from HST"
        )

    def __len__(self):
        return self.lco_len

    def __getitem__(self, index):

        if self.batch_counter == 0:
            self.die = random.random()

        elif self.batch_counter == self.batch_size:
            self.batch_counter = 0

        self.batch_counter += 1

        if self.die < 0.5:
            ret = self.lco_set[index]
        else:
            ret = self.hst_set[index]

        return ret


class MixedValidateSet(Dataset):
    def __init__(self, opt, LCO_dataset, HST_dataset):
        self.opt = opt

        self.lco_data = LCO_dataset
        self.hst_data = HST_dataset

        self.lco_len = len(LCO_dataset)
        self.hst_len = len(HST_dataset)

        print(
            f"creating fused validation dataset, {len(LCO_dataset)} fits from LCO and {len(HST_dataset)} images from HST"
        )

    def __len__(self):
        return self.lco_len + self.hst_len

    def __getitem__(self, index):
        # go through both data sets

        if index < self.lco_len:
            return self.lco_data[index]
        else:
            off_set_idx = index - self.lco_len
            return self.hst_data[off_set_idx]
