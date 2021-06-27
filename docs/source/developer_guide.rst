===============
Developer Guide
===============

.. note:: Please refer to the :ref:`install_label` guide to configure Cosmic-CoNN.

LCO CR masks reduction
======================

``cosmic_conn/cr_pipeline`` holds the source code for the CR labeling pipeline. In the first of the two-phases reduction, the pipeline searches for individual LCO imaging frames that are consecutive exposures to perform reprojection using the `astropy/reproject <https://github.com/astropy/reproject>`_ package. Consecutive frames are merged into a single FITS file that includes three image extensions and a valid mask, see :ref:`data_structure_label`. Results from phase one are saved in ``input_path/aligned_fits``.

.. note:: We provide the reprojected data in the released LCO CR dataset so phase one could be skipped by passing the ``--aligned`` flag.

In phase two, the pipeline takes an aligned FITS file from phase one as input, appends the corresponding CR masks and outputs the masked FITS in the ``input_path/masked_fits``. ``cosmic_conn/reduce_cr.py`` is the entry file for CR reduction.An example reduction script:

.. code-block:: console

    $ bash scripts/reduce_lco_cr.sh
    # includes the following arguments:

    $ python cosmic_conn/reduce_cr.py \
    --data data/demo_data \         # path to data directory
    --snr_thres 5 \                 # threshold to detect CR with simga > 5
    --snr_thres_low 2.5 \           # lower threshold to include CR's peripheral pixels
    --dilation 5 \                  # dilation range for peripheral pixels
    --min_exptime 99. \             # reject short exposed frames
    --min_cr_size 2 \               # a minimum CR size of 2 ignores isolated hot pixels
    --cpus 8 \                      # cores used for multiprocessing acceleration
    --aligned \                     # ignore phase one with this flag
    --no_png \                      # do not output png preview with this flag
    --comment dilation5-SNR5-2.5    # png preview files suffix

The reduction configuration and log are saved in ``CR_reduction_log.txt``.


Train a new model with the Cosmic-CoNN framework
==================================================

``cosmic_conn/dl_framework`` holds the source code for the Cosmic-CoNN deep-learning framework. ``trian.py`` is the entry file to initiate a new training. We recommend start a training by modifying an example script, e.g. the script that trains a LCO imaging model can be found at ``scripts/train_lco.sh``:

.. code-block:: text

    $ python cosmic_conn/train.py
    # basic training settings:
        --data                path to data directory
        --mode                train | inference, inference mode does not create checkpoint or log
        --seed                assign manual seed
        --random_seed         it will initialize the model multiple times to find a
                                best random seed if flagged
        --max_train_size      the # of samples randomly draw in each epoch, 0 uses entire dataset
        --lr LR               learning rate, 0.001 by default
        --milestones          [MILESTONES ...] milesstones to reduce the learning rate, 
                                e.g. '1000 2000 3000', '0' keeps LR constant
        --min_exposure        minimum exposure time when sampling training data
        --crop                training input stamp size
        --batch               training batch size
        --comment             comment is appended to the checkpoint directory name

    # define the model
        --model               lco | hst | nres, dataset specific dataloader
        --loss                bce | median_bce | dice | mse, loss function used for training
        --imbalance_alpha     number of iterations for the Median Weighted BCE Loss
                                to linearly increase the lower bound alpha to 1. See
                                paper for detail.
        --norm                batch | group | instance, feature normalization method
        --n_group             fixed group number for group normalization
        --gn_channel          fixed channel number, >0 will override n_group, 0 uses
                                fixed group number
        --conv_type           unet | resnet, types for convolution module
        --up_type             deconv | upscale, types for deconvolution module
        --down_type           maxpool | avgpool | stride, types for the pooling layer
        --deeper              deeper network, one more downsample and upsample layer
        --hidden              channel # of first conv layer
        --epoch               total training epochs
        --eval_epoch          number of phase0 training epochs, only applies to batch normalization

    # settings for model validation during the training
        --validate_freq       number per epochs to perform model validation
        --validRatio          the ratio of training data reserved for validation, 0.2 by default
        --max_valid_size      the number of sample reserved for validation, >0 will
                                override validRatio
        --valid_crop          stamp size for the center-cropping during validation

    # to continue a previous training, use the following arguments
        --continue_train      to continue a previous training, provide the checkpoint directory name
        --continue_epoch      the number of epoch to continue

    # only called during inference
        --load_model          path to load a model for inference


