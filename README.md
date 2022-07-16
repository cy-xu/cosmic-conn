# Cosmic-CoNN: A Cosmic Ray Detection Deep Learning Framework, Dataset, and Toolkit

[![arXiv](https://img.shields.io/badge/arXiv-2106.14922-b31b1b.svg?style=flat)](https://arxiv.org/abs/2106.14922) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5034763.svg)](https://doi.org/10.5281/zenodo.5034763) [![PyPI version](https://badge.fury.io/py/cosmic-conn.svg)](https://badge.fury.io/py/cosmic-conn) [![readthedocs](https://readthedocs.org/projects/cosmic-conn/badge/?version=latest)](https://cosmic-conn.readthedocs.io) [![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/) [![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg?style=flat-square)](https://tldrlegal.com/license/gnu-lesser-general-public-license-v3-(lgpl-3))

## [Documentation](https://cosmic-conn.readthedocs.io/) • [PyPI Release](https://pypi.org/project/cosmic-conn/) • [LCO CR Dataset](https://zenodo.org/record/5034763) • [Publications](https://github.com/cy-xu/cosmic-conn#publications)

### [[New] Demo video for interactive CR mask visualization and editing](https://www.youtube.com/watch?v=bdqmwcQeKyc&ab_channel=CYXu)

## About 
![Cosmic-CoNN overview](https://cosmic-conn.readthedocs.io/en/latest/_images/Cosmic-CoNN_overview.png)

Cosmic-CoNN is an end-to-end solution to help tackle the cosmic ray (CR) detection problem in CCD-captured astronomical images. It includes a deep-learning framework, high-performance CR detection models, a new dataset, and a suite of tools to use to the models, shown in the figure above:

1. [LCO CR dataset](https://zenodo.org/record/5034763), a large, diverse cosmic ray dataset that consists of over 4,500 scientific images from [Las Cumbres Observatory](https://lco.global/) (LCO) global telescope network's 23 instruments. CRs are labeled accurately and consistently across many diverse observations from various instruments. To the best of our knowledge, this is the largest dataset of its kind. 

2. A PyTorch deep-learning framework that trains generic, robust CR detection models for ground- and space-based imaging data, as well as spectroscopic observations.

3. A suite of tools including console commands, a web app, and Python APIs to make deep-learning models easily accessible to astronomers.

![Detection demo on Gemini data](https://cosmic-conn.readthedocs.io/en/latest/_images/fig11_gemini_results_demo.png)
Visual inspection of Cosmic-CoNNCR detection results. Detecting CRs in a Gemini GMOS-N 1×1 binning image with our generic ``ground-imaging`` model. The model was trained entirely on LCO data yet all visible CRs in the image stamp are correctly detected regardless of their shapes or sizes.

![Detection demo on LCO NRES data](https://cosmic-conn.readthedocs.io/en/latest/_images/fig11_nres_result_0034_1.png)
The Cosmic-CoNN ``NRES`` model detects CRs over the spectrum robustly on a LCO NRES spectroscopic image. The horizontal bands in the left image are the spectroscopic orders, which are left out of the CR mask.


## Installation

*We recently added optional dependencies install for pip.*

We recommend installing Cosmic-CoNN in a new virtual environment, see the step-by-step [installation guide](https://cosmic-conn.readthedocs.io/en/latest/source/installation.html). To get a ~10x speed-up with GPU acceleration, see [Install for a CUDA-enabled GPU](https://cosmic-conn.readthedocs.io/en/latest/source/installation.html).

```bash
  # basic install for CR detection or library integration
  $ pip install cosmic-conn

  # include Flask to use the interactive tool
  $ pip install "cosmic-conn[webapp]"

  # install all dependencies for development
  $ pip install "cosmic-conn[develop]"
```

## Command-line interface

After installation, you can batch process FITS files for CR detection from the terminal:

```bash
  $ cosmic-conn -m ground_imaging -e SCI -i input_dir
```

``-m`` or ``--model`` specifies the CR detection model. `"ground_imaging"` is loaded by default,  `"NRES"` is the spectroscopic model for LCO NRES instruments. You can also download a Hubble Space Telescope model trained by [deepCR](https://github.com/profjsb/deepCR) and pass in the model's path.

``-i`` or ``--input`` specifies the input file or directory. 

``-e`` or ``--ext`` defines which FITS extension to read image data, by default we read the first valid image array in the order of `hdul[0] -> hdul[1] -> hdul['SCI']` unless user specify an extension name.


See [documentation](https://cosmic-conn.readthedocs.io/en/latest/source/user_guide.html) for the complete user guide.

## Python APIs

It is also easy to integrate Cosmic-CoNN CR detection into your data workflow. Let `image` be a two-dimensional `float32 numpy` array of any size:

```Python

  from cosmic_conn import init_model

  # initialize a Cosmic-CoNN model
  cr_model = init_model("ground_imaging")

  # the model outputs a CR probability map in np.float32
  cr_prob = cr_model.detect_cr(image)

  # convert the probability map to a boolean mask with a 0.5 threshold
  cr_mask = cr_prob > 0.5

```

## Interactive CR mask visualization and editing

```bash
  $ cosmic-conn -am ground_imaging -e SCI
```

The Cosmic-CoNN web app automatically finds large CRs for close inspection. It supports live CR mask visualization and editing and is especially useful to find the suitable thresholds for different types of observations. We are working on addding the paintbrush tool for pixel-level manual editing.

<!-- <img src="https://cosmic-conn.readthedocs.io/en/latest/_images/cosmic_conn_web_app_interface.png" alt="web-based CR detector interface" width="600"/> -->

<a href="https://www.youtube.com/watch?v=bdqmwcQeKyc
" target="_blank"><img src="https://cosmic-conn.readthedocs.io/en/latest/_images/cosmic_conn_web_app_interface.png" 
alt="web-based CR detector interface" width="400" /></a>

The Cosmic-CoNN web app interface.

## Train new models with Cosmic-CoNN

See [documentation](https://cosmic-conn.readthedocs.io/en/latest/source/lco_cr_dataset.html) for the developer guide on using LCO CR dataset, data reduction, and model training.

## Publications

<p>
<!-- <a href="https://arxiv.org/abs/2106.14922"><img style="float: left; padding-right:30px;" src="https://cosmic-conn.readthedocs.io/en/latest/_images/paper_with_shadow.png"  width="220"/></a> -->

This repository is part of our Cosmic-CoNN research paper. Our methods and a thorough evaluation of models' performance are available in the paper. If you used the Cosmic-CoNN or the LCO CR dataset for your research, please cite our paper: [arXiv:2106.14922](https://arxiv.org/abs/2106.14922), [NASA ADS](https://ui.adsabs.harvard.edu/abs/2021arXiv210614922X/abstract)

```
@article{xu2021cosmic,
  title={Cosmic-CoNN: A Cosmic Ray Detection Deep-Learning Framework, Dataset, and Toolkit},
  author={Xu, Chengyuan and McCully, Curtis and Dong, Boning and Howell, D Andrew and Sen, Pradeep},
  journal={arXiv preprint arXiv:2106.14922},
  year={2021}
}
```

Please also cite the [LCO CR dataset](http://doi.org/10.5281/zenodo.5034763) if you used the Cosmic-CoNN `ground_imaging` model or the data in your research:
```
@dataset{xu_chengyuan_2021_5034763,
  author       = {Xu, Chengyuan and
                  McCully, Curtis and
                  Dong, Boning and
                  Howell, D. Andrew and
                  Sen, Pradeep},
  title        = {Cosmic-CoNN LCO CR Dataset},
  month        = jun,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.5034763},
  url          = {https://doi.org/10.5281/zenodo.5034763}
}
```

**Interactive Segmentation and Visualization for Tiny Objects in Multi-megapixel Images**  
[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_Interactive_Segmentation_and_Visualization_for_Tiny_Objects_in_Multi-Megapixel_Images_CVPR_2022_paper.html)
```
@InProceedings{Xu_2022_CVPR,
    author    = {Xu, Chengyuan and Dong, Boning and Stier, Noah and McCully, Curtis and Howell, D. Andrew and Sen, Pradeep and H\"ollerer, Tobias},
    title     = {Interactive Segmentation and Visualization for Tiny Objects in Multi-Megapixel Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21447-21452}
}
```

![interactive_segmentation_cvpr22_poster_v2](https://user-images.githubusercontent.com/24612082/174725216-8df9b89b-d5b2-483d-8cf7-d7c660302aeb.png)
  
</p>


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.
