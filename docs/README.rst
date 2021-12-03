============
Cosmic-CoNN
============

A Cosmic Ray Detection Deep Learning Framework, Dataset, and Toolkit

.. image:: /_static/Cosmic-CoNN_overview.png
        :alt: Cosmic-CoNN overview

Cosmic-CoNN is an end-to-end solution to help tackle the cosmic ray (CR) detection problem in CCD-captured astronomical images. It includes a deep-learning framework, high-performance CR detection models, a new dataset, and a suite of tools to use to the models, shown in the figure above:

1. `LCO CR dataset <https://zenodo.org/record/5034763>`_, a large, diverse cosmic ray dataset that consists of over 4,500 scientific images from `Las Cumbres Observatory <https://lco.global/>`_ (LCO) global telescope network's 23 instruments. CRs are labeled accurately and consistently across many diverse observations from various instruments. To the best of our knowledge, this is the largest dataset of its kind. 

2. A `PyTorch <https://pytorch.org/>`_ deep-learning framework that trains generic, robust CR detection models for ground- and space-based imaging data, as well as spectroscopic observations.

3. A suite of tools includings console commands, a web app, and Python APIs to make deep-learning models easily accessible to astronomers.

.. figure:: /_static/fig11_gemini_results_demo.png
        :alt: Detection demo on Gemini data

        Visual inspection of Cosmic-CoNN CR detection results. Detecting CRs in a Gemini GMOS-N 1×1 binning image with our generic ``ground-imaging`` model. The model was trained entirely on LCO data yet all visible CRs in the image stamp are correctly detected regardless of their shapes or sizes.

.. figure:: /_static/fig11_nres_result_0034_1.png
        :alt: Detection demo on LCO NRES data

        The Cosmic-CoNN ``NRES`` model detects CRs over the spectrum robustly on a LCO NRES spectroscopic image. The horizontal bands in the left image are the spectroscopic orders, which are left out of the CR mask.

Command line interface
======================

After :ref:`install_label`, you can start detecting CRs in your FITS files right from the command line::

  $ cosmic-conn -m ground_imaging -e SCI -i input_dir

This command launches a generic ``gorund_imaging`` model to detect cosmic rays. It reads data from the SCI extension in a FITS file and processes all files in the input_dir. We also provide the ``NRES`` model for CR detection in spectroscopic data and the ``HST_ACS_WFC`` model for Hubble ACS/WFC imaging data. You could also find more Hubble Space Telescope CR detection and inpainting models trained by `deepCR <https://github.com/profjsb/deepCR>`_.

Python APIs
===========

It is also easy to integrate Cosmic-CoNN CR detection into your data workflow. Let ``image`` be a two-dimensional ``float32 numpy`` array of any size:

.. code-block:: python

  from cosmic_conn import init_model

  # initialize a Cosmic-CoNN model
  cr_model = init_model("ground_imaging")

  # the model outputs a CR probability map in np.float32
  cr_prob = cr_model.detect_cr(image)

  # convert the probability map to a boolean mask with a 0.5 threshold
  cr_mask = cr_prob > 0.5

Web app
=======

The Cosmic-CoNN web app automatically finds large CRs for close inspection. It supports live CR mask editing and is especially useful to find the suitable threshold for different types of observations:

.. figure:: /_static/cosmic_conn_web_app_interface.png
  :alt: an image shows the web-based CR detector interface

  The Cosmic-CoNN web app interface.

Publication
===========

.. image:: /_static/paper_with_shadow.png
        :width: 200
        :target: https://arxiv.org/abs/2106.14922

This software is part of our Cosmic-CoNN research paper. Our methods and a thorough evaluation of models' performance are available in the paper. If you used the Cosmic-CoNN or the LCO CR dataset for your research, please cite our paper:

`deepCRarXiv:2106.14922 <https://arxiv.org/abs/2106.14922>`_, `NASA ADS <https://ui.adsabs.harvard.edu/abs/2021arXiv210614922X/abstract>`_

Please also cite the LCO CR dataset if you used the Cosmic-CoNN ``ground_imaging`` model or the data in your research:

Xu, Chengyuan, McCully, Curtis, Dong, Boning, Howell, D. Andrew, & Sen, Pradeep. (2021). Cosmic-CoNN LCO CR Dataset (Version 0.1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.5034763


Mics.
=====

.. image:: https://img.shields.io/pypi/v/cosmic-conn.svg
        :target: https://pypi.python.org/pypi/cosmic-conn

.. image:: https://readthedocs.org/projects/cosmic-conn/badge/?version=latest
        :target: https://cosmic-conn.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
        :: target: http://www.astropy.org/
        :alt: astropy

* Free software: GNU General Public License v3
* Documentation: https://cosmic-conn.readthedocs.io.


Credits
=======

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
