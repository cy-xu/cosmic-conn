.. _user-guide-label:

===========
User Guide
===========

.. note:: Please refer to the :ref:`install_label` guide to configure Cosmic-CoNN.

Batch detection with console commands
=====================================

By default, the console commands will load the generic "ground_imaging" model to perform CR detection on all FITS files in the working directory ``./`` and subdirectories. It reads the first valid image array in the ``*.fits`` or ``*.fz`` files::

  $ cosmic-conn
  # or equivalently (underscore)
  $ cosmic_conn

**Specifying the input directory**
  
Batch processing multiple FITS files by specifying the input directory with ``-i`` or ``--input``::

  $ cosmic-conn -i input_dir
  # to process a single file::
  $ cosmic-conn -i input_dir/target_file.fits.fz

**Specifying FITS extension for data**
  
Use ``-e`` or ``--ext`` to define which extension to read data from, by default we read the first valid image array in the order of hdul[0] -> hdul[1] -> hdul['SCI'] unless user provided a extension name to override::

  $ cosmic-conn -i input_dir -e SPECTRUM

**Specifying the detection model**

You could specify which model to use with ``-m`` or ``--model``::

  $ cosmic-conn -i input_dir -m ground_imaging       # the ground-imaging model (default)
  $ cosmic-conn -i input_dir -m NRES                 # the spectroscopic model for LCO NRES
  $ cosmic-conn -i input_dir -m HST_ACS_WFC          # the HST ACS/WFC model

Web-based app
====================

The ``-a`` or ``--app`` arguments will launch a *local instance* of the web-based CR detector app, which supports CR mask preview and editing. Access the interface from http://127.0.0.1:5000/.

.. code-block:: console

  $ cosmic-conn -a

The generic ``ground_imaging`` model is loaded by default, here is a shorthand to launch the web app with the NRES model and read image from the SPECTRUM extension:

.. code-block:: console

  $ cosmic-conn -am NRES -e SPECTRUM

.. figure:: /_static/cosmic_conn_web_app_interface.png
  :alt: an image shows the web-based CR detector interface

  The Cosmic-CoNN web app interface.

The preview windows help you to verify the results immediately after detection. We provided common scaling methods for visualzation, ``zscale`` is applied by default. You could also manually define the MIN-MAX range to disply, and their mapping to the ``UINT8`` image. The pointer location shows the true pixel value at the bottom-left corner.

The editing tools on top of the mask preview windows help you to fine-tune the threshold and morphological dilation applied to the probability mask to acquire a binary mask that suits your data. A new copy of the FITS with the masks appended is saved in ``cosmic_conn_output`` of the working directory. The ``Download`` button will append the edited binary mask to the FITS.

Above the preview windows is a row of CR thumbnails sorted by CR size, so you could quickly navigate to the largest CR found in the image. 

.. note:: The web-based app launches a localhost Python HTTP server, your observation data is never uploaded to the internet.

..
  _note:: Google Analytics help us understand how many people have used this tool. We only receive the *Page views* info and we do not track any user behavior. It is also easy to turn off. You can launch the app with Google Analytics **turned off** by::  $ cr_app --no-tracking


Import as Python package
========================


Adopting CR detection in your data workflow is simple. Let ``image`` be a two-dimensional ``float32 numpy`` array of any size:

.. code-block:: python

  from cosmic_conn import init_model

  # initialize a Cosmic-CoNN model
  cr_model = init_model("ground_imaging")

  # the model outputs a CR probability map in np.float32
  cr_prob = cr_model.detect_cr(image)

  # convert the probability map to a boolean mask with a 0.5 threshold
  cr_mask = cr_prob > 0.5

The returned array ``cr_prob`` is the predicted probability of each pixel being affected by CR, where :math:`\text{cr_prob}_{ij} \in [0, 1]`. A threshold of 0.5 is suitable for most data but using the interactive preview in the `Web-based app`_ could help find the suitable parameters based on your data. 

Lowering the threshold will include more peripheral CR pixels and applying morphological dilation will enlarge mask areas for the detected CRs. To dilate the mask by one pixel:

.. code-block:: python

  from skimage.morphology import dilation, square
  
  cr_mask = dilation(cr_mask, square(3))

