.. _cr_dataset_label:

==============
LCO CR Dataset
==============

Download
========

The LCO CR dataset is hosted on `Zenodo <https://zenodo.org/record/5034763>`_. 

.. _data_structure_label:

Data Structure
==============

The expanded LCO CR data set directory has the following structure:

.. code-block:: text

    LCO_CR_dataset/
    ├-train_set/
    │ |
    │ └-0m4_2019/
    │ │ ├-masked_fits/
    │ │ │ ├-coj0m403-kb24-20190101-0147-e91_3frms_masks.fits
    │ │ │ ├-coj0m403-kb24-20190101-0150-e91_3frms_masks.fits 
    │ │ │ └─...
    │ │ └─CR_reduction_log.txt
    │ |
    │ ├-1m0_2019_2018/
    │ └─2m0_2019/
    │
    ├-test_set/
    │ ├-aligned_fits/
    │ ├-masked_fits/
    │ └─CR_reduction_log.txt


The ``*-3frms_masks.fits`` has the following structure. The Ver 0,1,2 indicates the three exposures of a consecutive sequence. The ``*-3frms_aligned.fits`` includes only the 0-5 FITS extensions:

.. table::

    ===  ==========  ===  ============  =====  ============  =======
    No.  Name        Ver  Type          Cards  Dimensions    Format
    ===  ==========  ===  ============  =====  ============  =======
    0    PRIMARY     1    PrimaryHDU    6      
    1    SCI         0    CompImageHDU  266    (3054, 2042)  float32
    2    SCI         1    CompImageHDU  266    (3054, 2042)  float32
    3    SCI         2    CompImageHDU  266    (3054, 2042)  float32
    4    VALID_MASK  1    CompImageHDU  8      (3054, 2042)  uint8
    5    CAT         1    BinTableHDU   157    
    6    CR          0    CompImageHDU  266    (3054, 2042)  uint8
    7    CR          0    CompImageHDU  266    (3054, 2042)  uint8
    8    CR          0    CompImageHDU  266    (3054, 2042)  uint8
    9    IGNORE      0    CompImageHDU  266    (3054, 2042)  uint8
    10   IGNORE      0    CompImageHDU  266    (3054, 2042)  uint8
    11   IGNORE      0    CompImageHDU  266    (3054, 2042)  uint8
    ===  ==========  ===  ============  =====  ============  =======  

The ignore mask is a coded ``unit8`` array with the following definition:

- OK (0)
- boundary (1)
- no data (2)
- sources (4)
- hot pixels (8)

.. note:: 

    If you used the Cosmic-CoNN ground-imaging model or the LCO CR dataset for your research, pleaes cite the datset as well:

    Xu, Chengyuan, McCully, Curtis, Dong, Boning, Howell, D. Andrew, & Sen, Pradeep. (2021). Cosmic-CoNN LCO CR Dataset (Version 0.1.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.5034763
