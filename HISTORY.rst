=======
History
=======

0.4.0 (2022-04-18)
    - added pencil tool for pixel-level manual editing on the CR/segmentation mask
    - update zenodo archive

------------------

0.3.0 (2022-04-02)
    - added scripts to reproduce ablation study experiments
    - zenodo archive

------------------

0.2.8 (2022-01-04)
    - added `-c` option to CLI to specify crop size for stamp detection
    - stop using memory_check(), which is not robust on server nodes
    - moved messages to logger, stdout turned on only for CLI users
    - removed trained models' DataParallel wrapper
    - new threshold-based plots for BANZAI integration

------------------

0.2.7 (2021-12-03)
    - Trained models added to git repository

------------------

0.2.4 (2021-11-30)
    - Added pip optional dependencies install support.
    - pip install cosmic-conn # basic install for CR detection
    - pip install cosmic-conn[webapp] # include Flask for the web app 
    - pip install cosmic-conn[develop] # all dependencies for development

------------------

0.2.3 (2021-07-19)
    - fixed the disappeared scroll bar in Mac's Chrome

------------------

0.2.0 (2021-06-27)
    - Initial release verison

------------------

0.1.9 (2021-05-11)
    - Web app fully functional

------------------

0.1.0 (2021-04-19)
    - Test release on PyPI.