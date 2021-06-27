===============
Troubleshooting
===============

- Dependency issues

    You can find the complete environment dependencies in ``requirements.txt``. Be cautious only run this in a new virtual environment::

    $ pip install -r requirements.txt


- ``[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware``

    NNPACK is an acceleration package for neural network computations. It is a known issue that happens on limited hardware architectures, like Apple's ARM-based M1. The computation result is not affected except slower. Detail: https://github.com/Maratyszcza/NNPACK

    It is possible to `build PyTorch from the source <https://github.com/pytorch/pytorch#adjust-build-options-optional>`_ without NNPACK with::
        
        USE_NNPACK=0 python setup.py install



