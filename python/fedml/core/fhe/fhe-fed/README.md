# FHE-Based FedAvg
Secure FedAvg Functionality using Fully Homomorphic Encryption (CKKS)
### Dependencies (tested in Ubuntu)
Using Docker env with Dockerfile or install all dependencies yourself as bellow:

- `PALISADE`: a lattice-based homomorphic encryption library in C++. Follow the instructions on https://gitlab.com/palisade/palisade-release to download, compile, and install the library. Make sure to run `make install` in the user-created `\build` directory for a complete installation. 

- `pybind-11`: pip install pybind11, make sure to have have `python3` and `cmake` already installed. 

- `Clang`: install clang and set it as the default compiler

`palisade_pybind` folder contains the implementation of weighted average operation with python bindings.

#### To install After Self-Installing Dependencies

go to the `palisade_pybind/SHELFI_FHE/src` folder and run `pip install ../`

test the fhe function by running `python3 ../pythonApi/ckks_example.py`

