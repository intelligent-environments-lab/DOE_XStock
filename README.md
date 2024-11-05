# DOE XStock
## Description
This repository is used to manage the [End Use Load Profiles for the U.S. Building Stock](https://www.nrel.gov/buildings/end-use-load-profiles.html) dataset by providing an _Pythonic_ interface to download the [dataset files](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F) as well as run EnergyPlus simulations on the contained OSM models. Refer to the [README.md](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F) for more details on the full dataset.

## Installation
To install, clone the latest version of the repository from the project's homepage or install the Python package using `pip`. To clone, execute:
```console
git clone https://github.com/intelligent-environments-lab/doe_xstock.git
```

To install the Python package, execute:
```console
pip install git+https://github.com/intelligent-environments-lab/DOE_XStock.git@v1-develop
```

## Dependencies
The project's Python 3rd party library dependencies are listed in [requirements.txt](https://github.com/intelligent-environments-lab/DOE_XStock/blob/v1-develop/requirements.txt). These dependencies are automatically fulfilled if installed with `pip`.

Download [EnergyPlus 9.6.0](https://energyplus.net/downloads) to be able to run EnergyPlus simulations.

# Documentation
Coming soon :).