# DOE XStock: BS2023
## Description
This repository tag is used to reproduce the `A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn` Building and Simulation 2023 conference paper.

## Installation
To install, clone the repository and checkout the `bs2023` branch:
```console
git clone https://github.com/intelligent-environments-lab/doe_xstock.git
git checkout bs2023
```

## Dependencies
The project's Python 3rd party library dependencies are listed in [requirements.txt](https://github.com/intelligent-environments-lab/DOE_XStock/blob/master/requirements.txt) and can be installed via `pip`:
```console
pip install requirements.txt
```

Download [EnergyPlus 9.6.0](https://github.com/NREL/EnergyPlus/releases/tag/v9.6.0) to be able to run building energy model simulations.

The workflow assumes a Linux or Mac based machine is in use.

## Reproduction Workflow

### Neighborhoods
To reproduce the neighborhoods used in the `A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn` Building and Simulation 2023 conference paper, first download the relevant building data for the three neighborhoods (CA, Alameda County; TX, Travis County; and VT, Chittendem County)from the [End-Use Load Profiles (EULP) for the U.S. Building Stock database](https://www.nrel.gov/buildings/end-use-load-profiles.html):
```console
sh workflow/download.sh
```

Insert the ecobee setpoint profiles in the database by executing the [ecobee.ipynb](analysis/ecobee.ipynb) notebook from start to end. The inserted setpoint profiles are the representative profiles after clustering as described in `Figure 2b` in the paper.

The next step is to discover the representative building clusters (`Figure 2a` in paper):
```console
sh workflow/metadata_clustering.sh
```

The following step runs the EnergyPlus simulations for the selected representative neighborhood buildings:
```console
sh workflow/set_lstm_train_data.sh
```

Finally, execute the [post_simulation.ipynb](analysis/post_simulation.ipynb) notebook from start to end to set the CityLearn input data.

### Control Simulation
To reproduce the CityLearn control simulation, navigate to the `citylearn_simulation` path:
```console
cd citylearn_simulation
```

Run the `preprocess.sh` workflow to prepare the CityLearn schemas and simulation work orders for the three neighborhoods:
```console
sh workflow/preprocess.sh
```

Next, run the generated work orders:
```console
sh simulate.sh
```

When the CityLearn simulations complete, run the [post_simulation.ipynb](citylearn_simulation/analysis/analysis.ipynb) notebook from start to end to generate the control simulation results that are reported in the paper.