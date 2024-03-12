# A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn
This source code in the directory is used to reproduce the results in the `A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn` paper.

## Installation
First, clone this repository as:
```bash
git clone https://github.com/intelligent-environments-lab/DOE_XStock.git
```

Then navigate to this directory in the repository:
```bash
cd citylearn_simulation
```

Install the dependencies in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```

It is important that the specified `stable-baseline3` version or earlier is used to avoid issues with `stable-baseline3` stopping support for `gym` environments, which the `CityLearn` version used in this work is.

## Running simulations
Run the [workflow](workflow/preprocess.sh) to generate the simulation input dataset and workflows for running the simulations:
```bash
sh workflow/preprocess.sh
```

The current state of the repository already has the simulation input dataset in [citylearn_simulation/data/neighborhoods](citylearn_simulation/data/neighborhoods) where you will find three CityLearn schemas for the three different counties.

Finally, execute the [simulate.sh](workflow/simulate.sh) workflow that runs the CityLearn simulations for the three neighborhoods:
```bash
sh workflow/simulate.sh
```

## Results
The results of the simulation are stored in `data/simulation_output` that is automatically generated and has one subdirectory for each building that is simulated. This subdirectory has four output files:
1. `building_id-environment.csv`: Time series of static and runtime environment variables during training episodes and final evaluation.
2. `building_id-kpi.csv`: Energy flexibility KPIs calculated at the end of each episode during training episodes and final evaluation.
3. `building_id-reward.csv`: Time series of reward during training episodes and final evaluation.
4. `building_id-timer.csv`: Time it took for each episode to complete during training episodes and final evaluation.

The notebooks in the [analysis](analysis) directory reference these results to generate the figures and statistics reported in the paper.

## Citation
```bibtex
@inproceedings{bs2023_1404,
	doi = {https://doi.org/10.26868/25222708.2023.1404},
	url = {https://publications.ibpsa.org/conference/paper/?id=bs2023_1404},
	year = {2023},
	month = {September},
	publisher = {IBPSA},
	author = {Kingsley  Nweye  and   Kathryn  Kaspar  and   Giacomo  Buscemi  and   Giuseppe  Pinto  and   Han  Li  and   Tianzhen  Hong  and   Mohamed  Ouf  and   Alfonso  Capozzoli  and   Zoltan  Nagy},
	title  = {A framework for the design of representative neighborhoods for energy flexibility assessment in CityLearn},
	booktitle = {Proceedings of Building Simulation 2023: 18th Conference of IBPSA},
	volume  = {18},
	isbn = {},
	address  = {Shanghai, China},
	series  = {Building Simulation},
	pages = {3351--3358},
	issn = {2522-2708},
	Organisation = {IBPSA},
	Editors = {}
}
```