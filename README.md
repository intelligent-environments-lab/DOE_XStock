# DOE XStock
## Description
This repository is used to manage DOE's [End Use Load Profiles for the U.S. Building Stock](https://www.nrel.gov/buildings/end-use-load-profiles.html) data by providing an Pythonic interface to download and store the [full dataset](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F) in a local SQLite database as well as run EnergyPlus simulations on the contained OSM models. Refer to the [End Use Load Profiles for the U.S. Building Stock README.md](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F) for more details on the full dataset.

## Installation
To install, clone the latest version of the repository from the project's homepage or install the Python package using `pip`. To clone, execute:
```console
git clone https://github.com/intelligent-environments-lab/doe_xstock.git
```

To install the Python package, execute:
```console
python -m pip install git+https://github.com/intelligent-environments-lab/doe_xstock.git
```

## Dependencies
The project's Python 3rd party library dependencies are listed in [requirements.txt](https://github.com/https://github.com/intelligent-environments-lab/doe_xstock/blob/master/requirements.txt). The dependencies are automatically fulfilled if installed with `pip`.

Download [EnergyPlus 9.6.0](https://energyplus.net/downloads) to be able to run building energy model simulations.

## General Usage
The expected typical workflow is to download relevant data for buildings of interest. Additionally, there might be a need to simulate downloaded energy models. The full dataset is defined by a combination of field values where the fields include `dataset type`, `weather data`, `year of publication`, and `release number` (See [README.md](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F) for the field definition). For each combination, there is a corresponding `metadata.parquet` file that describes all contained buildings which, can be used to inform building data download selection. The provided library can also be used to read the metadata into memory. An example is given:

```python
from doe_xstock.doe_xstock import DOEXStockDatabase

metadata = DOEXStockDatabase.download_summary_data(
    summary_type = DOEXStockDatabase.SummaryType.METADATA,
    dataset_type = 'resstock',
    weather_data = 'tmy3',
    year_of_publication = 2021,
    release = 1
)
```

Other downloadable summary types that can be used to inform building selection include;
- `DOEXStockDatabase.SummaryType.DATA_DICTIONARY`;
- `DOEXStockDatabase.SummaryType.ENUMERATION_DICTIONARY`;
- `DOEXStockDatabase.SummaryType.UPGRADE_DICTIONARY`; and
- `DOEXStockDatabase.SummaryType.SPATIAL_TRACT`.

After it has been decided what relevant building data to insert into the database, use the `DOEXStockDatabase` class to perform the insertion. An example is given:

```python
from doe_xstock.doe_xstock import DOEXStockDatabase

database_filepath = 'doe_xstock.db'
database = DOEXStockDatabase(database_filepath)
database.insert_dataset(
    dataset_type = 'resstock',
    weather_data = 'tmy3',
    year_of_publication = 2021,
    release = 1,
    filters = {
        'in.geometry_building_type_recs': ['Single-Family Detached'],
        'in.vacancy_status': ['Occupied'],
        'in.resstock_county_id': ['TX, Travis County'],
        'bldg_id': [40]
    }
)
```

The example above will insert data into a number of tables that are relevant to the combination of dataset fields and the `filter` argument. The `filter` argument is a dictionary of metadata columns as keys and list of interested enumerations as values. The database schema can be retrieved by executing the following:

```python
from doe_xstock.doe_xstock import DOEXStockDatabase

database_filepath = 'doe_xstock.db'
database = DOEXStockDatabase(database_filepath)
schema = database.get_schema()
```

Inserted data can then be queried using SQL and is returned as a `pd.DataFrame` object. An example is given using the `metadata` table:

```python
from doe_xstock.doe_xstock import DOEXStockDatabase

database_filepath = 'doe_xstock.db'
database = DOEXStockDatabase(database_filepath)
metadata = database.query_table("""
    SELECT
        *
    FROM metadata
    WHERE in_geometry_building_type_recs = 'Single-Family Detached'
""")
```

Finally, EnergyPlus simulation can be run using downloaded building data. An example workflow is given below:

```python
from doe_xstock.doe_xstock import DOEXStockDatabase
from doe_xstock.simulate import OpenStudioModelEditor, Simulator

database_filepath = 'doe_xstock.db'
schedule_filename = 'schedules.csv'
idd_filepath = '/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd'

# get simulation data
database = DOEXStockDatabase(database_filepath)
metadata_id, osm, epw = database.query_table(f"""
    SELECT 
        i.metadata_id,
        i.bldg_osm AS osm, 
        i.bldg_epw AS epw,
        m.in_pv_system_size AS pv_system_size,
        m.in_ashrae_iecc_climate_zone_2004 AS climate_zone
    FROM building_energy_performance_simulation_input i
    LEFT JOIN metadata m ON m.id = i.metadata_id
    WHERE 
        i.dataset_type = 'resstock'
        AND i.dataset_weather_data = 'tmy3'
        AND i.dataset_year_of_publication = 2021
        AND i.dataset_release = 1
        AND i.bldg_id = 40
        AND i.bldg_upgrade = 0
""").to_records(index = False)[0]

# get schedule file referenced internally by energy model during simulation
schedule = database.query_table(f"""
    SELECT 
        * 
    FROM schedule 
    WHERE metadata_id = metadata_id
""")

# remove columns that displace the energy model's column indexing 
schedule = schedule.drop(columns=['metadata_id', 'timestep'])

# simulate
try:
    osm_editor = OpenStudioModelEditor(osm)
    schedule.to_csv(schedules_filename, index = False)

    # convert OSM to IDF
    idf = osm_editor.forward_translate()

    # run EnergyPlus simulation
    simulator = Simulator(idd_filepath, idf, epw)
    simulator.simulate()

except Exception as e:
    raise e

finally:
    os.remove(schedules_filename)
```

## CLI
For high level and less verbose usage that accomplishes most of the functionalities in [General Usage](#general-usage), refer to the CLI help message for instructions:
```console
python -m doe_xstock -h
```

### Workflow
To insert and simulate multiple files uisng the CLI, edit [insert_filters.json](https://github.com/https://github.com/intelligent-environments-lab/doe_xstock/blob/master/workflow/insert_filters.json), and [simulate.csv](https://github.com/https://github.com/intelligent-environments-lab/doe_xstock/blob/master/workflow/simulate.csv) to specify metadata filters for database insertion and buildings for simulation respectively. Also, edit the constant values in [](https://github.com/https://github.com/intelligent-environments-lab/doe_xstock/blob/master/workflow/insert_and_filter.sh) to meet insertion and simulation needs as well as point to relevant filepaths. Finally, execute:
```console
sh workflow/insert_and_simulate.sh
```

# Documentation
Coming soon :).