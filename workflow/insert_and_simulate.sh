#!/bin/sh
# ***************************** EDIT CONSTANT VALUES BELOW ACCORDINGLY TO MEET YOUR INSERTION AND SIMULATION NEEDS ****************************
DATABASE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db"
LOGGING_CONFIG_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/doe_xstock/misc/logging_config.json"
INSERT_DATASET_TYPE="resstock"
INSERT_WEATHER_DATA="tmy3"
INSERT_YEAR_OF_PUBLICATION="2021"
INSERT_RELEASE="1"
INSERT_FILTERS_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/workflow/insert_filters.json"
SIMULATE_ENERGYPLUS_PARAMETERS_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/workflow/simulate.csv"
IDD_FILEPATH="/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd"
SIMULATE_ENERGYPLUS_ROOT_OUTPUT_DIRECTORY="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/energyplus_simulation"
# ******************************************************************** END ********************************************************************

# DOWNLOAD & INSERT DATASET
python -m doe_xstock -d $DATABASE_FILEPATH dataset $INSERT_DATASET_TYPE $INSERT_WEATHER_DATA $INSERT_YEAR_OF_PUBLICATION $INSERT_RELEASE insert -f $INSERT_FILTERS_FILEPATH || exit 1

# ENERGYPLUS SIMULATION
OLDIFS=$IFS
IFS=','
{
    read
    while read -r dataset_type weather_data year_of_publication release bldg_id upgrade
    do
        python -m doe_xstock -d $DATABASE_FILEPATH dataset $dataset_type $weather_data $year_of_publication $release simulate $IDD_FILEPATH $bldg_id -u $upgrade -o $SIMULATE_ENERGYPLUS_ROOT_OUTPUT_DIRECTORY || exit 1
    done
} < $SIMULATE_ENERGYPLUS_PARAMETERS_FILEPATH
IFS=$OLDIFS