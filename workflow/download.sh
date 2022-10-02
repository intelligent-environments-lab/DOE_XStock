#!/bin/sh
# ***************************** EDIT ABSOLUTE FILEPATHS TO MATCH SYSTEM PATHS ****************************
DATABASE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db"
LOGGING_CONFIG_FILEPATH="doe_xstock/misc/logging_config.json"
INSERT_FILTERS_FILEPATH="data/filters.csv"
INSERT_DATASET_TYPE="resstock"
INSERT_WEATHER_DATA="amy2018"
INSERT_YEAR_OF_PUBLICATION="2021"
INSERT_RELEASE="1"
# ***************************************************** END ***********************************************

# DOWNLOAD & INSERT DATASET
OLDIFS=$IFS
IFS='|'
{
    read
    while read -r neighborhood_id filters
    do
        python -m doe_xstock -d $DATABASE_FILEPATH dataset $INSERT_DATASET_TYPE $INSERT_WEATHER_DATA $INSERT_YEAR_OF_PUBLICATION $INSERT_RELEASE -f $filters insert || exit 1
    done
} < $INSERT_FILTERS_FILEPATH
IFS=$OLDIFS