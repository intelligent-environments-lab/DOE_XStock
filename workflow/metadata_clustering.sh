#!/bin/sh
# ***************************** EDIT ABSOLUTE FILEPATHS TO MATCH SYSTEM PATHS ****************************
DATABASE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db"
INSERT_FILTERS_FILEPATH="data/filters.csv"
DATA_DIRECTORY="data"
FIGURE_DIRECTORY="figures"
INSERT_DATASET_TYPE="resstock"
INSERT_WEATHER_DATA="amy2018"
INSERT_YEAR_OF_PUBLICATION="2021"
INSERT_RELEASE="1"
MAXIMUM_N_CLUSTERS="10"
SAMPLE_COUNT="100"
# ***************************************************** END **********************************************

# DOWNLOAD & INSERT DATASET
OLDIFS=$IFS
IFS='|'
{
    read
    while read -r neighborhood_id filters
    do
        python -m doe_xstock -d $DATABASE_FILEPATH -t $DATA_DIRECTORY -g $FIGURE_DIRECTORY dataset $INSERT_DATASET_TYPE $INSERT_WEATHER_DATA $INSERT_YEAR_OF_PUBLICATION $INSERT_RELEASE -f $filters metadata_clustering $neighborhood_id $MAXIMUM_N_CLUSTERS -c $SAMPLE_COUNT || exit 1
    done
} < $INSERT_FILTERS_FILEPATH
IFS=$OLDIFS
