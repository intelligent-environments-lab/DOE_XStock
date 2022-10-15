#!/bin/sh

# SET WORKFLOW ENV
source workflow/workflow_env

# DOWNLOAD & INSERT DATASET
OLDIFS=$IFS
IFS='|'
{
    read
    while read -r neighborhood_id filters
    do
        neighborhood_id=${neighborhood_id// /_}
        neighborhood_id=${neighborhood_id//,/}
        neighborhood_id=$(echo $neighborhood_id | tr '[:upper:]' '[:lower:]')
        neighborhood_filepath=$DOE_XSTOCK_NEIGHBORHOOD_DIRECTORY$neighborhood_id"_neighborhood.csv"
        
        {
            read
            while read -r name metadata_id bldg_id label
            do
            python -m doe_xstock \
                -d $DOE_XSTOCK_DATABASE_FILEPATH \
                    -u $DOE_XSTOCK_ENERGYPLUS_OUTPUT_DIRECTORY \
                        -i $DOE_XSTOCK_IDD_FILEPATH \
                            dataset \
                                $DOE_XSTOCK_INSERT_DATASET_TYPE \
                                    $DOE_XSTOCK_INSERT_WEATHER_DATA \
                                        $DOE_XSTOCK_INSERT_YEAR_OF_PUBLICATION \
                                            $DOE_XSTOCK_INSERT_RELEASE \
                                                set_lstm_train_data \
                                                    $bldg_id \
                                                        -s $DOE_XSTOCK_DEFAULT_SEED \
                                                            -r $DOE_XSTOCK_LSTM_PARTIAL_LOAD_SIMULATION_ITERATIONS \
                                                                || exit 1
            done
        } < $neighborhood_filepath

    done
} < $DOE_XSTOCK_INSERT_FILTERS_FILEPATH
IFS=$OLDIFS