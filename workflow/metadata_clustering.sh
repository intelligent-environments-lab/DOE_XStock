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
        python -m doe_xstock
            -d $DOE_XSTOCK_DATABASE_FILEPATH \
                -t $DOE_XSTOCK_DATA_DIRECTORY \
                    -g $DOE_XSTOCK_FIGURE_DIRECTORY \
                        dataset \
                            $DOE_XSTOCK_INSERT_DATASET_TYPE \
                                $DOE_XSTOCK_INSERT_WEATHER_DATA \
                                    $DOE_XSTOCK_INSERT_YEAR_OF_PUBLICATION \
                                        $DOE_XSTOCK_INSERT_RELEASE \
                                            -f $filters \
                                                metadata_clustering \
                                                    $neighborhood_id \
                                                        $DOE_XSTOCK_MAXIMUM_N_CLUSTERS \
                                                            -c $DOE_XSTOCK_NEIGHBORHOOD_SIZE \
                                                                || exit 1
    done
} < $DOE_XSTOCK_INSERT_FILTERS_FILEPATH
IFS=$OLDIFS
