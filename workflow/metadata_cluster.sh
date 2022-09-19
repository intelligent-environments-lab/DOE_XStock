DATABASE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db"
FIGURE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/figures"
MAXIMUM_N_CLUSTERS="10"
SAMPLE_COUNT="100"

python -m doe_xstock -d $DATABASE_FILEPATH -g $FIGURE_FILEPATH metadata_cluster "TX, Travis County" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["TX, Travis County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}' -c $SAMPLE_COUNT
python -m doe_xstock -d $DATABASE_FILEPATH -g $FIGURE_FILEPATH metadata_cluster "VT, Chittenden County" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["VT, Chittenden County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}' -c $SAMPLE_COUNT
python -m doe_xstock -d $DATABASE_FILEPATH -g $FIGURE_FILEPATH metadata_cluster "CA, Alameda County" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["CA, Alameda County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}' -c $SAMPLE_COUNT
