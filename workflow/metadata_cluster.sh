DATABASE_FILEPATH="/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db"
MAXIMUM_N_CLUSTERS="50"

python -m doe_xstock -d $DATABASE_FILEPATH metadata_cluster "travis_county" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["TX, Travis County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}'
python -m doe_xstock -d $DATABASE_FILEPATH metadata_cluster "chittenden_county" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["VT, Chittenden County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}'
python -m doe_xstock -d $DATABASE_FILEPATH metadata_cluster "alameda_county" $MAXIMUM_N_CLUSTERS -f '{"dataset_id": [1], "in_resstock_county_id": ["CA, Alameda County"], "in_geometry_building_type_recs": ["Single-Family Detached"], "in_vacancy_status": ["Occupied"]}'
