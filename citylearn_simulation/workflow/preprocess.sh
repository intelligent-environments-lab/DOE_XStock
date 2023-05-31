#!/bin/sh

# size equipment
python src/preprocess.py ca_alameda_county_neighborhood winter size_equipment || exit 1
python src/preprocess.py tx_travis_county_neighborhood summer size_equipment || exit 1
python src/preprocess.py vt_chittenden_county_neighborhood winter size_equipment || exit 1

# set schema and work order
python src/preprocess.py ca_alameda_county_neighborhood winter set_sb3_work_order alameda || exit 1
python src/preprocess.py tx_travis_county_neighborhood summer set_sb3_work_order travis || exit 1
python src/preprocess.py vt_chittenden_county_neighborhood winter set_sb3_work_order chittenden || exit 1
