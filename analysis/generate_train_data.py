import os
from pathlib import Path
import sys
sys.path.insert(0,'..')
from doe_xstock.database import SQLiteDatabase
from doe_xstock.lstm_data import LSTMData
from doe_xstock.utilities import read_json

DATABASE_FILEPATH = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db')
SIMULATION_OUTPUT_DIRECTORY = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/energyplus_simulation')
NEIGHBOURHOOD_FILEPATH = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/data/neighborhoods/travis_county.json')
SCHEDULES_FILENAME = Path('schedules.csv')
IDD_FILEPATH = Path('/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd')
NEIGHBOURHOOD = read_json(NEIGHBOURHOOD_FILEPATH)
DATABASE = SQLiteDatabase(DATABASE_FILEPATH)

def main():
    bldg_id = NEIGHBOURHOOD[0]
    simulation_data = DATABASE.query_table(f"""
    SELECT 
        i.metadata_id,
        i.bldg_osm AS osm, 
        i.bldg_epw AS epw
    FROM building_energy_performance_simulation_input i
    LEFT JOIN metadata m ON m.id = i.metadata_id
    WHERE m.bldg_id = {bldg_id}
    """)
    simulation_data = simulation_data.to_dict(orient='records')[0]
    schedules = DATABASE.query_table(f"""SELECT * FROM schedule WHERE metadata_id = {simulation_data['metadata_id']}""")
    schedules = schedules.drop(columns=['metadata_id','timestep',])
    schedules = schedules.to_dict(orient='list')
    simulation_id = f'{bldg_id}'
    output_directory = os.path.join(SIMULATION_OUTPUT_DIRECTORY,f'output_{simulation_id}')
    ld = LSTMData(
        IDD_FILEPATH,
        simulation_data['osm'],
        simulation_data['epw'],
        schedules,
        ideal_loads=False,
        simulation_id=simulation_id,
        output_directory=output_directory,
        max_workers=4,
    )
    ld.run()

if __name__ == '__main__':
    main()