import os
from pathlib import Path
import sys
sys.path.insert(0,'..')
from doe_xstock.database import SQLiteDatabase
from doe_xstock.lstm import TrainData
from doe_xstock.utilities import read_json

def main():
    database_filepath = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db')
    simulation_output_directory = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/energyplus_simulation')
    neighbourhood_filepath = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/data/neighborhoods/travis_county.json')
    idd_filepath = Path('/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd')
    neighbourhood = read_json(neighbourhood_filepath)
    database = SQLiteDatabase(database_filepath)
    iterations = 4
    max_workers = min(iterations,4)

    for bldg_id in neighbourhood:
        simulation_data = database.query_table(f"""
        SELECT 
            i.metadata_id,
            i.bldg_osm AS osm, 
            i.bldg_epw AS epw
        FROM building_energy_performance_simulation_input i
        LEFT JOIN metadata m ON m.id = i.metadata_id
        WHERE m.bldg_id = {bldg_id}
        """)
        simulation_data = simulation_data.to_dict(orient='records')[0]
        schedules = database.query_table(f"SELECT * FROM schedule WHERE metadata_id = {simulation_data['metadata_id']}")
        schedules = schedules.drop(columns=['metadata_id','timestep',])
        schedules = schedules.to_dict(orient='list')
        simulation_id = f'{bldg_id}'
        output_directory = os.path.join(simulation_output_directory,f'output_{simulation_id}')
        ltd = TrainData(
            idd_filepath,
            simulation_data['osm'],
            simulation_data['epw'],
            schedules,
            ideal_loads=True,
            edit_ems=True,
            seed=bldg_id,
            iterations=iterations,
            max_workers=max_workers,
            simulation_id=simulation_id,
            output_directory=output_directory,
        )
        ltd.run()
        break

if __name__ == '__main__':
    main()