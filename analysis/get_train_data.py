import os
from pathlib import Path
import sys
sys.path.insert(0,'..')
import pandas as pd
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
    max_workers = iterations + 1

    # create table to output results to
    database.query("""
    CREATE TABLE IF NOT EXISTS weighted_average_zone_air_temperature (
        metadata_id INTEGER NOT NULL,
        timestep INTEGER NOT NULL,
        value,
        PRIMARY KEY (metadata_id, timestep),
        FOREIGN KEY (metadata_id) REFERENCES metadata (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );

    CREATE TABLE IF NOT EXISTS lstm_train_data (
        metadata_id INTEGER NOT NULL,
        simulation_id INTEGER NOT NULL,
        timestep INTEGER NOT NULL,
        month INTEGER NOT NULL,
        day INTEGER NOT NULL,
        day_name TEXT NOT NULL,
        day_of_week INTEGER NOT NULL,
        hour INTEGER NOT NULL,
        minute INTEGER NOT NULL,
        direct_solar_radiation REAL NOT NULL,
        outdoor_air_temperature REAL NOT NULL,
        average_indoor_air_temperature REAL NOT NULL,
        occupant_count INTEGER NOT NULL,
        cooling_load REAL NOT NULL,
        heating_load REAL NOT NULL,
        PRIMARY KEY (metadata_id, simulation_id, timestep),
        FOREIGN KEY (metadata_id) REFERENCES metadata (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );""")

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
        metadata_id = simulation_data['metadata_id']
        output_directory = os.path.join(simulation_output_directory,f'output_{simulation_id}')
        ltd = TrainData(
            idd_filepath,
            simulation_data['osm'],
            simulation_data['epw'],
            schedules,
            ideal_loads_air_system=False,
            edit_ems=True,
            seed=bldg_id,
            iterations=iterations,
            max_workers=max_workers,
            simulation_id=simulation_id,
            output_directory=output_directory,
        )
        
        # first simulation is to get the weighted average temperature for the as-is model
        ideal_loads_data = pd.DataFrame(ltd.get_ideal_loads_data()['temperature'])
        ideal_loads_data['metadata_id'] = metadata_id
        database.insert(
            'weighted_average_zone_air_temperature',
            ideal_loads_data.columns.tolist(),
            ideal_loads_data.values,
            on_conflict_fields=['metadata_id','timestep'],
            ignore_on_conflict=False,
        )

        # run ideal air loads system sims
        ltd.ideal_loads_air_system = True
        partial_loads_data = ltd.simulate_partial_loads()
        database.query(f"DELETE FROM lstm_train_data WHERE metadata_id = {metadata_id}")

        for simulation_id, data in partial_loads_data.items():
            data = pd.DataFrame(data)
            data['metadata_id'] = metadata_id
            data['simulation_id'] = int(simulation_id.split('_')[-1])
            database.insert('lstm_train_data',data.columns.tolist(),data.values,)

if __name__ == '__main__':
    main()