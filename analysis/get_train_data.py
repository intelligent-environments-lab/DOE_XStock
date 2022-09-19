from datetime import datetime
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
    idd_filepath = Path('/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd')
    database = SQLiteDatabase(database_filepath)
    iterations = 4
    max_workers = iterations + 1

    # get neighborhood
    neighbourhood = database.query_table(f"""
    SELECT
        location,
        xstock_metadata_id,
        ecobee_building_id
    FROM xstock_ecobee_pair
    """).to_records(index=False)

    # create table to output results to
    database.query("""
    CREATE TABLE IF NOT EXISTS energyplus_simulation (
        id INTEGER NOT NULL,
        metadata_id INTEGER NOT NULL,
        reference INTEGER NOT NULL,
        ecobee_building_id INTEGER,
        PRIMARY KEY (id),
        FOREIGN KEY (metadata_id) REFERENCES metadata (id)
            ON DELETE NO ACTION
            ON UPDATE CASCADE,
        FOREIGN KEY (ecobee_building_id) REFERENCES ecobee_building (id)
            ON DELETE NO ACTION
            ON UPDATE CASCADE,
        UNIQUE (metadata_id, reference)
    );
    CREATE TABLE IF NOT EXISTS weighted_average_zone_air_temperature (
        simulation_id INTEGER NOT NULL,
        timestep INTEGER NOT NULL,
        value REAL NOT NULL,
        PRIMARY KEY (simulation_id, timestep),
        FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );
    CREATE TABLE IF NOT EXISTS lstm_train_data (
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
        PRIMARY KEY (simulation_id, timestep),
        FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
            ON DELETE CASCADE
            ON UPDATE CASCADE
    );""")

    for location, metadata_id, ecobee_id in neighbourhood:
        simulation_data = database.query_table(f"""
        SELECT 
            i.metadata_id,
            i.bldg_osm AS osm, 
            i.bldg_epw AS epw
        FROM energyplus_simulation_input i
        LEFT JOIN metadata m ON m.id = i.metadata_id
        WHERE m.id = {metadata_id}
        """)
        simulation_data = simulation_data.to_dict(orient='records')[0]
        schedules = database.query_table(f"""
        SELECT 
            * 
        FROM schedule 
        WHERE metadata_id = {simulation_data['metadata_id']}
        """)
        schedules = schedules.drop(columns=['metadata_id','timestep',])
        schedules = schedules.to_dict(orient='list')
        setpoints = database.query_table(f"""
        SELECT
            cooling_setpoint,
            heating_setpoint
        FROM ecobee_timeseries
        WHERE building_id = {ecobee_id}
        ORDER BY
            timestamp ASC
        """).to_dict(orient='list')
        simulation_id = f'{metadata_id}'
        output_directory = os.path.join(simulation_output_directory,f'output_{simulation_id}')
        ltd = TrainData(
            idd_filepath,
            simulation_data['osm'],
            simulation_data['epw'],
            schedules,
            setpoints=setpoints,
            ideal_loads_air_system=False,
            edit_ems=True,
            seed=metadata_id,
            iterations=iterations,
            max_workers=max_workers,
            simulation_id=simulation_id,
            output_directory=output_directory,
        )

        # delete any existing simulations for metadata_id
        database.query(f"DELETE FROM energyplus_simulation WHERE metadata_id = {metadata_id}")
        
        # first simulation is to get the weighted average temperature for the as-is model
        ideal_loads_data = pd.DataFrame(ltd.get_ideal_loads_data()['temperature'])
        ideal_loads_data['metadata_id'] = metadata_id
        ideal_load_simulation_query = f"""
        INSERT INTO energyplus_simulation (metadata_id, reference, ecobee_building_id)
        VALUES (:metadata_id, 0, {ecobee_id})
        ;"""
        ideal_load_simulation_values = ideal_loads_data.groupby(['metadata_id']).size().reset_index().to_dict('records')
        ideal_load_temperature_query = f"""
        INSERT INTO weighted_average_zone_air_temperature (simulation_id, timestep, value)
        VALUES (
            (SELECT id FROM energyplus_simulation WHERE metadata_id = :metadata_id AND reference = 0),
            :timestep, :value
        )
        ;"""
        ideal_load_temperature_values = ideal_loads_data.to_dict('records')
        
        # run ideal air loads system sims
        ltd.ideal_loads_air_system = True
        partial_loads_data = ltd.simulate_partial_loads()
        data_list = []

        for simulation_id, data in partial_loads_data.items():
            data = pd.DataFrame(data)
            data['metadata_id'] = metadata_id
            data['reference'] = int(simulation_id.split('_')[-1]) + 1
            data_list.append(data)

        partial_loads_data = pd.concat(data_list, ignore_index=True, sort=False)
        partial_loads_simulation_query = f"""
        INSERT INTO energyplus_simulation (metadata_id, reference, ecobee_building_id)
        VALUES (:metadata_id, :reference, {ecobee_id})
        """
        partial_loads_simulation_values = partial_loads_data.groupby(['metadata_id','reference']).size().reset_index().to_dict('records')
        partial_loads_query = f"""
        INSERT INTO lstm_train_data (
            simulation_id, timestep, month, day, day_name, day_of_week, hour, minute, direct_solar_radiation,
            outdoor_air_temperature, average_indoor_air_temperature, occupant_count, cooling_load, heating_load
        ) VALUES (
            (SELECT id FROM energyplus_simulation WHERE metadata_id = :metadata_id AND reference = :reference),
            :timestep, :month, :day, :day_name, :day_of_week, :hour, :minute, :direct_solar_radiation,
            :outdoor_air_temperature, :average_indoor_air_temperature, :occupant_count, :cooling_load, :heating_load
        );
        """
        partial_loads_values = partial_loads_data.to_dict('records')
        database.insert_batch(
            [ideal_load_simulation_query, ideal_load_temperature_query, partial_loads_simulation_query, partial_loads_query],
            [ideal_load_simulation_values, ideal_load_temperature_values, partial_loads_simulation_values, partial_loads_values]
        )

        assert False

if __name__ == '__main__':
    main()