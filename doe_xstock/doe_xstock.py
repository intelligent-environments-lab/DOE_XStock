from datetime import timedelta
from enum import Enum
import gzip
import io
import logging
import logging.config
import os
import uuid
from bs4 import BeautifulSoup
import pandas as pd
import psychrolib
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
from doe_xstock.data import MeteostatWeather
from doe_xstock.database import SQLiteDatabase
from doe_xstock.exploration import MetadataClustering
from doe_xstock.lstm import TrainData
from doe_xstock.simulate import OpenStudioModelEditor, Simulator
from doe_xstock.utilities import read_json, write_data

logging_config = read_json(os.path.join(os.path.dirname(__file__),'misc/logging_config.json'))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class DOEXStock:
    DEFAULT_DATABASE_FILEPATH = 'doe_xstock.db'

    @staticmethod
    def insert(**kwargs):
        LOGGER.info(f'Started insert.')
        database = DOEXStockDatabase(
            kwargs.pop('filepath',DOEXStock.DEFAULT_DATABASE_FILEPATH),
            overwrite=kwargs.pop('overwrite',False),
            apply_changes=kwargs.pop('apply_changes',False)
        )
        filters = kwargs.pop('filters', None)
        dataset = {key:value for key,value in kwargs.items() if key in ['dataset_type','weather_data','year_of_publication','release']}
        database.insert_dataset(dataset, filters=filters)
        LOGGER.info(f'Ended insert.')


    @staticmethod
    def metadata_clustering(**kwargs):
        LOGGER.info(f'Started metadata cluster.')
        dataset = {key:value for key,value in kwargs.items() if key in ['dataset_type','weather_data','year_of_publication','release']}
        mc = MetadataClustering(
            kwargs.get('filepath',DOEXStock.DEFAULT_DATABASE_FILEPATH),
            dataset,
            kwargs.get('name', str(uuid.uuid1())),
            kwargs.get('maximum_n_clusters',10),
            data_directory=kwargs.get('data_directory'),
            figure_directory=kwargs.get('figure_directory'),
            minimum_n_clusters=kwargs.get('minimum_n_clusters'),
            filters=kwargs.get('filters'),
            sample_count=kwargs.get('sample_count'),
            seed=kwargs.get('seed')
        )
        mc.cluster()
        LOGGER.info(f'Ended metadata cluster.')

    @staticmethod
    def set_lstm_train_data(**kwargs):
        # initializa database
        database = DOEXStockDatabase(kwargs.pop('filepath',DOEXStock.DEFAULT_DATABASE_FILEPATH))
        TrainData.initialize_database(database)

        # store simulation variables
        simulation_output_directory = kwargs.get('energyplus_output_directory','energyplus_output')
        simulation_output_directory
        idd_filepath = kwargs['idd_filepath']
        iterations = kwargs.get('iterations',4)
        max_workers = iterations + 1
        dataset_type = kwargs['dataset_type']
        weather_data = kwargs['weather_data']
        year_of_publication = kwargs['year_of_publication']
        release = kwargs['release']
        bldg_id = kwargs['bldg_id']

        # get relevant simulation input parameters
        simulation_data = database.query_table(f"""
        SELECT 
            i.metadata_id,
            i.bldg_osm AS osm, 
            i.bldg_epw AS epw,
            i.ecobee_bldg_id AS ecobee_id
        FROM energyplus_simulation_input i
        WHERE 
            i.dataset_type = '{dataset_type}'
            AND i.dataset_weather_data = '{weather_data}'
            AND i.dataset_year_of_publication = {year_of_publication}
            AND i.dataset_release = {release}
            AND i.bldg_id = {bldg_id}
        """)
        simulation_data = simulation_data.to_dict(orient='records')[0]
        metadata_id = simulation_data['metadata_id']
        ecobee_id = simulation_data['ecobee_id']
        schedules = database.query_table(f"""
        SELECT 
            * 
        FROM schedule 
        WHERE metadata_id = {metadata_id}
        """)
        schedules = schedules.drop(columns=['metadata_id','timestep',])
        schedules = schedules.to_dict(orient='list')

        # use ecobee setpoint if available
        if simulation_data['ecobee_id']:
            setpoints = database.query_table(f"""
            SELECT
                setpoint
            FROM ecobee_timeseries
            WHERE building_id = {ecobee_id}
            ORDER BY
                timestep ASC
            """).to_dict(orient='list')
        else:
            setpoints = None

        # simulation ID is built from dataset reference and bldg_id
        simulation_id = f'{dataset_type}-{weather_data}-{year_of_publication}-release-{release}-{bldg_id}'\
            if kwargs.get('simulation_id') is None else kwargs['simulation_id']
        output_directory = os.path.join(simulation_output_directory,f'output_{simulation_id}')
        
        # initialize lstm train data class
        ltd = TrainData(
            idd_filepath,
            simulation_data['osm'],
            simulation_data['epw'],
            schedules,
            setpoints=setpoints,
            ideal_loads_air_system=False,
            edit_ems=True,
            seed=kwargs.get('seed',simulation_id),
            iterations=iterations,
            max_workers=max_workers,
            simulation_id=simulation_id,
            output_directory=output_directory,
        )
        queries, values, partial_loads_data_list = [], [], []

        # delete any existing simulations for metadata_id
        database.query(f"""
        PRAGMA foreign_keys = ON;
        DELETE FROM energyplus_simulation WHERE metadata_id = {metadata_id};
        """)
        
        # first simulation is to get the weighted average temperature for the as-is model (mechanical system loads)
        mechanical_loads_reference = 0
        ltd.update_kwargs('simulation_id',f'{ltd.kwargs["simulation_id"]}-{mechanical_loads_reference}-mechanical')
        ltd.update_kwargs('output_directory',f'{ltd.kwargs["output_directory"]}-{mechanical_loads_reference}-mechanical')
        mechanical_loads_data = pd.DataFrame(ltd.get_ideal_loads_data()['temperature'])
        ltd.update_kwargs('simulation_id',simulation_id)
        ltd.update_kwargs('output_directory',output_directory)
        mechanical_loads_data['metadata_id'] = metadata_id
        queries.append(f"""
        INSERT INTO energyplus_simulation (metadata_id, reference, ecobee_building_id)
        VALUES (:metadata_id, {mechanical_loads_reference}, {ecobee_id if ecobee_id is not None else 'NULL'})
        ;""")
        values.append(mechanical_loads_data.groupby(['metadata_id']).size().reset_index().to_dict('records'))
        queries.append(f"""
        INSERT INTO energyplus_mechanical_system_simulation (simulation_id, timestep, average_indoor_air_temperature)
        VALUES (
            (SELECT id FROM energyplus_simulation WHERE metadata_id = :metadata_id AND reference = {mechanical_loads_reference}),
            :timestep, :value
        );""")
        values.append(mechanical_loads_data.to_dict('records'))
        
        # run ideal air loads system and other equipment simulations
        ltd.ideal_loads_air_system = True
        ideal_loads_reference = 1
        ideal_loads_data_temp, partial_loads_data = ltd.simulate_partial_loads(ideal_loads_reference=ideal_loads_reference)
        ideal_loads_data = pd.DataFrame(ideal_loads_data_temp['load'])
        ideal_loads_data = ideal_loads_data.groupby(['timestep'])[['cooling','heating']].sum().reset_index()
        ideal_loads_data['average_indoor_air_temperature'] = ideal_loads_data_temp['temperature']['value']
        ideal_loads_data['metadata_id'] = metadata_id
        queries.append(f"""
        INSERT INTO energyplus_simulation (metadata_id, reference, ecobee_building_id)
        VALUES (:metadata_id, {ideal_loads_reference}, {ecobee_id if ecobee_id is not None else 'NULL'})
        ;""")
        values.append(ideal_loads_data.groupby(['metadata_id']).size().reset_index().to_dict('records'))
        queries.append(f"""
        INSERT INTO energyplus_ideal_system_simulation (simulation_id, timestep, average_indoor_air_temperature, cooling_load, heating_load)
        VALUES (
            (SELECT id FROM energyplus_simulation WHERE metadata_id = :metadata_id AND reference = {ideal_loads_reference}),
            :timestep, :average_indoor_air_temperature, :cooling, :heating
        );""")
        values.append(ideal_loads_data.to_dict('records'))

        for simulation_id, data in partial_loads_data.items():
            data = pd.DataFrame(data)
            data['metadata_id'] = metadata_id
            data['reference'] = int(simulation_id.split('-')[-2])
            partial_loads_data_list.append(data)

        partial_loads_data = pd.concat(partial_loads_data_list, ignore_index=True, sort=False)
        queries.append(f"""
        INSERT INTO energyplus_simulation (metadata_id, reference, ecobee_building_id)
        VALUES (:metadata_id, :reference, {ecobee_id if ecobee_id is not None else 'NULL'})
        ;""")
        values.append(partial_loads_data.groupby(['metadata_id','reference']).size().reset_index().to_dict('records'))
        queries.append(f"""
        INSERT INTO lstm_train_data (
            simulation_id, timestep, month, day, day_name, day_of_week, hour, minute, direct_solar_radiation,
            outdoor_air_temperature, average_indoor_air_temperature, occupant_count, cooling_load, heating_load
        ) VALUES (
            (SELECT id FROM energyplus_simulation WHERE metadata_id = :metadata_id AND reference = :reference),
            :timestep, :month, :day, :day_name, :day_of_week, :hour, :minute, :direct_solar_radiation,
            :outdoor_air_temperature, :average_indoor_air_temperature, :occupant_count, :cooling_load, :heating_load
        );""")
        values.append(partial_loads_data.to_dict('records'))
        database.insert_batch(queries, values)

    @staticmethod    
    def simulate(**kwargs):
        LOGGER.info(f'Started simulation.')
        dataset_type = kwargs['dataset_type']
        weather_data = kwargs['weather_data']
        year_of_publication = kwargs['year_of_publication']
        release = kwargs['release']
        bldg_id = kwargs['bldg_id']
        upgrade = kwargs['upgrade']
        simulation_id = f'{dataset_type}_{weather_data}_{year_of_publication}_release_{release}_{bldg_id}_{upgrade}'
        LOGGER.info(f'Simulation ID: {simulation_id}')
        database_filepath = kwargs.pop('filepath',None)
        assert os.path.isfile(database_filepath), f'Database with filepath {database_filepath} does not exist. Initalize database first.' 
        database = DOEXStockDatabase(database_filepath,overwrite=kwargs.pop('overwrite',False),apply_changes=kwargs.pop('apply_changes',False))
        idd_filepath = kwargs['idd_filepath']
        root_output_directory = kwargs.get('root_output_directory','')
        # get input data for simulation
        simulation_data = database.query_table(f"""
        SELECT 
            i.metadata_id,
            i.bldg_osm AS osm, 
            i.bldg_epw AS epw
        FROM energyplus_simulation_input i
        LEFT JOIN metadata m ON m.id = i.metadata_id
        WHERE 
            i.dataset_type = '{dataset_type}'
            AND i.dataset_weather_data = '{weather_data}'
            AND i.dataset_year_of_publication = {year_of_publication}
            AND i.dataset_release = {release}
            AND i.bldg_id = {bldg_id}
            AND i.bldg_upgrade = {upgrade}
        """)
        simulation_data = simulation_data.to_dict(orient='records')[0]
        assert simulation_data['osm'] is not None, f'osm not found.'
        assert simulation_data['epw'] is not None, f'epw not found.'
        output_directory = f'output_{simulation_id}'
        output_directory = os.path.join(root_output_directory,output_directory) if root_output_directory is not None else output_directory
        schedules_filename = 'schedules.csv'
        schedule = database.query_table(f"""SELECT * FROM schedule WHERE metadata_id = {simulation_data['metadata_id']}""")
        schedule = schedule.drop(columns=['metadata_id','timestep',])
         
        # simulate
        try:
            osm_editor = OpenStudioModelEditor(simulation_data['osm'])
            schedule.to_csv(schedules_filename,index=False)
            idf = osm_editor.forward_translate()
            simulator = Simulator(idd_filepath,idf,simulation_data['epw'],simulation_id=simulation_id,output_directory=output_directory)
            simulator.simulate()
        except Exception as e:
            raise e
        finally:
            os.remove(schedules_filename)
        
        write_data(simulation_data['osm'],os.path.join(output_directory,f'{simulation_id}.osm'))
        schedule.to_csv(os.path.join(output_directory,schedules_filename),index=False) # write to output directory
        LOGGER.info(f'Finished simulation.')

class DOEXStockDatabase(SQLiteDatabase):
    __ROOT_URL = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/'

    def __init__(self,filepath,overwrite=False,apply_changes=False):
        super().__init__(filepath)
        self.__build(overwrite,apply_changes)

    def __build(self,overwrite,apply_changes):
        schema_filepath = os.path.join(os.path.dirname(__file__),'misc/schema.sql')
        
        if os.path.isfile(self.filepath):
            if overwrite:
                os.remove(self.filepath)
            elif not apply_changes:
                return
            else:
                pass
        else:
            pass

        self.execute_sql_from_file(schema_filepath)

    def insert_dataset(self,dataset, filters):
        LOGGER.info(f'Updating dataset table.')
        dataset_id = self.update_dataset_table(dataset)
        LOGGER.info(f'Updating data_dictionary table.')
        self.update_data_dictionary_table(dataset,dataset_id)
        LOGGER.info(f'Updating metadata table.')
        buildings = self.update_metadata_table(dataset,dataset_id,filters=filters)

        if buildings is not None:
            LOGGER.info(f'Updating upgrade table.')
            self.update_upgrade_table(dataset,dataset_id,buildings['upgrade'].unique())
            LOGGER.info(f'Updating spatial_tract table.')
            self.update_spatial_tract_table(dataset,dataset_id,buildings)
            LOGGER.info(f'Updating weather table.')
            self.update_weather_table(dataset_id,dataset,buildings)
            LOGGER.info(f'Updating timeseries table.')
            
            # NOTE: Will skip downloading simulation time series to save on database size
            # self.update_timeseries_table(dataset,buildings)
            # LOGGER.info(f'Updating model table.')

            self.update_model_table(dataset,buildings)
            LOGGER.info(f'Updating schedule table.')
            self.update_schedule_table(dataset,buildings)
        else:
            pass
        
    def update_dataset_table(self,dataset):
        self.insert(
            'dataset',
            list(dataset.keys()),
            [tuple(dataset.values())],
            on_conflict_fields=list(dataset.keys())
        )
        dataset_id = self.query_table(f"""
            SELECT
                id
            FROM dataset
            WHERE
                dataset_type = '{dataset['dataset_type']}'
                AND weather_data = '{dataset['weather_data']}'
                AND year_of_publication = {dataset['year_of_publication']}
                AND release = {dataset['release']}
        """).iloc[0]['id']
        return dataset_id

    def update_data_dictionary_table(self,dataset,dataset_id):
        data = DOEXStockDatabase.download_summary_data(DOEXStockDatabase.SummaryType.DATA_DICTIONARY,**dataset)
        data.columns = [c.replace('.','_').lower() for c in data.columns]
        data['dataset_id'] = dataset_id
        self.insert(
            'data_dictionary',
            data.columns.tolist(),
            data.to_records(index=False),
            ['dataset_id','field_location','field_name']
        )

    def update_metadata_table(self,dataset,dataset_id,filters=None):
        data = DOEXStockDatabase.download_summary_data(DOEXStockDatabase.SummaryType.METADATA,**dataset)
        data = data.reset_index(drop=False)
        data.columns = [c.replace('.','_').lower() for c in data.columns]

        if filters is not None:
            for column, values in filters.items():
                data = data[data[column].isin(values)].copy()
        else:
            pass

        if data.shape[0] > 0:
            data['dataset_id'] = dataset_id
            self.insert(
                'metadata',
                data.columns.tolist(),
                data.to_records(index=False),
                ['bldg_id','dataset_id','upgrade']
            )
            buildings = self.query_table(f"""
                SELECT
                    bldg_id,
                    id AS metadata_id,
                    in_county AS county,
                    upgrade,
                    in_nhgis_county_gisjoin,
                    in_nhgis_puma_gisjoin,
                    in_weather_file_latitude,
                    in_weather_file_longitude,
                    in_weather_file_tmy3
                FROM metadata
                WHERE
                    bldg_id IN {tuple(data['bldg_id'].tolist())}
                    AND upgrade IN {tuple(data['upgrade'].unique().tolist())}
                    AND dataset_id = {dataset_id}
            """
            )
            LOGGER.info(f'Found {buildings.shape[0]} buildings that match filters.')
            return buildings
        else:
            LOGGER.warning('Did not find any buildings that match filters.')
            return None

    def update_upgrade_table(self,dataset,dataset_id,upgrade_ids):
        data = DOEXStockDatabase.download_summary_data(DOEXStockDatabase.SummaryType.UPGRADE_DICTIONARY,**dataset)
        data['dataset_id'] = dataset_id
        data.columns = [c.replace('.','_').lower() for c in data.columns]
        data = data[data['upgrade_id'].isin(upgrade_ids)].copy()
        
        if data.shape[0] > 0:
            self.insert(
                'upgrade_dictionary',
                data.columns.tolist(),
                data.to_records(index=False),
                ['dataset_id','upgrade_id']
            )
        else:
            pass

    def update_spatial_tract_table(self,dataset,dataset_id,buildings):
        data = DOEXStockDatabase.download_summary_data(DOEXStockDatabase.SummaryType.SPATIAL_TRACT,**dataset)
        data['dataset_id'] = dataset_id
        columns = ['in_nhgis_county_gisjoin','in_nhgis_puma_gisjoin']
        buildings = buildings.groupby(columns).size()
        buildings = buildings.reset_index(drop=False)[columns].copy()
        buildings.columns = [c.replace('in_','') for c in columns]
        data = pd.merge(data,buildings,on=buildings.columns.tolist(),how='left')
        data.columns = [c.replace('.','_').lower() for c in data.columns]
        self.insert(
            'spatial_tract',
            data.columns.tolist(),
            data.to_records(index=False),
            ['dataset_id','nhgis_tract_gisjoin']
        )

    def update_weather_table(self,dataset_id,dataset,buildings):
        tmy3 = self.download_energyplus_weather_metadata()
        tmy3 = tmy3[tmy3['provider']=='TMY3'].copy()
        tmy3[['longitude','latitude']] = tmy3[['longitude','latitude']].astype(str)
        tmy3 = tmy3.rename(columns={
            'longitude':'in_weather_file_longitude',
            'latitude':'in_weather_file_latitude',
            'title':'energyplus_title',
        })
        buildings = buildings.groupby(
            ['in_weather_file_latitude','in_weather_file_longitude','in_weather_file_tmy3','in_nhgis_county_gisjoin']
        ).size().reset_index().iloc[:,0:-1]
        buildings = pd.merge(buildings,tmy3,on=['in_weather_file_latitude','in_weather_file_longitude'],how='left')
        buildings['count'] = 1
        report = buildings.groupby(
            ['in_weather_file_latitude','in_weather_file_longitude','in_weather_file_tmy3']
        )[['count']].sum().reset_index()
        locations = report['in_weather_file_tmy3'].unique().tolist()
        unknown_locations = buildings[buildings['energyplus_title'].isnull()]['in_weather_file_tmy3'].unique().tolist()
        ambiguous_locations = report[report['count']>1]['in_weather_file_tmy3'].unique().tolist()
        data = buildings[~buildings['in_weather_file_tmy3'].isin(unknown_locations+ambiguous_locations)].copy()
        LOGGER.info(f'Found {data.shape[0]}/{len(locations)} weather_file_tmy3.')

        if len(unknown_locations) > 0:
            LOGGER.warning(f'{len(unknown_locations)}/{len(locations)} weather_file_tmy3 are unknown including {unknown_locations}.')
        else:
            pass

        if len(ambiguous_locations) > 0:
            LOGGER.warning(f'{len(ambiguous_locations)}/{len(locations)} weather_file_tmy3 are ambiguous including {ambiguous_locations}')
            
        else:
            pass
        
        if len(buildings) > 0:
            session = requests.Session()
            retries = Retry(total=5,backoff_factor=1)
            session.mount('http://',HTTPAdapter(max_retries=retries))
            urllib3.disable_warnings()
            epws = []
            ddys = []
            
            for epw_url, ddy_url, in_nhgis_county_gisjoin in data[['epw_url','ddy_url','in_nhgis_county_gisjoin']].to_records(index=False):
                response = session.get(epw_url)
                epw = response.content.decode()

                if dataset['weather_data'].startswith('amy'):
                    epw = self.__tmy_to_amy_epw(dataset,epw,in_nhgis_county_gisjoin)
                else:
                    pass

                epws.append(epw)
                response = session.get(ddy_url)
                ddys.append(response.content.decode(encoding='windows-1252'))
            
            data = data[[
                'in_weather_file_latitude',
                'in_weather_file_longitude',
                'in_weather_file_tmy3',
                'energyplus_title',
                'epw_url',
                'ddy_url'
            ]].copy()
            data['epw'] = epws
            data['ddy'] = ddys
            data['dataset_id'] = dataset_id
            data.columns = [c.replace('.','_').replace('in_','').lower() for c in data.columns]
            self.insert(
                'weather',
                data.columns.tolist(),
                data.to_records(index=False),
                ['dataset_id','weather_file_tmy3','weather_file_latitude','weather_file_longitude']
            )

        else:
            pass

    def __tmy_to_amy_epw(self,dataset,epw,in_nhgis_county_gisjoin):
        separator = '\r\n'
        epw_table_start_index = 8

        # read epw
        epw = epw.split(separator)
        epw_table = io.StringIO(separator.join(epw[epw_table_start_index:]))
        epw_table = pd.read_csv(epw_table, header=None)

        # location edits
        location = epw[0].split(',')
        location[4] = 'AMY2018'
        epw[0] = ','.join(location)
        latitude, longitude = float(location[6]), float(location[7])

        # read amy
        amy_table = self.__get_amy_table(dataset,in_nhgis_county_gisjoin,latitude,longitude)

        # column mapping
        column_map = {
            'Year': 0,
            'Month': 1,
            'Day': 2,
            'Hour': 3,
            'Minute': 4,
            'Dry Bulb Temperature [°C]': 6,
            'Dew Point Temperature [C]': 7,
            'Relative Humidity [%]': 8,
            'Atmospheric Station Pressure [Pa]': 9,
            'Global Horizontal Radiation [W/m2]': 13,
            'Direct Normal Radiation [W/m2]': 14,
            'Diffuse Horizontal Radiation [W/m2]': 15,
            'Wind Speed [m/s]': 21,
            'Wind Direction [Deg]': 20,
        }

        for k, v in column_map.items():
            ixs = amy_table[amy_table[k].notnull()].index
            epw_table.loc[ixs, v] = amy_table.loc[ixs][k]

        # update start day string
        data_periods = epw[7].split(',')
        data_periods[4] = amy_table['date_time'].min().strftime('%A')
        epw[7] = ','.join(data_periods)

        # amy epw
        epw = epw[:epw_table_start_index]
        epw_table = epw_table.to_csv(sep=',', header=False, index=False,line_terminator=separator)
        epw.append(epw_table)
        epw = separator.join(epw)
        
        return epw

    def __get_amy_table(self,dataset,in_nhgis_county_gisjoin,latitude,longitude):
        year = dataset['weather_data'][3:]
        url = self.__get_dataset_url(**dataset)
        url = os.path.join(url,'weather',dataset['weather_data'],f'{in_nhgis_county_gisjoin}_{year}.csv')
        amy_table = pd.read_csv(url,parse_dates=['date_time'])
        amy_table['date_time'] = amy_table['date_time'] - timedelta(hours=1)
        amy_table['Year'] = amy_table['date_time'].dt.year
        amy_table['Month'] = amy_table['date_time'].dt.month
        amy_table['Day'] = amy_table['date_time'].dt.day
        amy_table['Hour'] = amy_table['date_time'].dt.hour + 1
        amy_table['Minute'] = amy_table['date_time'].dt.minute + 1

        # calculate other fields
        # dew point temperature
        psychrolib.SetUnitSystem(psychrolib.SI)
        amy_table['Dew Point Temperature [C]'] = amy_table.apply(lambda x: psychrolib.GetTDewPointFromRelHum(
            x['Dry Bulb Temperature [°C]'], x['Relative Humidity [%]']/100.0
        ), axis=1)

        # station atmposheric pressure
        station_metadata = MeteostatWeather.get_station_from_coordinates(latitude, longitude, count=1).to_dict('index')
        station_id = list(station_metadata.keys())[0]
        elevation = station_metadata[station_id]['elevation']
        ms = MeteostatWeather(
            [station_id], 
            resolution='hourly', 
            earliest_start_timestamp=amy_table['date_time'].min() + timedelta(hours=1), 
            latest_end_timestamp=amy_table['date_time'].max() + timedelta(hours=1),
            model=True,
            gap_limit=3,
        )
        ms_data = ms.download()
        amy_table['Atmospheric Station Pressure [Pa]'] = ms_data['pres'].map(
            lambda x: ms.convert_sea_level_pressure_to_station_pressure(x, elevation)
        )

        return amy_table
        
    def update_timeseries_table(self,dataset,buildings):
        buildings = buildings[['bldg_id','metadata_id','county','upgrade']].to_records(index=False)
        dataset_url = DOEXStockDatabase.__get_dataset_url(**dataset)

        for i, (bldg_id, metadata_id, county, upgrade) in enumerate(buildings):
            LOGGER.debug(f'Downloading timeseries ({i+1}/{buildings.shape[0]}): bldg_id: {bldg_id}, upgrade: {upgrade}.')
            building_path = f'timeseries_individual_buildings/by_county/upgrade={upgrade}/county={county}/{bldg_id}-{upgrade}.parquet'
            url = os.path.join(dataset_url,building_path)
            data = pd.read_parquet(url)
            data = data.reset_index(drop=False)
            data.columns = [c.replace('.','_').lower() for c in data.columns]
            data['metadata_id'] = metadata_id
            self.insert(
                'timeseries',
                data.columns.tolist(),
                data.to_records(index=False),
                ['metadata_id','timestamp']
            )

    def update_model_table(self,dataset,buildings):
        buildings = buildings[['bldg_id','metadata_id','upgrade']].to_records(index=False)
        dataset_url = DOEXStockDatabase.__get_dataset_url(**dataset)
        values = []

        for i, (bldg_id,metadata_id,upgrade) in enumerate(buildings):
            LOGGER.debug(f'Downloading model ({i+1}/{buildings.shape[0]}): bldg_id: {bldg_id}, upgrade: {upgrade}.')
            building_path = f'building_energy_models/bldg{bldg_id:07d}-up{upgrade:02d}.osm.gz'
            url = os.path.join(dataset_url,building_path)
            response = requests.get(url)
            compressed_file = io.BytesIO(response.content)
            decompressed_file = gzip.GzipFile(fileobj=compressed_file,mode='rb')
            osm = decompressed_file.read().decode()
            values.append((metadata_id,osm))
        
        self.insert(
            'model',
            ['metadata_id','osm'],
            values,
            on_conflict_fields=['metadata_id']
        )

    def update_schedule_table(self,dataset,buildings):
        buildings = buildings[['bldg_id','metadata_id','upgrade']].to_records(index=False)
        dataset_url = DOEXStockDatabase.__get_dataset_url(**dataset)

        for i, (bldg_id,metadata_id,upgrade) in enumerate(buildings):
            LOGGER.debug(f'Downloading schedule ({i+1}/{buildings.shape[0]}): bldg_id: {bldg_id}, upgrade: {upgrade}.')
            building_path = f'occupancy_schedules/bldg{bldg_id:07d}-up{upgrade:02d}.csv.gz'
            url = os.path.join(dataset_url,building_path)
            data = pd.read_csv(url)
            data['metadata_id'] = metadata_id
            data['timestep'] = data.index
            self.insert(
                'schedule',
                data.columns.tolist(),
                data.to_records(index=False),
                on_conflict_fields=['metadata_id','timestep']
            )

    @classmethod
    def download_summary_data(cls,summary_type,dataset_type,weather_data,year_of_publication,release):
        downloader = {
            DOEXStockDatabase.SummaryType.METADATA.name:{
                'url':'metadata/metadata.parquet',
                'reader':pd.read_parquet,
                'reader_kwargs':{}
            },
            DOEXStockDatabase.SummaryType.DATA_DICTIONARY.name:{
                'url':'data_dictionary.tsv',
                'reader':pd.read_csv,
                'reader_kwargs':{'sep':'\t'}
            },
            DOEXStockDatabase.SummaryType.ENUMERATION_DICTIONARY.name:{
                'url':'enumeration_dictionary.tsv',
                'reader':pd.read_csv,
                'reader_kwargs':{'sep':'\t'}
            },
            DOEXStockDatabase.SummaryType.UPGRADE_DICTIONARY.name:{
                'url':'upgrade_dictionary.tsv',
                'reader':pd.read_csv,
                'reader_kwargs':{'sep':'\t'}
            },
            DOEXStockDatabase.SummaryType.SPATIAL_TRACT.name:{
                'url':'geographic_information/spatial_tract_lookup_table.csv',
                'reader':pd.read_csv,
                'reader_kwargs':{'sep':','}
            },
        }[summary_type.name]
        dataset_url = cls.__get_dataset_url(dataset_type,weather_data,year_of_publication,release)
        dataset_url = os.path.join(dataset_url,downloader['url'])
        data = downloader['reader'](dataset_url,**downloader['reader_kwargs'])
        return data

    @classmethod
    def __get_dataset_url(cls,dataset_type,weather_data,year_of_publication,release):
        dataset_path = f'{year_of_publication}/{dataset_type}_{weather_data}_release_{release}/'
        return os.path.join(DOEXStockDatabase.__ROOT_URL,dataset_path)

    @classmethod
    def download_energyplus_weather_metadata(cls):
        url = 'https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/master.geojson'
        response = requests.get(url)
        response = response.json()
        features = response['features']
        records = []

        for feature in features:
            title = feature['properties']['title']
            epw_url = BeautifulSoup(feature['properties']['epw'],'html.parser').find('a')['href']
            ddy_url = BeautifulSoup(feature['properties']['ddy'],'html.parser').find('a')['href']
            longitude, latitude = tuple(feature['geometry']['coordinates'])
            region = epw_url.split('/')[3]
            country = epw_url.split('/')[4]
            state = epw_url.split('/')[5]
            station_id = title.split('.')[-1].split('_')[0]
            provider = title.split('.')[-1].split('_')[-1]

            records.append({
                'title':title,
                'region':region,
                'country':country,
                'state':state,
                'station_id':station_id,
                'provider':provider,
                'epw_url':epw_url,
                'ddy_url':ddy_url,
                'longitude':longitude,
                'latitude':latitude,
            })

        data = pd.DataFrame(records)
        return data

    class SummaryType(Enum):
        METADATA = 'metadata'
        DATA_DICTIONARY = 'data_dictionary'
        ENUMERATION_DICTIONARY = 'enumaration_dictionary'
        UPGRADE_DICTIONARY = 'upgrade_dictionary'
        SPATIAL_TRACT = 'spatial_tract'
