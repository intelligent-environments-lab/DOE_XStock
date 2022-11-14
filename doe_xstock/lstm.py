import concurrent.futures
from copy import deepcopy
import logging
import logging.config
import os
from pathlib import Path
import shutil
from eppy.runner.run_functions import EnergyPlusRunError
from eppy.bunch_subclass import BadEPFieldError
import numpy as np
import pandas as pd
from doe_xstock.simulate import OpenStudioModelEditor, Simulator
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class TrainData:
    def __init__(self,idd_filepath,osm,epw,schedules,setpoints=None,ideal_loads_air_system=None,edit_ems=None,output_variables=None,timesteps_per_hour=None,iterations=None,max_workers=None,seed=None,**kwargs):
        self.idd_filepath = idd_filepath
        self.osm = osm
        self.epw = epw
        self.schedules = schedules
        self.setpoints = setpoints
        self.ideal_loads_air_system = ideal_loads_air_system
        self.edit_ems = edit_ems
        self.output_variables = output_variables
        self.timesteps_per_hour = timesteps_per_hour
        self.iterations = iterations
        self.max_workers = max_workers
        self.seed = seed
        self.__kwargs = kwargs
        self.__simulator = None
        self.__ideal_loads_data = None
        self.__zones = None
        self.__partial_loads_data = None

    @property
    def idd_filepath(self):
        return self.__idd_filepath

    @property
    def osm(self):
        return self.__osm

    @property
    def epw(self):
        return self.__epw

    @property
    def schedules(self):
        return self.__schedules

    @property
    def setpoints(self):
        return self.__setpoints

    @property
    def ideal_loads_air_system(self):
        return self.__ideal_loads_air_system

    @property
    def edit_ems(self):
        return self.__edit_ems

    @property
    def output_variables(self):
        return self.__output_variables
    
    @property
    def timesteps_per_hour(self):
        return self.__timesteps_per_hour

    @property
    def iterations(self):
        return self.__iterations

    @property
    def max_workers(self):
        return self.__max_workers

    @property
    def seed(self):
        return self.__seed

    @property
    def kwargs(self):
        return self.__kwargs

    @idd_filepath.setter
    def idd_filepath(self,idd_filepath):
        self.__idd_filepath = idd_filepath

    @osm.setter
    def osm(self,osm):
        self.__osm = osm

    @epw.setter
    def epw(self,epw):
        self.__epw = epw

    @schedules.setter
    def schedules(self, schedules):
        self.__schedules = schedules

    @setpoints.setter
    def setpoints(self, setpoints):
        self.__setpoints = setpoints

    @ideal_loads_air_system.setter
    def ideal_loads_air_system(self,ideal_loads_air_system):
        self.__ideal_loads_air_system = False if ideal_loads_air_system is None else ideal_loads_air_system
    
    @edit_ems.setter
    def edit_ems(self,edit_ems):
        self.__edit_ems = True if edit_ems is None else edit_ems

    @output_variables.setter
    def output_variables(self,output_variables):
        default_output_variables = [
            'Site Direct Solar Radiation Rate per Area', 'Site Diffuse Solar Radiation Rate per Area',
            'Site Outdoor Air Drybulb Temperature','Zone People Occupant Count',
            'Zone Air Temperature','Zone Thermostat Cooling Setpoint Temperature','Zone Thermostat Heating Setpoint Temperature',
            'Zone Air Relative Humidity',
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate','Other Equipment Convective Heating Rate',
            'Zone Ideal Loads Zone Sensible Cooling Rate','Zone Ideal Loads Zone Sensible Heating Rate',
            'Zone Air System Sensible Cooling Rate','Zone Air System Sensible Heating Rate',
            'Water Heater Use Side Heat Transfer Energy',
            'Exterior Lights Electricity Energy','Lights Electricity Energy','Electric Equipment Electricity Energy',
        ]
        self.__output_variables =default_output_variables if output_variables is None else output_variables

    @timesteps_per_hour.setter
    def timesteps_per_hour(self,timesteps_per_hour):
        self.__timesteps_per_hour = timesteps_per_hour = 1 if timesteps_per_hour is None else timesteps_per_hour

    @iterations.setter
    def iterations(self,iterations):
        self.__iterations = 3 if iterations is None else iterations

    @max_workers.setter
    def max_workers(self,max_workers):
        self.__max_workers = 1 if max_workers is None else max_workers

    @seed.setter
    def seed(self,seed):
        self.__seed = 0 if seed is None else seed

    def update_kwargs(self,key,value):
        self.__kwargs[key] = value

    def simulate_partial_loads(self,**kwargs):
        LOGGER.info('Started simulation.')
        ideal_loads_reference = kwargs.get('ideal_loads_reference',1)
        initial_simulation_id = self.kwargs["simulation_id"]
        initial_output_directory = self.kwargs["output_directory"]
        self.__kwargs['simulation_id'] = f'{self.kwargs["simulation_id"]}-{ideal_loads_reference}-ideal'
        self.__kwargs['output_directory'] = f'{self.kwargs["output_directory"]}-{ideal_loads_reference}-ideal'
        self.__set_ideal_loads(**kwargs)
        self.__kwargs['simulation_id'] = initial_simulation_id
        self.__kwargs['output_directory'] = initial_output_directory
        self.__transform_idf()
        seeds = [None] + [i for i in range(self.iterations + 1)]
        simulators = [self.__get_partial_load_simulator(i + ideal_loads_reference + 1,seed=s) for i, s in enumerate(seeds)]
        LOGGER.debug('Simulating partial load iterations.')
        Simulator.multi_simulate(simulators,max_workers=self.max_workers)
        self.__partial_loads_data = {}

        LOGGER.debug('Post processing partial load iterations.')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = [executor.submit(self.__post_process_partial_load_simulation,*[s]) for s in simulators]

            for _, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    r = future.result()
                    self.__partial_loads_data[r[0]] = deepcopy(r[1])
                    LOGGER.debug(f'Finished processing simulaton_id:{r[0]}')
                except Exception as e:
                    LOGGER.exception(e)
        
        LOGGER.info('Ended simulation.')
        return self.__ideal_loads_data, self.__partial_loads_data

    def __post_process_partial_load_simulation(self,simulator):
        # check that air system cooling and heating loads are 0
        query = """
        SELECT
            d.Name as name,
            SUM(r.Value) AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE d.Name IN ('Zone Air System Sensible Cooling Rate','Zone Air System Sensible Heating Rate')
        GROUP BY d.Name
        """
        data = simulator.get_database().query_table(query)
        air_system_cooling = data[data['name']=='Zone Air System Sensible Cooling Rate']['value'].iloc[0]
        air_system_heating = data[data['name']=='Zone Air System Sensible Heating Rate']['value'].iloc[0]

        try:
            assert air_system_cooling == 0 and air_system_heating == 0
        except AssertionError:
            LOGGER.warning(f'simulation_id-{simulator.simulation_id}: Non-zero Zone Air System Sensible Cooling Rate'\
                f' and/or Zone Air System Sensible Heating Rate: ({air_system_cooling, air_system_heating})')
            
        # create zone conditioning table
        self.__insert_zone_metadata(simulator)

        # get simulation summary
        query = """
        WITH u AS (
            -- site variables
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                'site_variable' AS label,
                r.Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            WHERE d.Name IN ('Site Direct Solar Radiation Rate per Area', 'Site Diffuse Solar Radiation Rate per Area', 'Site Outdoor Air Drybulb Temperature')

            UNION ALL

            -- weighted conditioned zone variables
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                'weighted_variable' AS label,
                r.Value
            FROM weighted_variable r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            WHERE d.Name IN ('Zone Air Temperature')

            UNION ALL

            -- thermal load variables
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                CASE WHEN r.Value > 0 THEN 'heating_load' ELSE 'cooling_load' END AS label,
                ABS(r.Value) AS Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            WHERE 
                d.Name = 'Other Equipment Convective Heating Rate' AND
                (d.KeyValue LIKE '%HEATING LOAD' OR d.KeyValue LIKE '%COOLING LOAD')

            UNION ALL

            -- other variables
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                'occupant_count' AS label,
                r.Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            WHERE 
                d.Name = 'Zone People Occupant Count'
        ), p AS (
            SELECT
                u.TimeIndex,
                MAX(CASE WHEN d.Name = 'Site Direct Solar Radiation Rate per Area' THEN Value END) AS direct_solar_radiation,
                MAX(CASE WHEN d.Name = 'Site Diffuse Solar Radiation Rate per Area' THEN Value END) AS diffuse_solar_radiation,
                MAX(CASE WHEN d.Name = 'Site Outdoor Air Drybulb Temperature' THEN Value END) AS outdoor_air_temperature,
                SUM(CASE WHEN d.Name = 'Zone Air Temperature' THEN Value END) AS average_indoor_air_temperature,
                SUM(CASE WHEN d.Name = 'Zone People Occupant Count' THEN Value END) AS occupant_count,
                SUM(CASE WHEN u.label = 'cooling_load' THEN Value END) AS cooling_load,
                SUM(CASE WHEN u.label = 'heating_load' THEN Value END) AS heating_load
            FROM u
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = u.ReportDataDictionaryIndex
            GROUP BY u.TimeIndex
        )
        SELECT
            t.TimeIndex AS timestep,
            t.Month AS month,
            t.Day AS day,
            t.DayType AS day_name,
            CASE
                WHEN t.DayType = 'Monday' THEN 1
                WHEN t.DayType = 'Tuesday' THEN 2
                WHEN t.DayType = 'Wednesday' THEN 3
                WHEN t.DayType = 'Thursday' THEN 4
                WHEN t.DayType = 'Friday' THEN 5
                WHEN t.DayType = 'Saturday' THEN 6
                WHEN t.DayType = 'Sunday' THEN 7
                WHEN t.DayType = 'Holiday' THEN 8
                ELSE NULL
            END AS day_of_week,
            t.Hour AS hour,
            t.Minute AS minute,
            p.direct_solar_radiation,
            p.diffuse_solar_radiation,
            p.outdoor_air_temperature,
            p.average_indoor_air_temperature,
            p.occupant_count,
            COALESCE(p.cooling_load, 0) AS cooling_load,
            COALESCE(p.heating_load, 0) AS heating_load
        FROM p
        LEFT JOIN Time t ON t.TimeIndex = p.TimeIndex
        WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
        """
        data = simulator.get_database().query_table(query)
        data = data.to_dict('list')
        _ = data.pop('index',None)
        
        return simulator.simulation_id, data

    def __get_partial_load_simulator(self,reference,seed=None):
        # get multiplier
        size = len(self.__ideal_loads_data['load']['timestep'])

        if seed is None:
            multiplier = [1.0]*size
        elif seed == 0:
            multiplier = [0.0]*size
        else:
            multiplier = self.get_multipliers(size,seed=seed)
        
        multiplier = pd.DataFrame(multiplier,columns=['multiplier'])
        multiplier['timestep'] = multiplier.index + 1

        # set load schedule file
        data = pd.DataFrame(self.__ideal_loads_data['load'])
        data = data.merge(multiplier,on='timestep',how='left')
        data = data.sort_values(['zone_name','timestep'])
        data['cooling'] *= data['multiplier']*-1
        data['heating'] *= data['multiplier']

        # save load schedule file
        simulation_id = f'{self.kwargs["simulation_id"]}-{reference}-partial'
        output_directory = f'{self.kwargs["output_directory"]}-{reference}-partial'
        filepath = os.path.join(output_directory,f'{simulation_id}_partial_load.csv')
        os.makedirs(output_directory,exist_ok=True)
        data[['cooling','heating']].to_csv(filepath,index=False)

        # set idf
        idf = self.__simulator.get_idf_object()

        # set load schedule
        for obj in idf.idfobjects['Schedule:File']:
            if 'lstm' in obj.Name.lower():
                obj.File_Name = filepath
            else:
                continue
        
        # convert idf to string
        idf = idf.idfstr()
        idf = idf.replace(self.__simulator.simulation_id,simulation_id)
        
        # update setpoint schedule filename
        source_filepath = os.path.join(self.__simulator.output_directory,f'{self.__simulator.simulation_id}_setpoint.csv')
        destination_filepath = os.path.join(output_directory,f'{simulation_id}_setpoint.csv')
        _ =shutil.copy2(source_filepath,destination_filepath)

        # update equipment schedule filename
        source_filepath = os.path.join(self.__simulator.output_directory,f'{self.__simulator.simulation_id}_schedules.csv')
        destination_filepath = os.path.join(output_directory,f'{simulation_id}_schedules.csv')
        _ =shutil.copy2(source_filepath,destination_filepath)

        return Simulator(self.idd_filepath,idf,self.epw,simulation_id=simulation_id,output_directory=output_directory)
    
    def __transform_idf(self):
        # generate idf for simulations with partial loads
        idf = self.__simulator.get_idf_object()

        # remove Ideal Air Loads System if any
        if self.ideal_loads_air_system:
            idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem'] = []
            obj_names = [
                'ZoneControl:Thermostat','ZoneControl:Humidistat','ZoneControl:Thermostat:ThermalComfort', 'ThermostatSetpoint:DualSetpoint',
                'ZoneControl:Thermostat:OperativeTemperature','ZoneControl:Thermostat:TemperatureAndHumidity','ZoneControl:Thermostat:StagedDualSetpoint'
            ]

            for name in obj_names:
                idf.idfobjects[name] = []
        else:
            # set hvac equipment availability to always off
            obj_names = [n.upper() for n in [
                'AirTerminal:SingleDuct:Uncontrolled','Fan:ZoneExhaust','ZoneHVAC:Baseboard:Convective:Electric','ZoneHVAC:Baseboard:Convective:Water','ZoneHVAC:Baseboard:RadiantConvective:Electric','ZoneHVAC:Baseboard:RadiantConvective:Water',
                'ZoneHVAC:Baseboard:RadiantConvective:Steam','ZoneHVAC:Dehumidifier:DX','ZoneHVAC:EnergyRecoveryVentilator','ZoneHVAC:FourPipeFanCoil',
                'ZoneHVAC:HighTemperatureRadiant','ZoneHVAC:LowTemperatureRadiant:ConstantFlow','ZoneHVAC:LowTemperatureRadiant:Electric',
                'ZoneHVAC:LowTemperatureRadiant:VariableFlow','ZoneHVAC:OutdoorAirUnit','ZoneHVAC:PackagedTerminalAirConditioner',
                'ZoneHVAC:PackagedTerminalHeatPump','ZoneHVAC:RefrigerationChillerSet','ZoneHVAC:UnitHeater','ZoneHVAC:UnitVentilator',
                'ZoneHVAC:WindowAirConditioner','ZoneHVAC:WaterToAirHeatPump','ZoneHVAC:VentilatedSlab',
                'AirTerminal:DualDuct:ConstantVolume','AirTerminal:DualDuct:VAV','AirTerminal:DualDuct:VAV:OutdoorAir',
                'AirTerminal:SingleDuct:ConstantVolume:Reheat','AirTerminal:SingleDuct:VAV:Reheat','AirTerminal:SingleDuct:VAV:NoReheat',
                'AirTerminal:SingleDuct:SeriesPIU:Reheat','AirTerminal:SingleDuct:ParallelPIU:Reheat'
                'AirTerminal:SingleDuct:ConstantVolume:FourPipeInduction','AirTerminal:SingleDuct:VAV:Reheat:VariableSpeedFan',
                'AirTerminal:SingleDuct:VAV:HeatAndCool:Reheat','AirTerminal:SingleDuct:VAV:HeatAndCool:NoReheat',
                'AirLoopHVAC:UnitarySystem','ZoneHVAC:IdealLoadsAirSystem','HVACTemplate:Zone:IdealLoadsAirSystem'
            ]]
            schedule_name = 'Always Off Discrete'

            for name in obj_names:
                if name in idf.idfobjects.keys():
                    for obj in idf.idfobjects[name]:
                        try:
                            obj.Availability_Schedule_Name = schedule_name
                        except BadEPFieldError:
                            obj.System_Availability_Schedule_Name = schedule_name
                else:
                    continue

        # schedule type limit object
        schedule_type_limit_name = 'other equipment hvac power'
        obj = idf.newidfobject('ScheduleTypeLimits')
        obj.Name = schedule_type_limit_name
        obj.Lower_Limit_Value = ''
        obj.Upper_Limit_Value = ''
        obj.Numeric_Type = 'Continuous'
        obj.Unit_Type = 'Dimensionless'

        # generate stochastic thermal load
        zone_names = set(self.__ideal_loads_data['load']['zone_name'])
        timesteps = max(self.__ideal_loads_data['load']['timestep'])
        loads = ['cooling','heating']
        
        for i, zone_name in enumerate(zone_names):
            for j, load in enumerate(loads):
                # put schedule obj
                obj = idf.newidfobject('Schedule:File')
                schedule_object_name = f'{zone_name} lstm {load} load'
                obj.Name = schedule_object_name
                obj.Schedule_Type_Limits_Name = schedule_type_limit_name
                obj.File_Name = ''
                obj.Column_Number = j + 1
                obj.Rows_to_Skip_at_Top = 1 + i*timesteps
                obj.Number_of_Hours_of_Data = 8760
                obj.Minutes_per_Item = int(60/self.timesteps_per_hour)

                # put other equipment
                obj = idf.newidfobject('OtherEquipment')
                obj.Name = f'{zone_name} {load} load'
                obj.Fuel_Type = 'None'
                obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name = zone_name
                obj.Schedule_Name = schedule_object_name
                obj.Design_Level_Calculation_Method = 'EquipmentLevel'
                obj.Design_Level = 1.0
                obj.Fraction_Latent = 0.0
                obj.Fraction_Radiant = 0.0
                obj.Fraction_Lost = 0.0
                obj.EndUse_Subcategory = f'lstm {load}'

        self.__simulator.idf = idf.idfstr()

    def get_multipliers(self,size,seed=0,minimum_value=0.3,maximum_value=1.7,probability=0.85):
        np.random.seed(seed)
        schedule = np.random.uniform(minimum_value, maximum_value, size)
        schedule[np.random.random(size) > probability] = 1.0
        schedule = schedule.tolist()
        return schedule

    def get_ideal_loads_data(self,**kwargs):
        self.__set_ideal_loads(**kwargs)
        return self.__ideal_loads_data

    def __set_ideal_loads(self,**kwargs):
        LOGGER.debug('Simulating ideal loads.')
        self.__set_simulator()
        self.__simulate_ideal_loads(**kwargs)
        self.__set_zones()
        self.__set_ideal_loads_data()
        LOGGER.debug('Finished simulating ideal loads.')

    def __set_ideal_loads_data(self):
        ideal_loads_data = {}

        # cooling and heating loads
        cooled_zones, heated_zones = self.__get_conditioned_zones()
        cooled_zone_names = [f'\'{z}\'' for z in cooled_zones]
        heated_zone_names = [f'\'{z}\'' for z in heated_zones]
        cooling_column = lambda x: f'{x}.value' if self.ideal_loads_air_system else f'CASE WHEN {x}.Value > 0 THEN 0 ELSE r.Value END'
        heating_column = lambda x: f'{x}.value' if self.ideal_loads_air_system else f'CASE WHEN {x}.Value < 0 THEN 0 ELSE r.Value END'
        cooling_variable = 'Zone Ideal Loads Zone Sensible Cooling Rate' if self.ideal_loads_air_system else 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate'
        heating_variable = 'Zone Ideal Loads Zone Sensible Heating Rate' if self.ideal_loads_air_system else cooling_variable
        zone_join_query = lambda x: f"REPLACE({x}.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')" if self.ideal_loads_air_system else f'{x}.KeyValue'
        query = f"""
        SELECT
            r.TimeIndex AS timestep,
            'cooling' AS load,
            z.ZoneIndex AS zone_index,
            z.ZoneName AS zone_name,
            ABS({cooling_column('r')}) AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        LEFT JOIN Zones z ON z.ZoneName = {zone_join_query('d')}
        WHERE d.Name = '{cooling_variable}' AND z.ZoneName IN ({','.join(cooled_zone_names)})
        UNION ALL
        SELECT
            r.TimeIndex AS timestep,
            'heating' AS load,
            z.ZoneIndex AS zone_index,
            z.ZoneName AS zone_name,
            ABS({heating_column('r')}) AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        LEFT JOIN Zones z ON z.ZoneName = {zone_join_query('d')}
        WHERE d.Name = '{heating_variable}' AND z.ZoneName IN ({','.join(heated_zone_names)})
        """
        data = self.__simulator.get_database().query_table(query)
        data = data.pivot(index=['zone_name','zone_index','timestep'],columns='load',values='value')
        data = data.reset_index(drop=False).to_dict(orient='list')
        data.pop('index',None)
        ideal_loads_data['load'] = deepcopy(data)

        # weighted average indoor dry-bulb temperature
        query = """
        SELECT
            r.TimeIndex AS timestep,
            SUM(r.Value) AS value
        FROM weighted_variable r
        LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        WHERE d.Name IN ('Zone Air Temperature')
        GROUP BY r.TimeIndex
        """
        self.__insert_zone_metadata(self.__simulator)
        data = self.__simulator.get_database().query_table(query).to_dict(orient='list')
        data.pop('index',None)
        ideal_loads_data['temperature'] = deepcopy(data)
        
        self.__ideal_loads_data = ideal_loads_data

    def __get_conditioned_zones(self):
        cooled_zones = [k for k,v in self.__zones.items() if v['is_cooled']==1]
        heated_zones = [k for k,v in self.__zones.items() if v['is_heated']==1]
        return cooled_zones, heated_zones

    def __insert_zone_metadata(self,simulator):
        data = pd.DataFrame([v for _,v in self.__zones.items()])
        query = """
        DROP TABLE IF EXISTS zone_metadata;
        CREATE TABLE IF NOT EXISTS zone_metadata (
            zone_index INTEGER PRIMARY KEY,
            zone_name TEXT,
            multiplier REAL,
            volume REAL,
            floor_area REAL,
            total_floor_area_proportion REAL,
            conditioned_floor_area_proportion REAL,
            is_cooled INTEGER,
            is_heated INTEGER,
            average_cooling_setpoint REAL,
            average_heating_setpoint REAL
        );

        DROP VIEW IF EXISTS weighted_variable;
        CREATE VIEW weighted_variable AS
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                r.Value*z.conditioned_floor_area_proportion AS Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            INNER JOIN (SELECT * FROM zone_metadata WHERE is_cooled + is_heated >= 1) z ON z.zone_name = d.KeyValue
            WHERE d.Name IN ('Zone Air Temperature')
        ;
        """
        simulator.get_database().query(query)
        simulator.get_database().insert('zone_metadata',data.columns.tolist(),data.values,)

    def __set_zones(self):
        query = """
        WITH zone_conditioning AS (
            SELECT
                z.ZoneIndex,
                SUM(CASE WHEN s.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND s.average_setpoint > 0 THEN 1 ELSE 0 END) AS is_cooled,
                SUM(CASE WHEN s.Name = 'Zone Thermostat Heating Setpoint Temperature' AND s.average_setpoint > 0 THEN 1 ELSE 0 END) AS is_heated,
                MAX(CASE WHEN s.Name = 'Zone Thermostat Cooling Setpoint Temperature' AND s.average_setpoint > 0 THEN s.average_setpoint 
                    ELSE NULL END) AS average_cooling_setpoint,
                MAX(CASE WHEN s.Name = 'Zone Thermostat Heating Setpoint Temperature' AND s.average_setpoint > 0 THEN s.average_setpoint 
                    ELSE NULL END) AS average_heating_setpoint
            FROM (
                SELECT
                    d.KeyValue,
                    d.Name,
                    AVG(r."value") AS average_setpoint
                FROM ReportData r
                INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
                WHERE d.Name IN ('Zone Thermostat Cooling Setpoint Temperature', 'Zone Thermostat Heating Setpoint Temperature')
                GROUP BY d.KeyValue, d.Name
            ) s
            LEFT JOIN Zones z ON z.ZoneName = s.KeyValue
            GROUP BY
                z.ZoneName,
                z.ZoneIndex
        )
        -- get zone floor area proportion of total zone floor area
        SELECT
            z.ZoneName AS zone_name,
            z.ZoneIndex AS zone_index,
            z.Multiplier AS multiplier,
            z.Volume AS volume,
            z.FloorArea AS floor_area,
            (z.FloorArea*z.Multiplier)/t.total_floor_area AS total_floor_area_proportion,
            CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN (z.FloorArea*z.Multiplier)/t.conditioned_floor_area ELSE 0 END AS conditioned_floor_area_proportion,
            c.is_cooled,
            c.is_heated,
            c.average_cooling_setpoint,
            c.average_heating_setpoint
        FROM Zones z
        CROSS JOIN (
            SELECT
                SUM(z.FloorArea*z.Multiplier) AS total_floor_area,
                SUM(CASE WHEN c.is_cooled != 0 OR c.is_heated != 0 THEN z.FloorArea*z.Multiplier ELSE 0 END) AS conditioned_floor_area
            FROM Zones z
            LEFT JOIN zone_conditioning c ON c.ZoneIndex = z.ZoneIndex
        ) t
        LEFT JOIN zone_conditioning c ON c.ZoneIndex = z.ZoneIndex
        """
        data = self.__simulator.get_database().query_table(query)
        self.__zones = {z['zone_name']:z for z in data.to_dict('records')}

    def __simulate_ideal_loads(self,**kwargs):
        found_objs = True

        while True:
            removed_objs = {}
            edited_objs = {}

            try:
                self.__simulator.simulate()
                break

            except EnergyPlusRunError as e:
                if self.ideal_loads_air_system and (self.__simulator.has_ems_input_error() or self.__simulator.has_ems_program_error()) and found_objs:
                    try:
                        removed_objs = self.__simulator.remove_ems_objs_in_error(patterns=kwargs.get('patterns',None))
                        
                        if self.edit_ems:
                            edited_objs = self.__simulator.redefine_ems_program_in_line_error()
                        else:
                            removed_objs = {**removed_objs,**self.__simulator.remove_ems_program_objs_in_line_error()}

                        found_objs = len(removed_objs) + len(edited_objs) > 0
                        LOGGER.debug(f'Removed objs: {removed_objs}')
                        LOGGER.debug(f'Edited objs: {edited_objs}')
                        LOGGER.debug('Rerunning sim.')
                    
                    except Exception as e:
                        LOGGER.exception(e)
                        raise e
                
                else:
                    LOGGER.exception(e)
                    raise e

    def __set_simulator(self):
        osm_editor = OpenStudioModelEditor(self.osm)

        if self.ideal_loads_air_system:
            osm_editor.use_ideal_loads_air_system()
        else:
            pass

        self.__simulator = Simulator(
            self.idd_filepath,
            osm_editor.forward_translate(),
            self.epw,
            simulation_id=self.__kwargs.get('simulation_id',None),
            output_directory=self.__kwargs.get('output_directory',None)
        )
        self.__preprocess_idf()
       
    def __preprocess_idf(self):
        # make output directory
        os.makedirs(self.__simulator.output_directory,exist_ok=True)

        # idf object
        idf = self.__simulator.get_idf_object()

        # change to hourly simulation
        idf.idfobjects['Timestep'] = []
        obj = idf.newidfobject('Timestep')
        obj.Number_of_Timesteps_per_Hour = self.timesteps_per_hour

        # update output variables
        idf.idfobjects['Output:Variable'] = []

        for output_variable in self.output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable
            obj.Reporting_Frequency = 'Timestep'

        # remove daylight savings definition
        idf.idfobjects['RunPeriodControl:DaylightSavingTime'] = []

        # set schedules filepath
        schedules_filepath = os.path.join(self.__simulator.output_directory,f'{self.__simulator.simulation_id}_schedules.csv')
        pd.DataFrame(self.schedules).to_csv(schedules_filepath,index=False)

        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name.lower() in self.schedules.keys():
                obj.File_Name = schedules_filepath
            else:
                continue
        
        # set ideal loads to satisfy solely sensible load
        if self.ideal_loads_air_system:
            for obj in idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem']:
                obj.Dehumidification_Control_Type = 'None'
                obj.Cooling_Sensible_Heat_Ratio = 1.0
        else:
            pass

        # set setpoints filepath
        if self.setpoints is not None:
            setpoints_filepath = os.path.join(self.__simulator.output_directory,f'{self.__simulator.simulation_id}_setpoint.csv')
            pd.DataFrame(self.setpoints).to_csv(setpoints_filepath,index=False)
            
            # put schedule obj
            obj = idf.newidfobject('Schedule:File')
            schedule_object_name = f'ecobee setpoint'
            obj.Name = schedule_object_name
            obj.Schedule_Type_Limits_Name = 'Temperature'
            obj.File_Name = setpoints_filepath
            obj.Column_Number = 1
            obj.Rows_to_Skip_at_Top = 1
            obj.Number_of_Hours_of_Data = 8760
            obj.Minutes_per_Item = 60

            for obj in idf.idfobjects['ThermostatSetpoint:DualSetpoint']:
                obj.Cooling_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'
                obj.Heating_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'
        else:
            pass

        self.__simulator.idf = idf.idfstr()
    
    @staticmethod
    def initialize_database(database):
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
        CREATE TABLE IF NOT EXISTS energyplus_mechanical_system_simulation (
            simulation_id INTEGER NOT NULL,
            timestep INTEGER NOT NULL,
            average_indoor_air_temperature REAL NOT NULL,
            PRIMARY KEY (simulation_id, timestep),
            FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        );
        CREATE TABLE IF NOT EXISTS energyplus_ideal_system_simulation (
            simulation_id INTEGER NOT NULL,
            timestep INTEGER NOT NULL,
            average_indoor_air_temperature REAL NOT NULL,
            cooling_load REAL NOT NULL,
            heating_load REAL NOT NULL,
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
            diffuse_solar_radiation REAL NOT NULL,
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
    