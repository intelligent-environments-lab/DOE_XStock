import concurrent.futures
from copy import deepcopy
import logging
import logging.config
import os
from pathlib import Path
from eppy.bunch_subclass import BadEPFieldError
import numpy as np
import pandas as pd
from doe_xstock.simulate import OpenStudioModelEditor, Simulator
from doe_xstock.utilities import read_json, write_data

logging_config = read_json(os.path.join(os.path.dirname(__file__),Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class TrainData:
    def __init__(self,idd_filepath,osm,epw,schedules,setpoints,output_variables=None,timesteps_per_hour=None,iterations=None,max_workers=None,seed=None,**kwargs):
        self.idd_filepath = idd_filepath
        self.osm = osm
        self.epw = epw
        self.schedules = schedules
        self.setpoints = setpoints
        self.output_variables = output_variables
        self.timesteps_per_hour = timesteps_per_hour
        self.iterations = iterations
        self.max_workers = max_workers
        self.seed = seed
        self.__kwargs = kwargs
        self.__simulator = None
        self.__partial_loads_data = None
        self.__errors = []

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
    def errors(self):
        return self.__errors

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

    @output_variables.setter
    def output_variables(self,output_variables):
        default_output_variables = [
            'Electric Equipment Electricity Energy',
            'Exterior Lights Electricity Energy',
            'Lights Electricity Energy',
            'Other Equipment Convective Heating Energy',
            'Other Equipment Convective Heating Rate',
            'Site Diffuse Solar Radiation Rate per Area', 
            'Site Direct Solar Radiation Rate per Area',
            'Site Outdoor Air Drybulb Temperature',
            'Site Outdoor Air Relative Humidity',
            'Water Heater Use Side Heat Transfer Energy', 
            'Zone Air Relative Humidity',
            'Zone Air System Sensible Cooling Energy',
            'Zone Air System Sensible Heating Energy',
            'Zone Air System Sensible Cooling Rate',
            'Zone Air System Sensible Heating Rate',
            'Zone Air Temperature',
            'Zone Ideal Loads Zone Sensible Cooling Energy',
            'Zone Ideal Loads Zone Sensible Heating Energy',
            'Zone Ideal Loads Zone Sensible Cooling Rate',
            'Zone Ideal Loads Zone Sensible Heating Rate',
            'Zone People Occupant Count',
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Energy',
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate',
            'Zone Thermostat Cooling Setpoint Temperature',
            'Zone Thermostat Heating Setpoint Temperature',
        ]
        self.__output_variables = default_output_variables if output_variables is None else output_variables

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

    def simulate_partial_loads(self):
        LOGGER.info('Started simulation.')
        self.__set_simulator()
        seeds = [i for i in range(self.iterations + 3)]
        simulators = [self.__get_partial_load_simulator(i, seed=s) for i, s in enumerate(seeds)]
        LOGGER.debug('Simulating partial load iterations.')
        Simulator.multi_simulate(simulators, max_workers=self.max_workers)
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
        return self.__partial_loads_data

    def __post_process_partial_load_simulation(self,simulator):            
        # create zone conditioning table
        zones = self.set_zones(simulator)
        self.__insert_zone_metadata(simulator, zones)
        cooled_zone_names = [f'\'{k}\'' for k,v in zones.items() if v['is_cooled']==1]
        cooled_zone_names = ','.join(cooled_zone_names)

        # get simulation summary
        query = f"""
        WITH u AS (
            -- site variables
            SELECT
                r.TimeIndex,
                r.ReportDataDictionaryIndex,
                'site_variable' AS label,
                r.Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            WHERE d.Name IN (
                'Site Direct Solar Radiation Rate per Area', 
                'Site Diffuse Solar Radiation Rate per Area', 
                'Site Outdoor Air Drybulb Temperature'
            )

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
                'thermal_load' AS label,
                r.Value
            FROM ReportData r
            LEFT JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
            LEFT JOIN Zones z ON z.ZoneName = d.KeyValue
            WHERE 
                d.Name IN (
                    'Zone Air System Sensible Cooling Rate', 
                    'Zone Air System Sensible Heating Rate', 
                    'Zone Thermostat Cooling Setpoint Temperature'
                )
                AND z.ZoneName IN ({cooled_zone_names})

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
                SUM(CASE WHEN d.Name = 'Zone Air System Sensible Cooling Rate' THEN Value END) AS cooling_load,
                SUM(CASE WHEN d.Name = 'Zone Air System Sensible Heating Rate' THEN Value END) AS heating_load,
                MIN(CASE WHEN d.Name = 'Zone Thermostat Cooling Setpoint Temperature' THEN Value END) AS setpoint
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
            COALESCE(p.heating_load, 0) AS heating_load,
            p.setpoint
        FROM p
        LEFT JOIN Time t ON t.TimeIndex = p.TimeIndex
        WHERE t.DayType NOT IN ('SummerDesignDay', 'WinterDesignDay')
        """
        data = simulator.get_database().query_table(query)
        data = data.to_dict('list')
        _ = data.pop('index',None)
        
        return simulator.simulation_id, data

    def __get_partial_load_simulator(self,reference,seed=None):
        idf = self.__simulator.get_idf_object()
        simulation_id = f'{self.kwargs["simulation_id"]}-{reference}-partial'
        output_directory = os.path.join(self.kwargs['output_directory'], simulation_id)
        os.makedirs(output_directory, exist_ok=True)
        
        # update setpoints
        setpoints_filepath = os.path.join(output_directory, f'{simulation_id}_setpoint.csv')
        setpoints = pd.DataFrame(self.setpoints)
        size = setpoints.shape[0]

        if seed > 1:
            setpoints['addition'] = self.get_addition(size, seed=seed)
            setpoints['setpoint'] = setpoints['setpoint'] + setpoints['addition']
            setpoints = setpoints.drop(columns=['addition'])

        elif seed == 1:
            idf = self.turn_off_equipment(idf)
        
        else:
            pass
        
        # insert setpoint schedule object
        obj = idf.newidfobject('Schedule:File')
        schedule_object_name = f'ecobee setpoint'
        obj.Name = schedule_object_name
        obj.Schedule_Type_Limits_Name = 'Temperature'
        obj.File_Name = setpoints_filepath
        obj.Column_Number = 1
        obj.Rows_to_Skip_at_Top = 1
        obj.Number_of_Hours_of_Data = len(setpoints)
        obj.Minutes_per_Item = 60

        for obj in idf.idfobjects['ThermostatSetpoint:DualSetpoint']:
            obj.Cooling_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'
            obj.Heating_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'

        setpoints.to_csv(setpoints_filepath, index=False)
        idf = idf.idfstr()

        return Simulator(self.idd_filepath,idf,self.epw,simulation_id=simulation_id,output_directory=output_directory)
    
    def turn_off_equipment(self, idf):
        hvac_equipment_objects = [n.upper() for n in [
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
            'AirLoopHVAC:UnitarySystem','ZoneHVAC:IdealLoadsAirSystem','HVACTemplate:Zone:IdealLoadsAirSystem',
            'Fan:OnOff', 'Coil:Cooling:DX:SingleSpeed', 'Coil:Heating:Electric', 'Coil:Heating:DX:SingleSpeed',
            'AirLoopHVAC:UnitarySystem', 'AirTerminal:SingleDuct:ConstantVolume:NoReheat'
        ]]

        # set hvac equipment availability to always off
        schedule_name = 'Always Off Discrete'

        for name in hvac_equipment_objects:
            if name in idf.idfobjects.keys():
                for obj in idf.idfobjects[name]:
                    try:
                        obj.Availability_Schedule_Name = schedule_name
                    except BadEPFieldError:
                        obj.System_Availability_Schedule_Name = schedule_name
            else:
                continue

        return idf

    def get_addition(self, size, seed, minimum_value=-5.0, maximum_value=5.0, probability=0.85):
        np.random.seed(seed)
        schedule = np.random.uniform(minimum_value, maximum_value, size)
        schedule[np.random.random(size) > probability] = 0.0

        return schedule

    def __insert_zone_metadata(self, simulator, zones):
        data = pd.DataFrame([v for _,v in zones.items()])
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
            WHERE d.Name IN ('Zone Air Temperature', 'Zone Air Relative Humidity')
        ;
        """
        simulator.get_database().query(query)
        simulator.get_database().insert('zone_metadata',data.columns.tolist(),data.values,)

    def set_zones(self, simulator):
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
        data = simulator.get_database().query_table(query)
        zones = {z['zone_name']:z for z in data.to_dict('records')}

        return zones

    def __set_simulator(self):
        osm_editor = OpenStudioModelEditor(self.osm)
        idf = osm_editor.forward_translate()

        try:
            assert\
                'OS:AirLoopHVAC:UnitarySystem' in self.osm\
                    and 'ZoneControl:Thermostat,' in idf\
                        and 'ThermostatSetpoint:DualSetpoint,' in idf
        except:
            self.__errors.append(1)
            raise EnergyPlusSimulationError

        # write_data(self.osm, 'test.osm')
        # write_data(osm_editor.forward_translate(), 'test.idf')
        # assert False

        self.__simulator = Simulator(
            self.idd_filepath,
            idf,
            self.epw,
            simulation_id=self.__kwargs.get('simulation_id',None),
            output_directory=self.__kwargs.get('output_directory',None)
        )
        self.__preprocess_idf()
       
    def __preprocess_idf(self):
        # idf object
        idf = self.__simulator.get_idf_object()

        # update simulation time step
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

        # insert schedule object
        output_directory = self.kwargs['output_directory']
        os.makedirs(output_directory, exist_ok=True)
        schedules_filepath = os.path.join(output_directory, f'{self.kwargs["simulation_id"]}_schedules.csv')
        pd.DataFrame(self.schedules).to_csv(schedules_filepath, index=False)

        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name.lower() in self.schedules.keys():
                obj.File_Name = schedules_filepath
            else:
                continue

        self.__simulator.idf = idf.idfstr()

    @staticmethod
    def get_train_data(database, metadata_id):
        return database.query_table(f"""
        SELECT
            m.in_resstock_county_id AS location,
            m.bldg_id AS resstock_building_id,
            b.name AS ecobee_building_id,
            s.reference AS simulation_reference,
            l.timestep,
            l.month,
            l.day,
            l.day_of_week,
            l.hour,
            l.minute,
            l.direct_solar_radiation,
            l.diffuse_solar_radiation,
            l.outdoor_air_temperature,
            l.average_indoor_air_temperature,
            l.occupant_count,
            l.cooling_load,
            l.heating_load,
            l.setpoint
        FROM lstm_train_data l
        LEFT JOIN energyplus_simulation s ON
            s.id = l.simulation_id
        LEFT JOIN ecobee_building b ON b.id = s.ecobee_building_id
        LEFT JOIN metadata m ON m.id = s.metadata_id
        WHERE l.simulation_id IN (
            SELECT id FROM energyplus_simulation WHERE metadata_id = {metadata_id}
        )""")
    
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
            setpoint REAL NOT NULL,
            PRIMARY KEY (simulation_id, timestep),
            FOREIGN KEY (simulation_id) REFERENCES energyplus_simulation (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE
        );
        CREATE INDEX IF NOT EXISTS lstm_train_data_simulation_id ON lstm_train_data(simulation_id);
        CREATE TABLE IF NOT EXISTS energyplus_simulation_error_description (
            id INTEGER NOT NULL,
            description TEXT NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (description)
        );
        INSERT OR IGNORE INTO energyplus_simulation_error_description (id, description)
        VALUES 
            (1, 'forward translate error: air loop not found in osm and thermostat not found in idf')
        ;
        CREATE TABLE IF NOT EXISTS energyplus_simulation_error (
            id INTEGER NOT NULL,
            metadata_id INTEGER NOT NULL,
            description_id INTEGER NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY (metadata_id) REFERENCES metadata (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            FOREIGN KEY (description_id) REFERENCES energyplus_simulation_error_description (id)
                ON DELETE NO ACTION
                ON UPDATE CASCADE,
            UNIQUE (metadata_id, description_id)
        );
        """)

class Error(Exception):
    """Base class for other exceptions."""

class EnergyPlusSimulationError(Error):
    __MESSAGE = 'Simulation errors were found.'
  
    def __init__(self,message=None):
        super().__init__(self.__MESSAGE if message is None else message)
    