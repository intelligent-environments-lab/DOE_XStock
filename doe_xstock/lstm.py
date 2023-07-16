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
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class TrainData:
    def __init__(
            self,idd_filepath,osm,epw,schedules,setpoints,run_period_begin_month=None,
            run_period_begin_day_of_month=None,run_period_begin_year=None,run_period_end_month=None,
            run_period_end_day_of_month=None,run_period_end_year=None,output_variables=None,
            timesteps_per_hour=None,iterations=None,max_workers=None,seed=None,**kwargs
        ):
        self.idd_filepath = idd_filepath
        self.osm = osm
        self.epw = epw
        self.schedules = schedules
        self.setpoints = setpoints
        self.run_period_begin_month = run_period_begin_month
        self.run_period_begin_day_of_month = run_period_begin_day_of_month
        self.run_period_begin_year = run_period_begin_year
        self.run_period_end_month = run_period_end_month
        self.run_period_end_day_of_month = run_period_end_day_of_month
        self.run_period_end_year = run_period_end_year
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
        default_output_variables = []

        #  weather_variables
        default_output_variables += [
            'Site Diffuse Solar Radiation Rate per Area', 
            'Site Direct Solar Radiation Rate per Area',
            'Site Outdoor Air Drybulb Temperature',
            'Site Outdoor Air Relative Humidity',
        ]

        # electric_equipment_variables
        default_output_variables += [
            'Zone Electric Equipment Electricity Rate',
            'Zone Lights Electricity Rate',
        ]

        # mechanical_hvac_variables
        default_output_variables += [
            'Air System Total Cooling Energy',
            'Air System Total Heating Energy',
            'Zone Air System Sensible Cooling Rate',
            'Zone Air System Sensible Heating Rate',
        ]

        # ideal_load_variables
        default_output_variables += [
            'Zone Ideal Loads Zone Sensible Cooling Rate',
            'Zone Ideal Loads Zone Sensible Heating Rate',
        ]

        # other_equipment_variables
        default_output_variables += [
            'Other Equipment Convective Heating Rate',
        ]

        # dhw_variables
        default_output_variables += [
            'Water Use Equipment Heating Rate',
        ]

        # ieq_variables
        default_output_variables += [
            'Zone Air Relative Humidity',
            'Zone Air Temperature',
            'Zone Thermostat Cooling Setpoint Temperature',
            'Zone Thermostat Heating Setpoint Temperature',
        ]

        # occupancy_variables
        default_output_variables += [
            'Zone People Occupant Count',
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
        conditioned_zone_names = [f'\'{k}\'' for k,v in zones.items() if v['is_cooled']==1 or v['is_heated']==1]
        conditioned_zone_names = ','.join(conditioned_zone_names)

        # get simulation summary
        query_filepath = os.path.join(os.path.dirname(__file__),'misc/queries/set_lstm_train_data.sql')
        data = simulator.get_database().query_table_from_file(query_filepath, replace={'<conditioned_zone_names>': conditioned_zone_names})
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
            'AirTerminal:SingleDuct:Uncontrolled','Fan:ZoneExhaust','ZoneHVAC:Baseboard:Convective:Electric','ZoneHVAC:Baseboard:Convective:Water',
            'ZoneHVAC:Baseboard:RadiantConvective:Electric','ZoneHVAC:Baseboard:RadiantConvective:Water',
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
        query_filepath = os.path.join(os.path.dirname(__file__),'misc/queries/set_lstm_zone_metadata.sql')
        simulator.get_database().execute_sql_from_file(query_filepath)
        simulator.get_database().insert('zone_metadata',data.columns.tolist(),data.values,)

    def set_zones(self, simulator):
        query_filepath = os.path.join(os.path.dirname(__file__),'misc/queries/get_lstm_zone_conditioning.sql')
        data = simulator.get_database().query_table_from_file(query_filepath)
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

        # update run period
        obj = idf.idfobjects['RunPeriod'][0]
        obj.Begin_Month = obj.Begin_Month if self.run_period_begin_month is None else self.run_period_begin_month
        obj.Begin_Day_of_Month = obj.Begin_Day_of_Month if self.run_period_begin_day_of_month is None else self.run_period_begin_day_of_month
        obj.Begin_Year = obj.Begin_Year if self.run_period_begin_year is None else self.run_period_begin_year
        obj.End_Month = obj.End_Month if self.run_period_end_month is None else self.run_period_end_month
        obj.End_Day_of_Month = obj.End_Day_of_Month if self.run_period_end_day_of_month is None else self.run_period_end_day_of_month
        obj.End_Year = obj.End_Year if self.run_period_end_year is None else self.run_period_end_year

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
        query_filepath = os.path.join(os.path.dirname(__file__),'misc/queries/get_lstm_train_data.sql')
        data = database.query_table_from_file(query_filepath, replace={'<metadata_id>': metadata_id})

        return data
    
    @staticmethod
    def initialize_database(database):
        query_filepath = os.path.join(os.path.dirname(__file__),'misc/queries/set_lstm_tables_and_views.sql')
        database.execute_sql_from_file(query_filepath)

class Error(Exception):
    """Base class for other exceptions."""

class EnergyPlusSimulationError(Error):
    __MESSAGE = 'Simulation errors were found.'
  
    def __init__(self,message=None):
        super().__init__(self.__MESSAGE if message is None else message)
    