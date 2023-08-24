import concurrent.futures
from copy import deepcopy
import logging
import logging.config
import os
from pathlib import Path
from eppy.bunch_subclass import BadEPFieldError
from eppy.runner.run_functions import EnergyPlusRunError
import numpy as np
import pandas as pd
from doe_xstock.simulate import OpenStudioModelEditor, Simulator
from doe_xstock.utilities import read_json, write_data

logging_config = read_json(os.path.join(os.path.dirname(__file__), Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class TrainData:
    def __init__(
            self, idd_filepath, osm, epw, schedules, setpoints, output_directory, simulation_id, ideal_loads_air_system=None, edit_ems=None, 
            run_period_begin_month=None, run_period_begin_day_of_month=None, run_period_begin_year=None, run_period_end_month=None, 
            run_period_end_day_of_month=None, run_period_end_year=None, output_variables=None, 
            timesteps_per_hour=None, iterations=None, max_workers=None, seed=None,
        ):
        self.idd_filepath = idd_filepath
        self.output_directory = output_directory
        self.simulation_id = simulation_id
        self.osm = osm
        self.epw = epw
        self.__setpoints_filepath = None
        self.__schedules_filepath = None
        self.schedules = schedules
        self.setpoints = setpoints
        self.ideal_loads_air_system = ideal_loads_air_system
        self.edit_ems = edit_ems
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
        self.__simulator = {}
        self.__design_loads_data = {}
        self.__zone_metadata = {}
        self.__partial_loads_data = None
        self.__design_simulation_references = {
            'mechanical': 0,
            'ideal': 1
        }

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
    def output_directory(self):
        return self.__output_directory

    @idd_filepath.setter
    def idd_filepath(self, idd_filepath):
        self.__idd_filepath = idd_filepath

    @osm.setter
    def osm(self, osm):
        write_data(osm, os.path.join(self.output_directory, f'{self.simulation_id}.osm'))
        self.__osm = self.__validate_osm(osm)
        
    @epw.setter
    def epw(self, epw):
        self.__epw = epw

    @schedules.setter
    def schedules(self, schedules):
        self.__schedules = schedules
        self.__schedules_filepath = os.path.join(self.output_directory, f'schedules.csv')
        pd.DataFrame(self.schedules).to_csv(self.__schedules_filepath, index=False)

    @setpoints.setter
    def setpoints(self, setpoints):
        self.__setpoints = setpoints

        if self.setpoints:
            self.__setpoints_filepath = os.path.join(self.output_directory, f'setpoint.csv')
            pd.DataFrame(self.setpoints).to_csv(self.__setpoints_filepath, index=False)
        else:
            pass

    @ideal_loads_air_system.setter
    def ideal_loads_air_system(self, ideal_loads_air_system):
        self.__ideal_loads_air_system = False if ideal_loads_air_system is None else ideal_loads_air_system
    
    @edit_ems.setter
    def edit_ems(self, edit_ems):
        self.__edit_ems = True if edit_ems is None else edit_ems

    @output_variables.setter
    def output_variables(self, output_variables):
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
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate',
        ]

        # ideal_load_variables
        default_output_variables += [
            'Zone Ideal Loads Zone Sensible Cooling Rate', 
            'Zone Ideal Loads Zone Sensible Heating Rate', 
        ]

        # other_equipment_variables
        default_output_variables += [
            'Other Equipment Convective Heating Rate', 
            'Other Equipment Convective Heating Energy'
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
    def timesteps_per_hour(self, timesteps_per_hour):
        self.__timesteps_per_hour = timesteps_per_hour = 1 if timesteps_per_hour is None else timesteps_per_hour

    @iterations.setter
    def iterations(self, iterations):
        self.__iterations = 3 if iterations is None else iterations

    @max_workers.setter
    def max_workers(self, max_workers):
        self.__max_workers = 1 if max_workers is None else max_workers

    @seed.setter
    def seed(self, seed):
        self.__seed = 0 if seed is None else seed

    @output_directory.setter
    def output_directory(self, output_directory):
        self.__output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

    def run(self, ems_patterns=None):
        LOGGER.info('Started simulation run.')
        
        # run for design mechanical and ideal load
        for k in self.__design_simulation_references:
            LOGGER.debug(f'Simulating {k} load.')
            self.set_design_loads_data(mode=k, patterns=ems_patterns)
            LOGGER.debug(f'Ended simulating {k}.')

        # run for partial load
        LOGGER.debug('Simulating partial load iterations.')
        seeds = [None] + [i for i in range(self.iterations + 1)]
        partial_references = [i + self.__design_simulation_references['ideal'] + 1 for i, _ in enumerate(seeds)]
        simulators = [
            self.__get_partial_load_simulator(r, seed=s) 
            for r, s in zip(partial_references, seeds)
        ]
        simulation_ids = [s.simulation_id for s in simulators]
        Simulator.multi_simulate(simulators, max_workers=self.max_workers)
        LOGGER.debug('Ended simulating partial load iterations.')
        self.__partial_loads_data = {}

        LOGGER.debug('Post processing partial load iterations.')
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = [executor.submit(self.post_process_partial_load_simulation, *[s]) for s in simulators]

            for _, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    r = future.result()
                    reference = partial_references[simulation_ids.index(r[0])]
                    self.__partial_loads_data[reference] = deepcopy(r[1])
                    LOGGER.debug(f'Finished processing simulaton_id:{r[0]}')
                except Exception as e:
                    LOGGER.exception(e)
        
        LOGGER.info('Ended simulation run.')

        return self.__design_loads_data, self.__partial_loads_data

    def post_process_partial_load_simulation(self, simulator):
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_air_system_load.sql')
        data = simulator.get_database().query_table_from_file(query_filepath)
        air_system_cooling = data[data['name']=='Zone Air System Sensible Cooling Rate']['value'].iloc[0]
        air_system_heating = data[data['name']=='Zone Air System Sensible Heating Rate']['value'].iloc[0]

        if air_system_cooling != 0 or air_system_heating != 0:
            message = f'Non-zero Zone Air System Sensible Cooling Rate and/or'\
                f' Heating Rate: ({air_system_cooling, air_system_heating})'
            raise EnergyPlusSimulationError(error_id=4, message=message)
        else:
            pass
                  
        # create zone conditioning table
        zones = self.__zone_metadata[self.ideal_loads_air_system]
        TrainData.__insert_zone_metadata(simulator, zones)
        conditioned_zone_names = [f'\'{k}\'' for k, v in zones.items() if v['is_cooled']==1 or v['is_heated']==1]
        conditioned_zone_names = ', '.join(conditioned_zone_names)

        # get simulation summary
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/set_lstm_train_data.sql')
        data = simulator.get_database().query_table_from_file(query_filepath, replace={'<conditioned_zone_names>': conditioned_zone_names})
        data = {**data.to_dict('list'), **self.setpoints}
        _ = data.pop('index', None)
        
        return simulator.simulation_id, data

    def __get_partial_load_simulator(self, reference, seed=None):
        idf = self.__get_transformed_idf_for_partial_load_simulation()
        simulation_id = f'{self.simulation_id}-{reference}-partial'
        output_directory = os.path.join(self.output_directory, simulation_id)
        os.makedirs(output_directory, exist_ok=True)

        # get multiplier
        size = len(self.__design_loads_data[int(self.ideal_loads_air_system)]['load']['timestep'])

        if seed is None:
            multiplier = [1.0]*size
        elif seed == 0:
            multiplier = [0.0]*size
        else:
            multiplier = TrainData.get_multipliers(size, seed=seed)
        
        multiplier = pd.DataFrame(multiplier, columns=['multiplier'])
        multiplier['timestep'] = multiplier.index + 1

        # set load schedule file
        data = pd.DataFrame(self.__design_loads_data[int(self.ideal_loads_air_system)]['load'])
        data = data.merge(multiplier, on='timestep', how='left')
        data = data.sort_values(['zone_name', 'timestep'])
        data['cooling_load'] *= data['multiplier']*-1
        data['heating_load'] *= data['multiplier']
        filepath = os.path.join(output_directory, f'{simulation_id}_partial_load.csv')
        data[['cooling_load', 'heating_load']].to_csv(filepath, index=False)

        # set load schedule
        for obj in idf.idfobjects['Schedule:File']:
            if 'lstm' in obj.Name.lower():
                obj.File_Name = filepath
            else:
                continue
        
        # convert idf to string
        idf = idf.idfstr()
        idf = idf.replace(self.__simulator[self.ideal_loads_air_system].simulation_id, simulation_id)

        return Simulator(self.idd_filepath, idf, self.epw, simulation_id=simulation_id, output_directory=output_directory)
    
    def __get_transformed_idf_for_partial_load_simulation(self):
        idf = self.__simulator[self.ideal_loads_air_system].get_idf_object()
        
        if self.ideal_loads_air_system:
            self.__remove_ideal_loads_air_system(idf)
        else:
            self.__turn_off_equipment(idf)

        # schedule type limit object
        schedule_type_limit_name = 'other equipment hvac power'
        obj = idf.newidfobject('ScheduleTypeLimits')
        obj.Name = schedule_type_limit_name
        obj.Lower_Limit_Value = ''
        obj.Upper_Limit_Value = ''
        obj.Numeric_Type = 'Continuous'
        obj.Unit_Type = 'Dimensionless'

        # generate stochastic thermal load
        zone_names = set(self.__design_loads_data[int(self.ideal_loads_air_system)]['load']['zone_name'])
        zone_names = sorted(list(zone_names))
        timesteps = max(self.__design_loads_data[int(self.ideal_loads_air_system)]['load']['timestep'])
        loads = ['cooling_load', 'heating_load']
        
        for i, zone_name in enumerate(zone_names):
            for j, load in enumerate(loads):
                # put schedule obj
                obj = idf.newidfobject('Schedule:File')
                schedule_object_name = f'{zone_name} lstm {load}'
                obj.Name = schedule_object_name
                obj.Schedule_Type_Limits_Name = schedule_type_limit_name
                obj.File_Name = ''
                obj.Column_Number = j + 1
                obj.Rows_to_Skip_at_Top = 1 + i*timesteps
                obj.Number_of_Hours_of_Data = 8760
                obj.Minutes_per_Item = int(60/self.timesteps_per_hour)

                # put other equipment
                obj = idf.newidfobject('OtherEquipment')
                obj.Name = f'{zone_name} {load}'
                obj.Fuel_Type = 'None'
                obj.Zone_or_ZoneList_or_Space_or_SpaceList_Name = zone_name
                obj.Schedule_Name = schedule_object_name
                obj.Design_Level_Calculation_Method = 'EquipmentLevel'
                obj.Design_Level = 1000.0 # to covert to Watts
                obj.Fraction_Latent = 0.0
                obj.Fraction_Radiant = 0.0
                obj.Fraction_Lost = 0.0
                obj.EndUse_Subcategory = f'lstm {load}'

        return idf
    
    def __remove_ideal_loads_air_system(self, idf):
        idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem'] = []
        obj_names = [
            'ZoneControl:Thermostat', 'ZoneControl:Humidistat', 'ZoneControl:Thermostat:ThermalComfort', 'ThermostatSetpoint:DualSetpoint', 
            'ZoneControl:Thermostat:OperativeTemperature', 'ZoneControl:Thermostat:TemperatureAndHumidity', 'ZoneControl:Thermostat:StagedDualSetpoint'
        ]

        for name in obj_names:
            idf.idfobjects[name] = []

        return idf

    def __turn_off_equipment(self, idf):
        hvac_equipment_objects = [n.upper() for n in [
            'AirTerminal:SingleDuct:Uncontrolled', 'Fan:ZoneExhaust', 'ZoneHVAC:Baseboard:Convective:Electric', 'ZoneHVAC:Baseboard:Convective:Water', 
            'ZoneHVAC:Baseboard:RadiantConvective:Electric', 'ZoneHVAC:Baseboard:RadiantConvective:Water', 
            'ZoneHVAC:Baseboard:RadiantConvective:Steam', 'ZoneHVAC:Dehumidifier:DX', 'ZoneHVAC:EnergyRecoveryVentilator', 'ZoneHVAC:FourPipeFanCoil', 
            'ZoneHVAC:HighTemperatureRadiant', 'ZoneHVAC:LowTemperatureRadiant:ConstantFlow', 'ZoneHVAC:LowTemperatureRadiant:Electric', 
            'ZoneHVAC:LowTemperatureRadiant:VariableFlow', 'ZoneHVAC:OutdoorAirUnit', 'ZoneHVAC:PackagedTerminalAirConditioner', 
            'ZoneHVAC:PackagedTerminalHeatPump', 'ZoneHVAC:RefrigerationChillerSet', 'ZoneHVAC:UnitHeater', 'ZoneHVAC:UnitVentilator', 
            'ZoneHVAC:WindowAirConditioner', 'ZoneHVAC:WaterToAirHeatPump', 'ZoneHVAC:VentilatedSlab', 
            'AirTerminal:DualDuct:ConstantVolume', 'AirTerminal:DualDuct:VAV', 'AirTerminal:DualDuct:VAV:OutdoorAir', 
            'AirTerminal:SingleDuct:ConstantVolume:Reheat', 'AirTerminal:SingleDuct:VAV:Reheat', 'AirTerminal:SingleDuct:VAV:NoReheat', 
            'AirTerminal:SingleDuct:SeriesPIU:Reheat', 'AirTerminal:SingleDuct:ParallelPIU:Reheat'
            'AirTerminal:SingleDuct:ConstantVolume:FourPipeInduction', 'AirTerminal:SingleDuct:VAV:Reheat:VariableSpeedFan', 
            'AirTerminal:SingleDuct:VAV:HeatAndCool:Reheat', 'AirTerminal:SingleDuct:VAV:HeatAndCool:NoReheat', 
            'AirLoopHVAC:UnitarySystem', 'ZoneHVAC:IdealLoadsAirSystem', 'HVACTemplate:Zone:IdealLoadsAirSystem', 
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
    
    @staticmethod
    def get_multipliers(size, seed=None, minimum_value=None, maximum_value=None, probability=None):
        seed = 0 if seed is None else seed
        minimum_value = 0.3 if minimum_value is None else minimum_value
        maximum_value = 1.7 if maximum_value is None else maximum_value
        probability = 0.85 if probability is None else probability
        np.random.seed(seed)
        schedule = np.random.uniform(minimum_value, maximum_value, size)
        schedule[np.random.random(size) > probability] = 1.0
        schedule = schedule.tolist()
        return schedule
    
    def set_design_loads_data(self, mode, patterns=None):
        self.ideal_loads_air_system = bool(self.__design_simulation_references[mode])
        LOGGER.debug(f'Simulating HVAC loads for ideal loads = {self.ideal_loads_air_system}.')
        simulation_id = f'{self.simulation_id}-{self.__design_simulation_references[mode]}-{mode}'
        output_directory = os.path.join(self.output_directory, simulation_id)
        os.makedirs(output_directory, exist_ok=True)

        try:
            self.set_simulator(simulation_id, output_directory)
        except Exception as e:
            LOGGER.exception(e)
            raise e

        self.__simulate_design_loads(patterns=patterns)
        self.__set_design_loads_data()
        LOGGER.debug(f'Finished simulating HVAC loads for ideal loads = {self.ideal_loads_air_system}.')

        return self.__design_loads_data
    
    def __set_design_loads_data(self):
        design_loads_data = {}

        # cooling and heating loads
        zones = TrainData.get_zone_conditioning_metadata(self.__simulator[self.ideal_loads_air_system])
        TrainData.__insert_zone_metadata(self.__simulator[self.ideal_loads_air_system], zones)
        cooled_zone_names = [f'\'{k}\'' for k, v in zones.items() if v['is_cooled']==1]
        heated_zone_names = [f'\'{k}\'' for k, v in zones.items() if v['is_heated']==1]
        cooled_zone_names = ', '.join(cooled_zone_names)
        heated_zone_names = ', '.join(heated_zone_names)
        self.__zone_metadata[self.ideal_loads_air_system] = zones
        
        if self.ideal_loads_air_system:
            query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_design_load_loads_for_ideal_hvac.sql')
        else:
            query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_design_load_loads_for_mechanical_hvac.sql')

        data = self.__simulator[self.ideal_loads_air_system].get_database().query_table_from_file(query_filepath, replace={
            '<cooled_zone_names>': cooled_zone_names, 
            '<heated_zone_names>': heated_zone_names, 
        })
        data = data.pivot(index=['zone_name', 'zone_index', 'timestep'], columns='load', values='value')
        data = data.reset_index(drop=False).to_dict(orient='list')
        data.pop('index', None)
        design_loads_data['load'] = deepcopy(data)

        # weighted average indoor dry-bulb temperature
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_design_load_temperature.sql')
        data = self.__simulator[self.ideal_loads_air_system].get_database().query_table_from_file(query_filepath)
        data = data.to_dict(orient='list')
        data.pop('index', None)
        design_loads_data['temperature'] = deepcopy(data)
        
        self.__design_loads_data[int(self.ideal_loads_air_system)] = design_loads_data
    
    def __simulate_design_loads(self, patterns=None):
        found_objs = True

        while True:
            removed_objs = {}
            edited_objs = {}

            try:
                self.__simulator[self.ideal_loads_air_system].simulate()
                break

            except EnergyPlusRunError as e:
                if self.ideal_loads_air_system and (
                    self.__simulator[self.ideal_loads_air_system].has_ems_input_error() 
                    or self.__simulator[self.ideal_loads_air_system].has_ems_program_error()
                ) and found_objs:
                    try:
                        removed_objs = self.__simulator[self.ideal_loads_air_system].remove_ems_objs_in_error(
                            patterns=patterns
                        )
                        
                        if self.edit_ems:
                            edited_objs = self.__simulator[self.ideal_loads_air_system].redefine_ems_program_in_line_error()
                        else:
                            removed_objs = {
                                **removed_objs, 
                                **self.__simulator[self.ideal_loads_air_system].remove_ems_program_objs_in_line_error()
                            }

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

    @classmethod
    def __insert_zone_metadata(cls, simulator, zones):
        data = pd.DataFrame([v for _, v in zones.items()])
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/set_lstm_zone_metadata.sql')
        simulator.get_database().execute_sql_from_file(query_filepath)
        simulator.get_database().insert('zone_metadata', data.columns.tolist(), data.values)

    @classmethod
    def get_zone_conditioning_metadata(cls, simulator):
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_lstm_zone_conditioning.sql')
        data = simulator.get_database().query_table_from_file(query_filepath)
        zones = {z['zone_name']:z for z in data.to_dict('records')}

        return zones

    def set_simulator(self, simulation_id, output_directory):
        # set osm editor
        osm_editor = OpenStudioModelEditor(self.osm)
        
        if self.ideal_loads_air_system:
            osm_editor.use_ideal_loads_air_system()
        else:
            pass
        
        # set idf
        idf = osm_editor.forward_translate()
        idf = self.__validate_idf(idf)
        self.__simulator[self.ideal_loads_air_system] = Simulator(
            self.idd_filepath, 
            idf, 
            self.epw,
            simulation_id=simulation_id, 
            output_directory=output_directory
        )
        self.__preprocess_idf()

    def __validate_idf(self, idf):
        if not 'ZoneControl:Thermostat,' in idf:
            write_data(idf, 'test.idf')
            raise EnergyPlusSimulationError(error_id=3, message='ZoneControl:Thermostat not found in idf.')
        
        else:
            pass
            
        return idf

    def __validate_osm(self, osm):
        if not 'OS:AirLoopHVAC:UnitarySystem' in osm:
            raise EnergyPlusSimulationError(error_id=1, message='OS:AirLoopHVAC:UnitarySystem not found in osm.')
        
        elif not 'OS:ThermostatSetpoint:DualSetpoint' in osm:
            raise EnergyPlusSimulationError(error_id=2, message='OS:ThermostatSetpoint:DualSetpoint not found in osm.')
        
        else:
            pass

        return osm
            
    def __preprocess_idf(self):
        # idf object
        idf = self.__simulator[self.ideal_loads_air_system].get_idf_object()

        # set ideal loads to satisfy solely sensible load
        if self.ideal_loads_air_system:
            for obj in idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem']:
                obj.Dehumidification_Control_Type = 'None'
                obj.Cooling_Sensible_Heat_Ratio = 1.0
        else:
            pass

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
        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name.lower() in self.schedules.keys():
                obj.File_Name = self.__schedules_filepath
            else:
                continue

        # put setpoint schedule object
        if self.setpoints is not None:
            obj = idf.newidfobject('Schedule:File')
            schedule_object_name = f'ecobee setpoint'
            obj.Name = schedule_object_name
            obj.Schedule_Type_Limits_Name = 'Temperature'
            obj.File_Name = self.__setpoints_filepath
            obj.Column_Number = 1
            obj.Rows_to_Skip_at_Top = 1
            obj.Number_of_Hours_of_Data = 8760
            obj.Minutes_per_Item = 60

            for obj in idf.idfobjects['ThermostatSetpoint:DualSetpoint']:
                obj.Cooling_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'
                obj.Heating_Setpoint_Temperature_Schedule_Name = f'ecobee setpoint'
        else:
            pass
        
        self.__simulator[self.ideal_loads_air_system].idf = idf.idfstr()

    @staticmethod
    def get_train_data(database, metadata_id):
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/get_lstm_train_data.sql')
        data = database.query_table_from_file(query_filepath, replace={'<metadata_id>': metadata_id})

        return data
    
    @staticmethod
    def initialize_database(database):
        query_filepath = os.path.join(os.path.dirname(__file__), 'misc/queries/set_lstm_tables_and_views.sql')
        database.execute_sql_from_file(query_filepath)

class Error(Exception):
    """Base class for other exceptions."""

class EnergyPlusSimulationError(Error):
    __MESSAGE = 'Simulation errors were found.'
  
    def __init__(self, error_id, message=None):
        self.error_id = error_id
        self.message = self.__MESSAGE if message is None else message
        super().__init__(message)
    