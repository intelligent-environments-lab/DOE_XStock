import concurrent.futures
import logging
import logging.config
import os
from pathlib import Path
import uuid
from eppy.runner.run_functions import EnergyPlusRunError
import numpy as np
import pandas as pd
from doe_xstock.database import SQLiteDatabase
from doe_xstock.simulate import OpenStudioModelEditor, Simulator
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),'misc/logging_config.json'))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_w')

class LSTMData:
    def __init__(self,idd_filepath,osm,epw,schedules,ideal_loads=None,edit_ems=None,output_variables=None,iterations=None,max_workers=None,seed=None,**kwargs):
        self.idd_filepath = idd_filepath
        self.osm = osm
        self.epw = epw
        self.schedules = schedules
        self.ideal_loads = ideal_loads
        self.edit_ems = edit_ems
        self.output_variables = output_variables
        self.iterations = iterations
        self.max_workers = max_workers
        self.seed = seed
        self.kwargs = kwargs
        self.__simulator = None
        self.__ideal_loads = None
        self.__zones = None

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
    def ideal_loads(self):
        return self.__ideal_loads

    @property
    def edit_ems(self):
        return self.__edit_ems

    @property
    def output_variables(self):
        return self.__output_variables

    @property
    def iterations(self):
        return self.__iterations

    @property
    def max_workers(self):
        return self.__max_workers

    @property
    def seed(self):
        return self.__seed

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

    @ideal_loads.setter
    def ideal_loads(self,ideal_loads):
        self.__ideal_loads = False if ideal_loads is None else ideal_loads
    
    @edit_ems.setter
    def edit_ems(self,edit_ems):
        self.__edit_ems = True if edit_ems is None else edit_ems

    @output_variables.setter
    def output_variables(self,output_variables):
        default_output_variables = [
            'Site Direct Solar Radiation Rate per Area','Site Outdoor Air Drybulb Temperature','Zone People Occupant Count',
            'Zone Air Temperature','Zone Thermostat Cooling Setpoint Temperature','Zone Thermostat Heating Setpoint Temperature',
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate','Other Equipment Convective Heating Rate',
            'Zone Ideal Loads Zone Sensible Cooling Rate','Zone Ideal Loads Zone Sensible Heating Rate',
            'Zone Air System Sensible Cooling Rate','Zone Air System Sensible Heating Rate',
        ]
        self.__output_variables =default_output_variables if output_variables is None else output_variables

    @iterations.setter
    def iterations(self,iterations):
        self.__iterations = 3 if iterations is None else iterations

    @max_workers.setter
    def max_workers(self,max_workers):
        self.__max_workers = 1 if max_workers is None else max_workers

    @seed.setter
    def seed(self,seed):
        self.__seed = 0 if seed is None else seed

    def run(self,**kwargs):
        LOGGER.info('Started simulation.')
        self.__set_simulator()
        LOGGER.debug('Simulating ideal loads.')
        # self.__simulate_ideal_loads(**kwargs)
        self.__set_zones()
        self.__set_ideal_loads()
        self.__transform_idf()
        seeds = [None] + [i for i in range(self.iterations)]
        simulators = [self.__get_partial_load_simulator(i,seed=s) for i,s in enumerate(seeds)]
        # Simulator.multi_simulate(simulators,max_workers=self.max_workers)
        LOGGER.debug('Simulating partial load iterations.')

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = [executor.submit(self.__post_process_partial_load_simulation,*[s]) for s in simulators]

            for _, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    r = future.result()
                    LOGGER.debug(f'finished simulaton_id:{r}')
                except Exception as e:
                    LOGGER.exception(e)

        LOGGER.info('Ended simulation.')

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
        assert air_system_cooling == 0 and air_system_heating == 0,\
            f'simulation_id-{simulator.simulation_id}: Non-zero Zone Air System Sensible Cooling Rate'\
                f' and/or Zone Air System Sensible Heating Rate: ({air_system_cooling, air_system_heating})'

        # create zone conditioning table
        data = pd.DataFrame([v for _,v in self.__zones.items()])
        query = """
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
        """
        simulator.get_database().query(query)
        simulator.get_database().insert('zone_metadata',data.columns.tolist(),data.values,)

        # # query summary data
        # query = """
        # SELECT
        #     r.TimeIndex,
        #     'air_temperature' AS label,
        #     r.Value*z.conditioned_floor_area_proportion AS value
        # FROM ReportData r
        # INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        # INNER JOIN "zone" z ON z.ZoneName = d.KeyValue
        # WHERE
        #     (d.Name = 'Zone Air Temperature' AND (z.is_cooled != 0 OR z.is_heated != 0))
        # """

        return simulator.simulation_id

    def __get_iteration_data(self):
        zones = self.__zones
        variable = 'Zone Mean Air Temperature'
        cooled_zones, heated_zones = self.__get_conditioned_zones()
        zone_ixs = list(set(cooled_zones + heated_zones))
        zone_ixs = ','.join(zone_ixs)
        query = f"""
        SELECT
            t.Hour AS hour,
            t.DayType AS day_of_week,
            t.Month AS month,
            r.Value*z.conditioned_floor_area_proportion AS temperature
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        INNER JOIN Zones z ON z.ZoneName = d.KeyValue
        INNER JOIN Time t ON t.TimeIndex = r.TimeIndex
        WHERE d.Name = '{variable}' AND z.ZoneIndex IN ({zone_ixs})
        """

    def __set_ideal_loads(self):
        cooled_zones, heated_zones = self.__get_conditioned_zones()
        cooled_zone_names = [f'\'{z}\'' for z in cooled_zones]
        heated_zone_names = [f'\'{z}\'' for z in heated_zones]
        cooling_column = lambda x: f'{x}.value' if self.ideal_loads else f'CASE WHEN {x}.Value > 0 THEN 0 ELSE r.Value END'
        heating_column = lambda x: f'{x}.value' if self.ideal_loads else f'CASE WHEN {x}.Value < 0 THEN 0 ELSE r.Value END'
        cooling_variable = 'Zone Ideal Loads Zone Sensible Cooling Rate' if self.ideal_loads else 'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate'
        heating_variable = 'Zone Ideal Loads Zone Sensible Heating Rate' if self.ideal_loads else cooling_variable
        zone_join_query = lambda x: f"REPLACE({x}.KeyValue, ' IDEAL LOADS AIR SYSTEM', '')" if self.ideal_loads else f'{x}.KeyValue'
        query = f"""
        SELECT
            r.TimeIndex AS timestep,
            'cooling' AS load,
            z.ZoneIndex AS zone_index,
            z.ZoneName AS zone_name,
            {cooling_column('r')} AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        LEFT JOIN Zones z ON z.ZoneName = {zone_join_query('d')}
        WHERE d.Name = '{cooling_variable}' AND z.ZoneName IN ({','.join(cooled_zone_names)})
        UNION
        SELECT
            r.TimeIndex AS timestep,
            'heating' AS load,
            z.ZoneIndex AS zone_index,
            z.ZoneName AS zone_name,
            {heating_column('r')} AS value
        FROM ReportData r
        INNER JOIN ReportDataDictionary d ON d.ReportDataDictionaryIndex = r.ReportDataDictionaryIndex
        LEFT JOIN Zones z ON z.ZoneName = {zone_join_query('d')}
        WHERE d.Name = '{heating_variable}' AND z.ZoneName IN ({','.join(heated_zone_names)})
        """
        data = self.__simulator.get_database().query_table(query)
        data = data.pivot(index=['zone_name','zone_index','timestep'],columns='load',values='value')
        data = data.reset_index(drop=False).to_dict(orient='list')
        data.pop('index',None)
        self.__ideal_loads = data

    def __get_partial_load_simulator(self,uid,seed=None):
        # get multiplier
        size = len(self.__ideal_loads['timestep'])
        multiplier = [1]*size if seed is None else self.get_multipliers(size,seed=self.seed*seed)
        multiplier = pd.DataFrame(multiplier,columns=['multiplier'])
        multiplier['timestep'] = multiplier.index + 1

        # set load schedule file
        data = pd.DataFrame(self.__ideal_loads)
        data = data.merge(multiplier,on='timestep',how='left')
        data['cooling'] *= data['multiplier']
        data['heating'] *= data['multiplier']

        # save load schedule file
        simulation_id = f'{self.__simulator.simulation_id}_{uid}'
        output_directory = f'{self.__simulator.output_directory}_{uid}'
        filepath = os.path.join(output_directory,f'load_{simulation_id}.csv')
        os.makedirs(output_directory,exist_ok=True)
        data[['cooling','heating']].to_csv(filepath,index=False)

        # set idf
        idf = self.__simulator.get_idf_object()

        for obj in idf.idfobjects['Schedule:File']:
            if 'lstm' in obj.Name.lower():
                obj.File_Name = filepath
            else:
                continue

        # simulate
        return Simulator(
            self.idd_filepath,
            idf.idfstr(),
            self.epw,
            simulation_id=simulation_id,
            output_directory=output_directory,
        )
    
    def __transform_idf(self):
        # generate idf for simulations with partial loads
        idf = self.__simulator.get_idf_object()

        # remove Ideal Air Loads System if any
        idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem'] = []

        # set hvac equipment availability to always off
        obj_names = [n.upper() for n in [
            'AirTerminal:SingleDuct:Uncontrolled','Fan:ZoneExhaust','WaterHeater:HeatPump','ZoneHVAC:Baseboard:Convective:Electric','ZoneHVAC:Baseboard:Convective:Water','ZoneHVAC:Baseboard:RadiantConvective:Electric','ZoneHVAC:Baseboard:RadiantConvective:Water',
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
            'AirLoopHVAC:UnitarySystem'
        ]]

        for name in obj_names:
            if name in idf.idfobjects.keys():
                for obj in idf.idfobjects[name]:
                    obj.Availability_Schedule_Name = 'Always Off Discrete'
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
        zone_names = set(self.__ideal_loads['zone_name'])
        timesteps = max(self.__ideal_loads['timestep'])
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
                obj.Minutes_per_Item = 15

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

    def get_multipliers(self,size,seed=0,minimum_value=0.3,maximum_value=1.7,probability=0.6):
        np.random.seed(seed)
        schedule = np.random.uniform(minimum_value,maximum_value,size)
        schedule[np.random.random(size) > probability] = 1
        return schedule.tolist()

    def __get_conditioned_zones(self):
        cooled_zones = [k for k,v in self.__zones.items() if v['is_cooled']==1]
        heated_zones = [k for k,v in self.__zones.items() if v['is_heated']==1]
        return cooled_zones, heated_zones

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
        if self.ideal_loads and self.__initial_simulation:
            self.__simulate_ideal_loads(**kwargs)
        else:
            found_objs = True

            while True:
                removed_objs = {}
                edited_objs = {}

                try:
                    self.__simulator.simulate()
                    break

                except EnergyPlusRunError as e:
                    if (self.__simulator.has_ems_input_error() or self.__simulator.has_ems_program_error()) and found_objs:
                        try:
                            removed_objs = {
                                **self.__simulator.remove_ems_objs_in_error(patterns=kwargs.get('patterns',None)),
                                **(self.__simulator.remove_ems_program_objs_in_line_error() if not self.edit_ems else {}),
                            }
                            edited_objs = {
                                **(self.__simulator.redefine_ems_program_in_line_error() if self.edit_ems else {})
                            }
                            found_objs = len(removed_objs) + len(edited_objs) == 0
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

        if self.ideal_loads:
            osm_editor.use_ideal_loads_air_system()
        else:
            pass

        self.__simulator = Simulator(
            self.idd_filepath,
            osm_editor.forward_translate(),
            self.epw,
            simulation_id=self.kwargs.get('simulation_id',None),
            output_directory=self.kwargs.get('output_directory',None)
        )
        self.__preprocess_idf()
       
    def __preprocess_idf(self):
        # make output directory
        os.makedirs(self.__simulator.output_directory,exist_ok=True)

        # update output variables
        idf = self.__simulator.get_idf_object()
        idf.idfobjects['Output:Variable'] = []

        for output_variable in self.output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable
            obj.Reporting_Frequency = 'Timestep'

        # remove daylight savings definition
        idf.idfobjects['RunPeriodControl:DaylightSavingTime'] = []

        # set schedules filepath
        schedules_filepath = os.path.join(self.__simulator.output_directory,f'schedules_{self.__simulator.simulation_id}.csv')
        pd.DataFrame(self.schedules).to_csv(schedules_filepath,index=False)

        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name in self.schedules.keys():
                obj.File_Name = schedules_filepath
            else:
                continue
        
        # set ideal loads to satisfy solely sensible load
        if self.ideal_loads:
            for obj in idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem']:
                obj.Dehumidification_Control_Type = 'None'
                obj.Cooling_Sensible_Heat_Ratio = 1.0
        else:
            pass

        self.__simulator.idf = idf.idfstr()
    