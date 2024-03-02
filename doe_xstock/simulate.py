from io import StringIO
import os
from pathlib import Path
import re
from typing import Mapping, List, Union
from eppy.modeleditor import IDF
from eppy.runner.run_functions import EnergyPlusRunError, runIDFs
from openstudio import energyplus, osversion, openstudiomodelcore
import pandas as pd
from doe_xstock.database import SQLiteDatabase

class OpenStudioModelEditor:
    def __init__(self, osm: Union[Path, str]):
        self.osm = osm

    @property
    def osm(self) -> str:
        return self.__osm

    @osm.setter
    def osm(self, value: Union[Path, str]):
        if os.path.isfile(value):
            with open(value, 'r') as f:
                value = f.read()

        else:
            pass

        self.__osm = value

    def forward_translate(self) -> str:
        osm = self.get_model()
        forward_translator = energyplus.ForwardTranslator()
        idf = forward_translator.translateModel(osm)
        idf = str(idf)

        return idf

    def use_ideal_loads_air_system(self):
        # Reference: https://www.rubydoc.info/gems/openstudio-standards/Standard#remove_hvac-instance_method
        osm = self.get_model()

        # remove air loop hvac
        for air_loop in osm.getAirLoopHVACs():
            air_loop.remove()

        # remove plant loops
        for plant_loop in osm.getPlantLoops():
            shw_use = False

            for component in plant_loop.demandComponents():
                if component.to_WaterUseConnections().is_initialized() or component.to_CoilWaterHeatingDesuperheater().is_initialized():
                    shw_use = True
                    break
                else:
                    pass

            if  not shw_use:
                plant_loop.remove()
            else:
                continue

        # remove vrf
        for ac_refrigerant_flow in osm.getAirConditionerVariableRefrigerantFlows():
            ac_refrigerant_flow.remove()

        for terminal_unit_refrigerant_flow in osm.getZoneHVACTerminalUnitVariableRefrigerantFlows():
            terminal_unit_refrigerant_flow.remove()

        # remove zone equipment
        for zone in osm.getThermalZones():
            for equipment in zone.equipment():
                if not equipment.to_FanZoneExhaust().is_initialized():
                    equipment.remove()
                else:
                    pass

        # remove unused curves
        for curve in osm.getCurves():
            if curve.directUseCount() == 0:
                curve.remove()
            else:
                pass

        # add ideal load system
        for zone in osm.getThermalZones():
            has_thermostat = zone.thermostat().is_initialized()
            zone.setUseIdealAirLoads(has_thermostat)
            
        osm = str(osm)

        self.osm = osm

    def get_model(self) -> openstudiomodelcore.Model:
        version_translator = osversion.VersionTranslator()
        osm = version_translator.loadModelFromString(self.osm).get()
        
        return osm

class EnergyPlusSimulator:
    def __init__(self, idd_filepath: Union[Path, str], idf: Union[Path, str], epw: Union[Path, str], simulation_id: str = None,output_directory: Union[Path, str] = None):
        self.idd_filepath = idd_filepath
        self.epw = epw
        self.idf = idf
        self.simulation_id = simulation_id
        self.output_directory = output_directory
        self.__epw_filepath = None
    
    @property
    def idd_filepath(self) -> str:
        return self.__idd_filepath

    @property
    def idf(self) -> str:
        return self.__idf

    @property
    def epw(self) -> str:
        return self.__epw

    @property
    def simulation_id(self) -> str:
        return self.__simulation_id

    @property
    def output_directory(self) -> Union[Path, str]:
        return self.__output_directory

    @property
    def epw_filepath(self) -> str:
        return self.__epw_filepath

    @idd_filepath.setter
    def idd_filepath(self, value: Union[Path, str]):
        self.__idd_filepath = value
        IDF.setiddname(self.idd_filepath)

    @idf.setter
    def idf(self, value: Union[Path, str]):
        if os.path.isfile(value):
            with open(value, 'r') as f:
                value = f.read()

        else:
            pass

        self.__idf = value

    @epw.setter
    def epw(self, value: Union[Path, str]):
        if os.path.isfile(value):
            with open(value, 'r') as f:
                value = f.read()

        else:
            pass

        self.__epw = value

    @simulation_id.setter
    def simulation_id(self, value: str):
        self.__simulation_id = value if value is not None else 'simulation'

    @output_directory.setter
    def output_directory(self, value: Union[Path, str]):
        self.__output_directory = os.path.abspath(value) if value is not None else os.path.abspath('simulation')

    def get_result_database(self) -> SQLiteDatabase:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}.sql')

        if os.path.isfile(filepath):
            return SQLiteDatabase(filepath)
        
        else:
            raise FileNotFoundError(f'No SQLite database exists for simulation. Make sure a simluation has been run'\
                ' using simulate function and the simulation is set to output into SQLite database.')
        
    def get_result_csv(self) -> pd.DataFrame:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}.csv')

        if os.path.isfile(filepath):
            return pd.read_csv(filepath)
        
        else:
            raise FileNotFoundError(f'No CSV exists for simulation. Make sure a simluation has been run'\
                ' using simulate function and the simulation is set to output into SQLite database.')

    
    @classmethod
    def multi_simulate(cls, simulators: list, max_workers=None):
        simulators: List[EnergyPlusSimulator] = simulators
        max_workers = 1 if max_workers is None else max_workers
        runs = []

        for simulator in simulators:
            os.makedirs(simulator.output_directory, exist_ok=True)
            simulator.__write_epw()
            idf = simulator.preprocess_idf_for_simulation()
            simulator.__write_idf(idf.idfstr())
            kwargs = simulator.get_run_kwargs()
            runs.append([idf, kwargs])
        
        runIDFs(runs, max_workers)

    def simulate(self, **run_kwargs):
        os.makedirs(self.output_directory, exist_ok=True)
        run_kwargs = self.get_run_kwargs(**run_kwargs if run_kwargs is not None else {})
        self.__write_epw()
        idf = self.preprocess_idf_for_simulation()
        self.__write_idf(idf.idfstr())
        idf.run(**run_kwargs)

    def get_error(self) -> str:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}.err')

        with open(filepath, 'r') as f:
            error = f.read()

        return error

    def get_run_kwargs(self, **kwargs) -> Mapping[str, Union[bool, str]]:
        idf = self.get_idf_object()
        idf_version = idf.idfobjects['version'][0].Version_Identifier.split('.')
        idf_version.extend([0] * (3 - len(idf_version)))
        idf_version_str = '-'.join([str(item) for item in idf_version])
        options = {
            'ep_version': idf_version_str,
            'output_prefix': str(self.simulation_id),
            'output_suffix': 'C',
            'output_directory': str(self.output_directory),
            'readvars': True,
            'expandobjects': True,
            'verbose': 'q',
        }
        options = {**options, **kwargs}

        return options

    def __write_epw(self):
        filepath = os.path.join(self.output_directory, 'weather.epw')

        with open(filepath, 'w') as f:
            f.write(self.epw)
        
        self.__epw_filepath = filepath

    def __write_idf(self, idf: str):
        filepath = os.path.join(self.output_directory,f'{self.simulation_id}.idf')

        with open(filepath, 'w') as f:
            f.write(idf)

    def preprocess_idf_for_simulation(self) -> IDF:
        return self.get_idf_object(self.epw_filepath)

    def get_idf_object(self, epw_filepath: Union[Path, str] = None) -> IDF:
        return IDF(StringIO(self.idf), epw_filepath)
    
class EndUseLoadProfilesEnergyPlusSimulator(EnergyPlusSimulator):
    __DEFAULT_SCHEDULES_FILENAME = 'schedules.csv'

    def __init__(
            self, idd_filepath: Union[Path, str], model: Union[Path, str], epw: Union[Path, str], schedules_filepath: Union[Path, str] = None, output_variables: List[str] = None, osm: bool = None, ideal_loads: bool = None, edit_ems: bool = None, simulation_id: str = None, 
            output_directory: Union[Path, str] = None
    ):
        self.__ideal_loads = False if ideal_loads is None else ideal_loads
        idf = self.__set_idf(model, osm=osm)
        super().__init__(idd_filepath, idf, epw, simulation_id, output_directory)
        self.schedules_filepath = schedules_filepath
        self.output_variables = output_variables
        self.edit_ems = edit_ems
    
    @property
    def schedules_filepath(self) -> Union[Path, str]:
        return self.__schedules_filepath
    
    @property
    def output_variables(self) -> List[str]:
        return self.__output_variables
    
    @property
    def edit_ems(self) -> bool:
        return self.__edit_ems
    
    @schedules_filepath.setter
    def schedules_filepath(self, value: Union[Path, str]):
        self.__schedules_filepath = os.path.join(self.output_directory, self.__DEFAULT_SCHEDULES_FILENAME) if value is None else value

    @output_variables.setter
    def output_variables(self, value: List[str]):
        self.__output_variables = self.get_default_simulation_output_variables() if value is None else value

    @edit_ems.setter
    def edit_ems(self, value: bool):
        self.__edit_ems = True if value is None else value

    def simulate(self, **run_kwargs):
        found_objects = True

        while True:
            try:
                super().simulate(**run_kwargs)
                break

            except EnergyPlusRunError as e:
                if self.__ideal_loads:
                    try:
                        found_objects = self.__fix_ideal_load_ems_errors(found_objects, patterns=run_kwargs.get('ems_patterns'))
                    
                    except Exception as e:
                        raise e

                else:
                    raise e
    
    def __fix_ideal_load_ems_errors(self, found_objects: bool, patterns: List[str] = None):
        assert (self.has_ems_input_error() or self.has_ems_program_error())
        assert found_objects
        
        try:
            removed_objects = self.remove_ems_objs_in_error(patterns=patterns)
            
            if self.edit_ems:
                edited_objects = self.redefine_ems_program_in_line_error()
            
            else:
                removed_objects = {**removed_objects, **self.remove_ems_program_objs_in_line_error()}

            found_objects = len(removed_objects) + len(edited_objects) > 0
        
        except Exception as e:
            raise e
        
        return found_objects

    def remove_ems_objs_in_error(self, patterns: List[str] = None) -> Mapping[str, List[str]]:
        default_patterns = [
            r'EnergyManagementSystem:Sensor=\S+',
            # r'EnergyManagementSystem:ProgramCallingManager=.+\s+'
        ]
        patterns = default_patterns if patterns is None else patterns
        objs = {}
        removed_objs = {}
        
        error = self.get_error()
        idf = self.get_idf_object()
    
        for k, v in [o.strip().strip('\n').split('=') for p in patterns for o in re.findall(p, error)]:
            v = v.lower()
            objs[k] = objs[k] + [v] if k in objs.keys() else [v]

        for k, obj in [(k, obj) for k, v in objs.items() for obj in idf.idfobjects[k] if obj.Name.lower() in v]:
            idf.removeidfobject(obj)
            removed_objs[k] = removed_objs[k] + [obj.Name] if k in removed_objs.keys() else [obj.Name]

        self.idf = idf.idfstr()

        return removed_objs

    def redefine_ems_program_in_line_error(self) -> Mapping[str, dict]:
        objs = {}
        idf = self.get_idf_object()

        for t, i, k, v in self.get_ems_program_line_error():
            o = idf.idfobjects[t][i]
            
            for l in v:
                current_line = o[f'Program_Line_{l}']

                if current_line.startswith('Set'):
                    o[f'Program_Line_{l}'] = f'{current_line.split("=")[0]} = 0'
                
                elif current_line.startswith('If'):
                    o[f'Program_Line_{l}'] = f'If 1<0'
                
                else:
                    raise AssertionError(f'Unknown line format: {current_line}')
                
            objs[t] = {**objs.get(t, {}), **{k: v}}

        self.idf = idf.idfstr()

        return objs

    def remove_ems_program_objs_in_line_error(self) -> Mapping[str, list]:
        objs = {}
        idf = self.get_idf_object()
        
        for t, i, k, _ in self.get_ems_program_line_error():
            idf.removeidfobject(idf.idfobjects[t][i])
            objs[t] = objs[t] + [k] if t in objs.keys() else [k]

        self.idf = idf.idfstr()

        return objs

    def get_ems_program_line_error(self) -> List[tuple]:
        target_objs = ['EnergyManagementSystem:Program', 'EnergyManagementSystem:Subroutine']
        error = self.get_error()
        objs = {}
        line_errors = []
        idf = self.get_idf_object()
        matches = re.findall(r'\*\* Severe  \*\* Problem found in EMS EnergyPlus Runtime Language\.\s+'\
            r'\*\*   ~~~   \*\* Erl program name:.+\s+'\
                r'\*\*   ~~~   \*\* Erl program line number:.+\s+',error
        )

        for match in matches:
            match = match.split('\n   **   ~~~   ** ')
            program_name = match[1].split(': ')[-1].lower()
            line_number = match[2].strip().strip('\n').split(': ')[-1]
            objs[program_name] = objs[program_name] + [line_number] if program_name in objs.keys() else [line_number]

        for k, v in objs.items():
            for t, o, i in [(t, o, i) for t in target_objs for i, o in enumerate(idf.idfobjects[t])]:
                if o.Name.lower() == k:
                    line_errors.append((t, i, k, v))
                
                else:
                    continue

        return line_errors

    def has_ems_program_error(self) -> bool:
        return len(re.findall(r'\*\*  Fatal  \*\* Previous EMS error caused program termination', self.get_error())) > 0

    def has_ems_input_error(self) -> bool:
        patterns = [
            r'\*\*  Fatal  \*\* Errors found in processing Energy Management System input. Preceding condition causes termination',
            r'\*\*  Fatal  \*\* Errors found in getting Energy Management System input. Preceding condition causes termination',
        ]
        error = self.get_error()

        return len([re.findall(p, error) for p in patterns]) > 0
    
    def preprocess_idf_for_simulation(self) -> IDF:
        idf = super().preprocess_idf_for_simulation()

        # remove daylight savings definition
        idf.idfobjects['RunPeriodControl:DaylightSavingTime'] = []

        # set schedules filepath in model
        schedule_names = pd.read_csv(self.schedules_filepath).columns.tolist()

        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name.lower() in schedule_names:
                obj.File_Name = self.schedules_filepath
            
            else:
                continue
        
        # set output variables
        for output_variable in self.output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable
            obj.Reporting_Frequency = 'Timestep'

        return idf
    
    def get_default_simulation_output_variables(self) -> List[str]:
        default_output_variables = []

        # weather
        default_output_variables += [
            'Site Diffuse Solar Radiation Rate per Area', 
            'Site Direct Solar Radiation Rate per Area', 
            'Site Outdoor Air Drybulb Temperature', 
            'Site Outdoor Air Relative Humidity', 
        ]

        # electric equipment
        default_output_variables += [
            'Zone Electric Equipment Electricity Rate', 
            'Zone Lights Electricity Rate', 
        ]

        # mechanical hvac
        default_output_variables += [
            'Air System Total Cooling Energy', 
            'Air System Total Heating Energy', 
            'Zone Air System Sensible Cooling Rate', 
            'Zone Air System Sensible Heating Rate', 
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate',
        ]

        # ideal loads system
        default_output_variables += [
            'Zone Ideal Loads Zone Sensible Cooling Rate', 
            'Zone Ideal Loads Zone Sensible Heating Rate', 
        ]

        # other equipment
        default_output_variables += [
            'Other Equipment Convective Heating Rate', 
            'Other Equipment Convective Heating Energy'
        ]

        # dhw
        default_output_variables += [
            'Water Use Equipment Heating Rate', 
        ]

        # ieq
        default_output_variables += [
            'Zone Air Relative Humidity', 
            'Zone Air Temperature', 
            'Zone Thermostat Cooling Setpoint Temperature', 
            'Zone Thermostat Heating Setpoint Temperature', 
        ]

        # occupancy
        default_output_variables += [
            'Zone People Occupant Count', 
        ]

        return default_output_variables
    
    def __set_idf(self, model: Union[Path, str], osm: bool = None) -> str:
        osm = False if osm is None else osm

        if osm:
            osm_model = OpenStudioModelEditor(model)

            if self.__ideal_loads:
                osm_model.use_ideal_loads_air_system()

            else:
                pass

            idf = osm_model.forward_translate()

        elif self.__ideal_loads:
            raise Exception('To simulate ideal loads, model parsed at intialization must be an OpenStudio model and osm must be set to True')
        
        else:
            idf = model

        return idf