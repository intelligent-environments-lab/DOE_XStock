import hashlib
from io import StringIO
from multiprocessing import cpu_count
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, List, Mapping, Union
from eppy.modeleditor import IDDNotSetError, IDF
from eppy.runner.run_functions import EnergyPlusRunError, runIDFs
import openstudio
from openstudio import energyplus, isomodel, osversion, openstudiomodelcore
import pandas as pd
from doe_xstock.data import Data, Version, VersionDatasetType
from doe_xstock.database import SQLiteDatabase
from doe_xstock.utilities import FileHandler

class Base:
    def __init__(self, simulation_id: str = None, output_directory: Union[Path, str] = None, epw: Union[Path, str] = None):
        self.simulation_id = simulation_id
        self.output_directory = output_directory
        self.epw = epw
        self.__epw_filepath = None

    @property
    def simulation_id(self) -> str:
        return self.__simulation_id
    
    @property
    def output_directory(self) -> Union[Path, str]:
        return self.__output_directory

    @property
    def epw(self) -> Union[Path, str]:
        return self.__epw

    @property
    def epw_filepath(self) -> str:
        return self.__epw_filepath
    
    @epw.setter
    def epw(self, value: Union[Path, str]):
        if value is not None and os.path.isfile(value):
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
        self.__output_directory = os.path.abspath(value) if value is not None else os.path.abspath(self.simulation_id)

    def _write_epw(self):
        filepath = os.path.join(self.output_directory, 'weather.epw')

        with open(filepath, 'w') as f:
            f.write(self.epw)
        
        self.__epw_filepath = filepath

class OpenStudioModelEditor(Base):
    def __init__(self, osm: Union[Path, str], measures_directory: Union[Path, str] = None, executable_path: Union[Path, str] = None, root_directory: Union[Path, str] = None, simulation_id: str = None, output_directory: Union[Path, str] = None, epw: Union[Path, str] = None):
        super().__init__(simulation_id=simulation_id, output_directory=output_directory, epw=epw)
        self.osm = osm
        self.measures_directory = measures_directory
        self.executable_path = executable_path
        self.root_directory = root_directory

    @property
    def osm(self) -> str:
        return self.__osm

    @property
    def measures_directory(self) -> Union[Path, str]:
        return self.__measures_directory
    
    @property
    def executable_path(self) -> Union[Path, str]:
        return self.__executable_path
    
    @property
    def root_directory(self) -> Union[Path, str]:
        return self.__root_directory

    @osm.setter
    def osm(self, value: Union[Path, str]):
        if os.path.isfile(value):
            with open(value, 'r') as f:
                value = f.read()

        else:
            pass

        self.__osm = value

    @measures_directory.setter
    def measures_directory(self, value: Union[Path, str]):
        self.__measures_directory = value

    @executable_path.setter
    def executable_path(self, value: Union[Path, str]):
        self.__executable_path = value

    @root_directory.setter
    def root_directory(self, value: Union[Path, str]):
        self.__root_directory = value

    def forward_translate(self) -> str:
        osm = self.get_model()
        forward_translator = energyplus.ForwardTranslator()
        idf = forward_translator.translateModel(osm)
        idf = str(idf)

        return idf
    
    def apply_measures(self, measures: list[tuple[str, dict[str, Any]]], overwrite: bool = None, simulate: bool = None) -> tuple[bool, str, str, str, SQLiteDatabase]:
        overwrite = True if overwrite is False else overwrite
        simulate = False if simulate is None else simulate

        assert self.root_directory is not None
        assert self.executable_path is not None
        assert self.measures_directory is not None
        assert self.epw is not None

        # create simulation directory
        simulation_directory = os.path.join(self.output_directory, self.simulation_id)
        workflow_filepath = os.path.join(simulation_directory, f'{self.simulation_id}.osw')
        seed_model_filepath = os.path.join(simulation_directory, f'{self.simulation_id}.osm')

        if os.path.isdir(simulation_directory):
            shutil.rmtree(simulation_directory)
        
        else:
            pass
        
        os.makedirs(simulation_directory, exist_ok=True)

        # initialize workflow
        workflow = openstudio.WorkflowJSON()

        # set seed model
        FileHandler.write_data(self.osm, seed_model_filepath)
        workflow.setSeedFile(seed_model_filepath)

        # set EPW weather file
        self._write_epw()
        workflow.setWeatherFile(self.epw_filepath)

        # add measures
        workflow.addMeasurePath(self.measures_directory)
        measure_steps = []

        for m, a in measures:
            measure_step = openstudio.MeasureStep(m)

            for k, v in a.items():
                measure_step.setArgument(k, v)

            measure_steps.append(measure_step)

        workflow.setWorkflowSteps(measure_steps)


        # save workflow
        workflow.setOswPath(workflow_filepath)
        workflow.save()

        # run simulation
        args = [self.executable_path, 'run', '-w', workflow_filepath]
        args += ['-m'] if not simulate else []
        _ = subprocess.run(args, cwd=self.root_directory)

        # get results
        osm_filepath = os.path.join(simulation_directory, 'run', 'in.osm')
        idf_filepath = os.path.join(os.path.join(simulation_directory, 'run', 'in.idf'))
        osm = FileHandler.get_data_from_path(osm_filepath)
        idf = FileHandler.get_data_from_path(idf_filepath)
        status = FileHandler.read_json(os.path.join(simulation_directory, 'out.osw'))['completed_status']
        status = True if status == 'Success' else False

        if simulate:
            database = self.__get_output_database(simulation_directory)
        
        else:
            database = None

        return status, osm, idf, simulation_directory, database
    
    def __get_output_database(self, simulation_directory: Union[Path, str]) -> SQLiteDatabase:
        filepath = os.path.join(simulation_directory, 'run', f'eplusout.sql')

        if os.path.isfile(filepath):
            return SQLiteDatabase(filepath)
        
        else:
            raise FileNotFoundError(f'No SQLite database exists for simulation. Make sure a simluation has been run'\
                ' using simulate function and the simulation is set to output into SQLite database.')

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

    def get_roof_area(self) -> float:
        iso_model = isomodel.ISOModelForwardTranslator().translateModel(self.get_model())
        area = iso_model.roofArea()
        
        return area

    def get_model(self) -> openstudiomodelcore.Model:
        version_translator = osversion.VersionTranslator()
        osm = version_translator.loadModelFromString(self.osm).get()
        
        return osm

class EnergyPlusSimulator(Base):
    def __init__(
        self, idd_filepath: Union[Path, str], idf: Union[Path, str], number_of_time_steps_per_hour: int = None,idf_preprocessing_customization_function: Callable[[IDF], IDF] = None, idf_preprocessing_customization_function_kwargs: dict = None, simulation_id: str = None, output_directory: Union[Path, str] = None, epw: Union[Path, str] = None):
        super().__init__(simulation_id=simulation_id, output_directory=output_directory, epw=epw)
        self.idd_filepath = idd_filepath
        self.idf = idf
        self.number_of_time_steps_per_hour = number_of_time_steps_per_hour
        self.idf_preprocessing_customization_function = idf_preprocessing_customization_function
        self.idf_preprocessing_customization_function_kwargs = idf_preprocessing_customization_function_kwargs
            
    @property
    def idd_filepath(self) -> str:
        return self.__idd_filepath

    @property
    def idf(self) -> str:
        return self.__idf
    
    @property
    def number_of_time_steps_per_hour(self) -> int:
        return self.__number_of_time_steps_per_hour

    @property
    def simulation_id(self) -> str:
        return self.__simulation_id

    @property
    def output_directory(self) -> Union[Path, str]:
        return self.__output_directory
    
    @property
    def idf_preprocessing_customization_function(self) -> Callable[[IDF], IDF]:
        return self.__idf_preprocessing_customization_function
    
    @property
    def idf_preprocessing_customization_function_kwargs(self) -> dict:
        return self.__idf_preprocessing_customization_function_kwargs

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

    @number_of_time_steps_per_hour.setter
    def number_of_time_steps_per_hour(self, value: int):
        self.__number_of_time_steps_per_hour = value

    @simulation_id.setter
    def simulation_id(self, value: str):
        self.__simulation_id = value if value is not None else 'simulation'

    @output_directory.setter
    def output_directory(self, value: Union[Path, str]):
        self.__output_directory = os.path.abspath(value) if value is not None else os.path.abspath(self.simulation_id)

    @idf_preprocessing_customization_function.setter
    def idf_preprocessing_customization_function(self, value: Callable[[IDF], IDF]):
        if value is None:
            value = lambda idf: idf
        
        else:
            pass

        self.__idf_preprocessing_customization_function = value

    @idf_preprocessing_customization_function_kwargs.setter
    def idf_preprocessing_customization_function_kwargs(self, value: dict):
        self.__idf_preprocessing_customization_function_kwargs = {} if value is None else value 

    def get_output_database(self) -> SQLiteDatabase:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}.sql')

        if os.path.isfile(filepath):
            return SQLiteDatabase(filepath)
        
        else:
            raise FileNotFoundError(f'No SQLite database exists for simulation. Make sure a simluation has been run'\
                ' using simulate function and the simulation is set to output into SQLite database.')
        
    def get_output_variables_csv(self) -> pd.DataFrame:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}.csv')

        if os.path.isfile(filepath):
            return pd.read_csv(filepath)
        
        else:
            raise FileNotFoundError(f'No CSV exists for simulation output variables. Make sure a simluation has been run'\
                ' using simulate function and the IDF contains Output:Variable objects.')
        
    def get_output_meters_csv(self) -> pd.DataFrame:
        filepath = os.path.join(self.output_directory, f'{self.simulation_id}Meter.csv')

        if os.path.isfile(filepath):
            return pd.read_csv(filepath)
        
        else:
            raise FileNotFoundError(f'No CSV exists for simulation output meters. Make sure a simluation has been run'\
                ' using simulate function and the IDF contains Output:Meter objects.')

    @classmethod
    def multi_simulate(cls, simulators: list, max_workers=None):
        simulators: List[EnergyPlusSimulator] = simulators
        max_workers = cpu_count() if max_workers is None else max_workers
        runs = []

        for simulator in simulators:
            os.makedirs(simulator.output_directory, exist_ok=True)
            simulator._write_epw()
            idf = simulator.preprocess_idf_for_simulation()
            idf_preprocessing_customization_function_kwargs = {'idf': idf, **simulator.idf_preprocessing_customization_function_kwargs}
            idf = simulator.idf_preprocessing_customization_function(**idf_preprocessing_customization_function_kwargs)
            simulator._write_idf(idf.idfstr())
            kwargs = simulator.get_run_kwargs()
            runs.append([idf, kwargs])
        
        runIDFs(runs, max_workers)

    def simulate(self, **run_kwargs):
        os.makedirs(self.output_directory, exist_ok=True)
        run_kwargs = self.get_run_kwargs(**run_kwargs if run_kwargs is not None else {})
        self._write_epw()
        idf = self.preprocess_idf_for_simulation()
        idf_preprocessing_customization_function_kwargs = {'idf': idf, **self.idf_preprocessing_customization_function_kwargs}
        idf = self.idf_preprocessing_customization_function(**idf_preprocessing_customization_function_kwargs)
        self._write_idf(idf.idfstr())
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
            'readvars': False,
            'expandobjects': True,
            'idd': self.idd_filepath,
            'verbose': 'q',
        }
        options = {**options, **kwargs}

        return options

    def _write_idf(self, idf: str):
        filepath = os.path.join(self.output_directory,f'{self.simulation_id}.idf')

        with open(filepath, 'w') as f:
            f.write(idf)

    def preprocess_idf_for_simulation(self) -> IDF:
        idf = self.get_idf_object(self.epw_filepath)

        # update simulation time step
        if self.number_of_time_steps_per_hour is not None:
            idf.idfobjects['Timestep'] = []
            obj = idf.newidfobject('Timestep')
            obj.Number_of_Timesteps_per_Hour = self.number_of_time_steps_per_hour

        else:
            pass

        return idf

    def get_idf_object(self, epw_filepath: Union[Path, str] = None) -> IDF:
        idf = StringIO(self.idf)

        try:
            idf = IDF(idf, epw_filepath)
        
        except IDDNotSetError as e:
            self.idd_filepath = self.idd_filepath
            idf = IDF(idf, epw_filepath)
        
        return idf 
    
class EndUseLoadProfilesEnergyPlusSimulator(EnergyPlusSimulator, Data):
    __DEFAULT_SCHEDULES_FILENAME = 'schedules.csv'
    __DEFAULT_THERMOSTAT_SETPOINT_FILENAME = 'thermostat_setpoint.csv'

    def __init__(
            self, version: Version, idd_filepath: Union[Path, str], model: Union[Path, str], epw: Union[Path, str], 
            schedules_filepath: Union[Path, str] = None, thermostat_setpoint_filepath: Union[Path, str] = None, number_of_time_steps_per_hour: int = None, 
            output_variables: List[str] = None, output_meters: List[str] = None, osm: bool = None, ideal_loads: bool = None, 
            edit_ems: bool = None, simulation_id: str = None, output_directory: Union[Path, str] = None,
            idf_preprocessing_customization_function: Callable[[IDF], IDF] = None, cache: bool = None,
    ):
        self.__ideal_loads = False if ideal_loads is None else ideal_loads
        idf = self.__set_idf(model, osm=osm)
        EnergyPlusSimulator.__init__(
            self, idd_filepath, idf, epw, number_of_time_steps_per_hour=number_of_time_steps_per_hour, 
            simulation_id=simulation_id, output_directory=output_directory, 
            idf_preprocessing_customization_function=idf_preprocessing_customization_function
        )
        Data.__init__(self, version=version, cache=cache, relative_path='end_use_load_profiles_energyplus_simulator')
        self.schedules_filepath = schedules_filepath
        self.thermostat_setpoint_filepath = thermostat_setpoint_filepath
        self.output_variables = output_variables
        self.output_meters = output_meters
        self.edit_ems = edit_ems
        self.__cached_fixed_ideal_loads_idf_filepath = None
    
    @property
    def schedules_filepath(self) -> Union[Path, str]:
        return self.__schedules_filepath
    
    @property
    def thermostat_setpoint_filepath(self) -> Union[Path, str]:
        return self.__thermostat_setpoint_filepath
    
    @property
    def output_variables(self) -> List[str]:
        return self.__output_variables
    
    @property
    def output_meters(self) -> List[str]:
        return self.__output_meters
    
    @property
    def edit_ems(self) -> bool:
        return self.__edit_ems
    
    @property
    def fixed_ideal_loads_idf_cache_path(self) -> str:
        return os.path.join(self.cache_path, 'ideal_loads_idf')
    
    @property
    def cached_fixed_ideal_loads_idf_filepath(self) -> str:
        return self.__cached_fixed_ideal_loads_idf_filepath
    
    @schedules_filepath.setter
    def schedules_filepath(self, value: Union[Path, str]):
        self.__schedules_filepath = os.path.join(self.output_directory, self.__DEFAULT_SCHEDULES_FILENAME) if value is None else value

    @thermostat_setpoint_filepath.setter
    def thermostat_setpoint_filepath(self, value: Union[Path, str]):
        self.__thermostat_setpoint_filepath = os.path.join(self.output_directory, self.__DEFAULT_THERMOSTAT_SETPOINT_FILENAME) \
            if value is None else value

    @output_variables.setter
    def output_variables(self, value: List[str]):
        self.__output_variables = [] if value is None else value

    @output_meters.setter
    def output_meters(self, value: List[str]):
        self.__output_meters = [] if value is None else value

    @edit_ems.setter
    def edit_ems(self, value: bool):
        self.__edit_ems = True if value is None else value

    def simulate(self, **run_kwargs):
        found_objects = True
        error_idf = self.idf
        fixed_idf = None
        self.__cached_fixed_ideal_loads_idf_filepath = None

        if self.cache and self.__ideal_loads:
            self.idf = self.__get_cached_fixed_ideal_loads_idf(error_idf)
        
        else:
            pass

        while True:
            try:
                fixed_idf = self.idf
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
                
        if self.cache and self.__ideal_loads:
            self.__cached_fixed_ideal_loads_idf_filepath = self.__cache_fixed_ideal_loads_idf(error_idf, fixed_idf)
            
        
        else:
            pass
    
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
            r'EnergyManagementSystem:InternalVariable=\S+',
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

                elif current_line.startswith('ElseIf'):
                    o[f'Program_Line_{l}'] = f'ElseIf 1<0'
                
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
        if os.path.isfile(self.schedules_filepath):
            schedule_names = pd.read_csv(self.schedules_filepath).columns.tolist()

            for obj in idf.idfobjects['Schedule:File']:
                if obj.Name.lower() in schedule_names:
                    obj.File_Name = self.schedules_filepath
                    item_count = pd.read_csv(self.schedules_filepath, header=None, skiprows=obj.Rows_to_Skip_at_Top).shape[0]
                    obj.Minutes_per_Item = int(60/(item_count/obj.Number_of_Hours_of_Data))
                
                else:
                    continue
        
        elif self.version.dataset_type == VersionDatasetType.RESSTOCK.value:
            raise Exception(f'{self.version.dataset_type} building simulations require a schedules_filepath variable')
        
        else:
            pass

        # put setpoint schedule object
        if os.path.isfile(self.thermostat_setpoint_filepath):
            assert len(idf.idfobjects['ThermostatSetpoint:DualSetpoint']) > 0, \
                'Using thermostat setpoint schedule is only supported for ThermostatSetpoint:DualSetpoint objects'

            for i, name in enumerate(['cooling', 'heating']):
                obj = idf.newidfobject('Schedule:File')
                schedule_object_name = f'thermostat {name} setpoint'
                obj.Name = schedule_object_name
                obj.Schedule_Type_Limits_Name = 'Temperature'
                obj.File_Name = self.thermostat_setpoint_filepath
                obj.Column_Number = i + 1
                obj.Rows_to_Skip_at_Top = 1
                obj.Number_of_Hours_of_Data = 8760
                item_count = pd.read_csv(self.thermostat_setpoint_filepath, header=None, skiprows=obj.Rows_to_Skip_at_Top).shape[0]
                obj.Minutes_per_Item = int(60/(item_count/obj.Number_of_Hours_of_Data))

                for obj in idf.idfobjects['ThermostatSetpoint:DualSetpoint']:
                    if name == 'cooling':
                        obj.Cooling_Setpoint_Temperature_Schedule_Name = schedule_object_name

                    else:
                        obj.Heating_Setpoint_Temperature_Schedule_Name = schedule_object_name

        else:
            pass
        
        # set ideal loads to satisfy solely sensible load
        if self.__ideal_loads:
            for obj in idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem']:
                obj.Dehumidification_Control_Type = 'None'
                obj.Cooling_Sensible_Heat_Ratio = 1.0

            # turn off sizing
            for obj in idf.idfobjects['SimulationControl']:
                obj.Do_Zone_Sizing_Calculation = 'No'
                obj.Do_System_Sizing_Calculation = 'No'
                obj.Do_Plant_Sizing_Calculation = 'No'
                obj.Run_Simulation_for_Sizing_Periods = 'No'
        
        else:
            pass
        
        # set output variables and meters
        idf.idfobjects['Output:Variable'] = []
        idf.idfobjects['Output:Meter'] = []

        for output_variable in self.output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable
            obj.Reporting_Frequency = 'Timestep'

        for output_meter in self.output_meters:
            obj = idf.newidfobject('Output:Meter')
            obj.Key_Name = output_meter
            obj.Reporting_Frequency = 'Timestep'

        return idf
    
    def __cache_fixed_ideal_loads_idf(self, error_idf: str, fixed_idf) -> str:
        filepath = self.__get_cached_fixed_ideal_loads_idf_filepath(error_idf)
        FileHandler.write_data(fixed_idf, filepath)

        return filepath
    
    def __get_cached_fixed_ideal_loads_idf(self, error_idf: str) -> str:
        filepath = self.__get_cached_fixed_ideal_loads_idf_filepath(error_idf)

        try:
            with open(filepath, 'r') as f:
                error_idf = f.read()
        
        except FileNotFoundError:
            pass

        return error_idf

    def __get_cached_fixed_ideal_loads_idf_filepath(self, error_idf: str) -> str:
        md5_hash = hashlib.md5()
        md5_hash.update(error_idf.encode('utf-8'))
        filename = f'{md5_hash.hexdigest()}.idf'
        os.makedirs(self.fixed_ideal_loads_idf_cache_path, exist_ok=True)
        filepath = os.path.join(self.fixed_ideal_loads_idf_cache_path, filename)

        return filepath

    @classmethod
    def get_default_simulation_output_variables(cls) -> List[str]:
        variables = []

        # weather
        variables += [
            'Site Diffuse Solar Radiation Rate per Area', 
            'Site Direct Solar Radiation Rate per Area', 
            'Site Outdoor Air Drybulb Temperature', 
            'Site Outdoor Air Relative Humidity', 
        ]

        # electric equipment
        variables += [
            'Zone Electric Equipment Electricity Rate', 
            'Zone Lights Electricity Rate', 
        ]

        # mechanical hvac
        variables += [
            'Air System Total Cooling Energy', 
            'Air System Total Heating Energy', 
            'Zone Air System Sensible Cooling Rate', 
            'Zone Air System Sensible Heating Rate', 
            'Zone Predicted Sensible Load to Setpoint Heat Transfer Rate',
        ]

        # ideal loads system
        variables += [
            'Zone Ideal Loads Zone Sensible Cooling Rate', 
            'Zone Ideal Loads Zone Sensible Heating Rate', 
        ]

        # other equipment
        variables += [
            'Other Equipment Convective Heating Rate', 
            'Other Equipment Convective Heating Energy'
        ]

        # dhw
        variables += [
            'Water Use Equipment Heating Rate', 
        ]

        # ieq
        variables += [
            'Zone Air Relative Humidity', 
            'Zone Air Temperature', 
            'Zone Thermostat Cooling Setpoint Temperature', 
            'Zone Thermostat Heating Setpoint Temperature', 
        ]

        # occupancy
        variables += [
            'Zone People Occupant Count', 
        ]

        return variables
    
    @classmethod
    def get_default_simulation_output_meters(cls) -> List[str]:
        # ref: https://bigladdersoftware.com/epx/docs/9-6/input-output-reference/input-for-output.html#outputmeter-and-outputmetermeterfileonly
        fuels =  [
            'Electricity',
            'NaturalGas',
            'Gasoline',
            'Diesel',
            'Coal',
            'Propane',
            'Steam',
            'Water',
        ]

        # facility = building + plant + hvac + exterior
        # building = sum(zones)
        sites = [
            'Facility',
            'Building'
        ]
        meters = [f'{f}:{s}' for f in fuels for s in sites]
        
        return meters
    
    def __set_idf(self, model: Union[Path, str], osm: bool = None) -> str:
        osm = False if osm is None else osm

        if os.path.isfile(model):
            with open(model, 'r') as f:
                model = f.read()

        else:
            pass

        if osm:
            osm_model = OpenStudioModelEditor(model)

            if self.__ideal_loads:
                osm_model.use_ideal_loads_air_system()

            else:
                pass

            idf = osm_model.forward_translate()
        
        else:
            idf = model

        self.validate_idf(idf)

        return idf
    
    def validate_idf(self, idf: str):
        # if not 'AirLoopHVAC:UnitarySystem,' in idf:
        #     raise EnergyPlusSimulationError(error_id=1, message='AirLoopHVAC:UnitarySystem not found in idf.')
        
        if not 'ThermostatSetpoint:DualSetpoint,' in idf:
             raise EnergyPlusSimulationError(error_id=2, message='ThermostatSetpoint:DualSetpoint not found in idf.')
        
        elif not 'ZoneControl:Thermostat,' in idf:
            raise EnergyPlusSimulationError(error_id=3, message='ZoneControl:Thermostat not found in idf.')
        
        else:
            pass
    
class Error(Exception):
    """Base class for other exceptions."""

class EnergyPlusSimulationError(Error):
    __MESSAGE = 'Simulation errors were found.'
  
    def __init__(self, error_id: int, message: str = None):
        self.error_id = error_id
        self.message = self.__MESSAGE if message is None else message
        super().__init__(message)