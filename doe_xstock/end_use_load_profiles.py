import os
from pathlib import Path
from typing import Any, Callable, List, Mapping, Union
from eppy.modeleditor import IDF
from pandas import DataFrame
from doe_xstock.data import (
    DataDictionary, 
    EnumerationDictionary, 
    Metadata, 
    OpenStudioModel, 
    Schedules, 
    SpatialTract, 
    TimeSeries, 
    UpgradeDictionary, 
    Version,
    VersionDatasetType,
    Weather
)
from doe_xstock.simulate import EndUseLoadProfilesEnergyPlusSimulator

class EndUseLoadProfilesMetadata:
    def __init__(self, version: Version):
        self.version = version

    @property
    def metadata(self) -> Metadata:
        return Metadata(version=self.version)
    
    @property
    def data_dictionary(self) -> DataDictionary:
        return DataDictionary(version=self.version)
    
    @property
    def enumeration_dictionary(self) -> EnumerationDictionary:
        return EnumerationDictionary(version=self.version)
    
    @property
    def upgrade_dictionary(self) -> UpgradeDictionary:
        return UpgradeDictionary(version=self.version)
    
    @property
    def spatial_tract(self) -> SpatialTract:
        return SpatialTract(version=self.version)

class EndUseLoadProfilesBuilding:
    def __init__(self, bldg_id: int, version: Version):
        self.bldg_id = bldg_id
        self.version = version
        self.simulator: EndUseLoadProfilesEnergyPlusSimulator = None

    @property
    def metadata(self) -> Mapping[str, Union[float, int, str]]:
        return self.time_series.get_metadata()

    @property
    def time_series(self) -> TimeSeries:
        return TimeSeries(bldg_id=self.bldg_id, version=self.version)

    @property
    def schedules(self) -> Schedules:
        return Schedules(bldg_id=self.bldg_id, version=self.version)

    @property
    def open_studio_model(self) -> OpenStudioModel:
        return OpenStudioModel(bldg_id=self.bldg_id, version=self.version)
    
    @property
    def weather(self) -> Weather:
        return Weather(bldg_id=self.bldg_id, version=self.version)

class EndUseLoadProfiles:
    def __init__(self, dataset_type: str = None, weather_data: str = None, year_of_publication: int = None, release: int = None, cache: bool = None):
        self.version = Version(cache=cache)
        self.dataset_type = dataset_type
        self.weather_data = weather_data
        self.year_of_publication = year_of_publication
        self.release = release
        self.metadata = EndUseLoadProfilesMetadata(self.version)

    @property
    def dataset_type(self) -> str:
        return self.__dataset_type
    
    @property
    def weather_data(self) -> str:
        return self.__weather_data
    
    @property
    def year_of_publication(self) -> int:
        return self.__year_of_publication
    
    @property
    def release(self) -> int:
        return self.__release
    
    @dataset_type.setter
    def dataset_type(self, value: str):
        self.__dataset_type = 'resstock' if value is None else value
        self.version.dataset_type = self.dataset_type

    @weather_data.setter
    def weather_data(self, value: str):
        self.__weather_data = 'tmy3' if value is None else value
        self.version.weather_data = self.weather_data

    @year_of_publication.setter
    def year_of_publication(self, value: int):
        self.__year_of_publication = 2021 if value is None else value
        self.version.year_of_publication = self.year_of_publication

    @release.setter
    def release(self, value: int):
        self.__release = 1 if value is None else value
        self.version.release = self.release
    
    def simulate_buildings(self, buildings: List[EndUseLoadProfilesBuilding], max_workers: int = None) -> List[EndUseLoadProfilesBuilding]:
        EndUseLoadProfilesEnergyPlusSimulator.multi_simulate([b.simulator for b in buildings], max_workers=max_workers)

        return buildings
    
    def simulate_building(
        self, bldg_id: int, idd_filepath: Union[str, Path], ideal_loads: bool = None, edit_ems: bool = None, simulation_id: str = None, 
        output_directory: Union[Path, str] = None, output_variables: List[str] = None, output_meters: List[str] = None, 
        default_output_variables: bool = None, default_output_meters: bool = None, model: Union[Path, str] = None, epw: Union[Path, str] = None, 
        osm: bool = None, schedules: Union[Path, DataFrame, str] = None, thermostat_setpoint: Union[Path, DataFrame, str] = None,  number_of_time_steps_per_hour: int = None, idf_preprocessing_customization_function: Callable[[IDF], IDF] = None, **kwargs
    ) -> EndUseLoadProfilesBuilding:
        building: EndUseLoadProfilesBuilding = self.prepare_building_for_simulation(
            bldg_id,
            idd_filepath,
            ideal_loads=ideal_loads,
            edit_ems=edit_ems,
            simulation_id=simulation_id,
            output_directory=output_directory,
            output_variables=output_variables,
            output_meters=output_meters,
            default_output_variables=default_output_variables,
            default_output_meters=default_output_meters,
            model=model,
            epw=epw,
            osm=osm,
            schedules=schedules,
            thermostat_setpoint=thermostat_setpoint,
            number_of_time_steps_per_hour=number_of_time_steps_per_hour,
            idf_preprocessing_customization_function=idf_preprocessing_customization_function,
        )
        building.simulator.simulate(**kwargs)

        return building
    
    def prepare_building_for_simulation(
        self, bldg_id: int, idd_filepath: Union[str, Path], ideal_loads: bool = None, edit_ems: bool = None, simulation_id: str = None, 
        output_directory: Union[Path, str] = None, output_variables: List[str] = None, output_meters: List[str] = None, 
        default_output_variables: bool = None, default_output_meters: bool = None, model: Union[Path, str] = None, epw: Union[Path, str] = None, 
        osm: bool = None, schedules: Union[Path, DataFrame, str] = None, thermostat_setpoint: Union[Path, DataFrame, str] = None, number_of_time_steps_per_hour: int = None, idf_preprocessing_customization_function: Callable[[IDF], IDF] = None,
    ) -> EndUseLoadProfilesBuilding:
        # set output variables and meters
        default_output_variables = False if default_output_variables is None else default_output_variables
        default_output_meters = False if default_output_meters is None else default_output_meters
        default_output_variables = EndUseLoadProfilesEnergyPlusSimulator.get_default_simulation_output_variables() if default_output_variables else []
        default_output_meters = EndUseLoadProfilesEnergyPlusSimulator.get_default_simulation_output_meters() if default_output_meters else []
        output_variables = default_output_variables if output_variables is None else output_variables
        output_meters = default_output_meters if output_meters is None else output_meters

        # get building object
        building = self.get_building(bldg_id)
        building.simulator = EndUseLoadProfilesEnergyPlusSimulator(
            building.version,
            idd_filepath, 
            building.open_studio_model.get() if model is None else model,
            building.weather.get()[0] if epw is None else epw,
            osm=True if model is None else osm,
            number_of_time_steps_per_hour=number_of_time_steps_per_hour,
            ideal_loads=ideal_loads, 
            edit_ems=edit_ems,
            output_variables=output_variables,
            output_meters=output_meters, 
            simulation_id=simulation_id,
            output_directory=output_directory,
            idf_preprocessing_customization_function=idf_preprocessing_customization_function,
        )

        # set schedules
        os.makedirs(Path(building.simulator.schedules_filepath).parent, exist_ok=True)
        
        if schedules is None:
            if building.version.dataset_type == VersionDatasetType.RESSTOCK.value:
                building.schedules.get().to_csv(building.simulator.schedules_filepath, index=False)
            
            else:
                pass
        
        elif isinstance(schedules, (Path, str)):
            assert os.path.isfile(schedules), f'Did not find schedules at: {schedules}'
            building.simulator.schedules_filepath = schedules

        elif isinstance(schedules, DataFrame):
            schedules.to_csv(building.simulator.schedules_filepath, index=False)
            del schedules

        else:
            raise Exception('Unknown schedules format')
        
        # set thermostat setpoint
        if thermostat_setpoint is None:
            pass
        
        else:
            os.makedirs(Path(building.simulator.thermostat_setpoint_filepath).parent, exist_ok=True)
            
            if isinstance(thermostat_setpoint, (Path, str)):
                assert os.path.isfile(thermostat_setpoint), f'Did not find thermostat_setpoint at: {thermostat_setpoint}'
                building.simulator.thermostat_setpoint_filepath = thermostat_setpoint

            elif isinstance(thermostat_setpoint, DataFrame):
                thermostat_setpoint.to_csv(building.simulator.thermostat_setpoint_filepath, index=False)
                del thermostat_setpoint

            else:
                raise Exception('Unknown thermostat_setpoint format')
        
        # save osm model
        with open(os.path.join(building.simulator.output_directory, f'{building.simulator.simulation_id}.osm'), 'w') as f:
            f.write(building.open_studio_model.get())

        return building
    
    def get_buildings(self, bldg_ids: List[int] = None, filters: Mapping[str, List[Any]] = None) -> List[EndUseLoadProfilesBuilding]:
        if bldg_ids is None:
            bldg_ids = self.metadata.metadata.get(filters=filters).index.tolist()

        else:
            pass 

        return [self.get_building(b) for b in bldg_ids]

    def get_building(self, bldg_id: int) -> EndUseLoadProfilesBuilding:
        return EndUseLoadProfilesBuilding(bldg_id=bldg_id, version=self.version)