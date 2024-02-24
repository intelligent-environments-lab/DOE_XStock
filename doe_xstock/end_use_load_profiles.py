import os
from pathlib import Path
import shutil
from typing import Any, List, Mapping, Union
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

    def get_buildings(self, bldg_ids: List[int] = None, filters: Mapping[str, List[Any]] = None) -> List[EndUseLoadProfilesBuilding]:
        if bldg_ids is None:
            bldg_ids = self.metadata.metadata.get(filters=filters).index.tolist()

        else:
            pass 

        return [self.get_building(b) for b in bldg_ids]

    def get_building(self, bldg_id: int) -> EndUseLoadProfilesBuilding:
        return EndUseLoadProfilesBuilding(bldg_id=bldg_id, version=self.version)
    
    def simulate_building(
            self, bldg_id: int, idd_filepath: Union[str, Path], ideal_loads: bool = None, edit_ems: bool = None, simulation_id: str = None, 
            output_directory: Union[Path, str] = None, output_variables: List[str] = None, model: Union[Path, str] = None, 
            epw: Union[Path, str] = None, osm: bool = None, schedules: Union[Path, DataFrame, str] = None, **kwargs
    ) -> EndUseLoadProfilesBuilding:
        building = self.get_building(bldg_id)
        building.simulator = EndUseLoadProfilesEnergyPlusSimulator(
            idd_filepath, 
            building.open_studio_model.get() if model is None else model,
            building.weather.get()[0] if epw is None else epw,
            osm=True if model is None else osm,
            ideal_loads=ideal_loads, 
            edit_ems=edit_ems,
            output_variables=output_variables, 
            simulation_id=simulation_id, 
            output_directory=output_directory
        )
        os.makedirs(building.simulator.output_directory, exist_ok=True)
        schedules_filepath = os.path.join(building.simulator.output_directory, 'schedules.csv')
        
        if schedules is None:
            building.schedules.get().to_csv(schedules_filepath, index=False)
        
        elif isinstance(schedules, (Path, str)):
            assert os.path.isfile(schedules)
            _ = shutil.copy2(schedules, schedules_filepath)

        elif isinstance(schedules, DataFrame):
            schedules.to_csv(schedules_filepath, index=False)

        else:
            raise Exception('Unknown schedules format')
        
        building.simulator.simulate(**kwargs)

        return building