import os
from pathlib import Path
from typing import Any, List, Mapping, Union
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
from doe_xstock.simulate import EnergyPlusSimulator, OpenStudioModelEditor

class VersionMetadata:
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

class VersionBuilding:
    def __init__(self, bldg_id: int, version: Version):
        self.bldg_id = bldg_id
        self.version = version
        self.simulator: EnergyPlusSimulator = None

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
        self.metadata = VersionMetadata(self.version)

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

    def get_buildings(self, bldg_ids: List[int] = None, filters: Mapping[str, List[Any]] = None) -> List[VersionBuilding]:
        if bldg_ids is None:
            bldg_ids = self.metadata.metadata.get(filters=filters).index.tolist()

        else:
            pass 

        return [self.get_building(b) for b in bldg_ids]

    def get_building(self, bldg_id: int) -> VersionBuilding:
        return VersionBuilding(bldg_id=bldg_id, version=self.version)
    
    def simulate_building(
            self, bldg_id: int, idd_filepath: Union[str, Path], simulation_id: str = None, output_directory: Union[Path, str] = None, 
            output_variables: List[str] = None, **kwargs
    ) -> VersionBuilding:
        building = self.get_building(bldg_id)
        osm = OpenStudioModelEditor(building.open_studio_model.get())
        idf = osm.forward_translate()
        epw, _ = building.weather.get()
        schedules = building.schedules.get()

        building.simulator = EnergyPlusSimulator(idd_filepath, idf, epw, simulation_id=simulation_id, output_directory=output_directory)
        os.makedirs(building.simulator.output_directory, exist_ok=True)
        schedules_filepath = os.path.join(building.simulator.output_directory, 'schedules.csv')
        schedules.to_csv(schedules_filepath, index=False)

        # ************************ MUST-DO EDITS TO IDF ************************
        idf = building.simulator.get_idf_object()

        # remove daylight savings definition
        idf.idfobjects['RunPeriodControl:DaylightSavingTime'] = []

        # set schedules filepath in model
        for obj in idf.idfobjects['Schedule:File']:
            if obj.Name.lower() in schedules.columns:
                obj.File_Name = schedules_filepath
            else:
                continue
        
        # set output variables
        output_variables = self.get_default_simulation_output_variables() if output_variables is None else output_variables

        for output_variable in output_variables:
            obj = idf.newidfobject('Output:Variable')
            obj.Variable_Name = output_variable
            obj.Reporting_Frequency = 'Timestep' 
        
        del schedules

        building.simulator.idf = idf.idfstr()
        # ********************************* END ********************************

        building.simulator.simulate(**kwargs)

        return building

    def get_default_simulation_output_variables(self) -> List[str]:
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

        return default_output_variables