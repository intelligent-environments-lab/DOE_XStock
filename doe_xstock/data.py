from enum import Enum, unique
import gzip
import io
import logging
import os
from platformdirs import user_cache_dir
import shutil
from typing import Any, List, Mapping, Tuple, Union
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from doe_xstock.__init__ import __version__, __appname__

LOGGER = logging.getLogger(__appname__)
LOGGER.addHandler(logging.NullHandler())

@unique
class VersionDatasetType(Enum):
    RESSTOCK = 'resstock'
    COMSTOCK = 'comstock'

@unique
class VersionWeatherData(Enum):
    TMY3 = 'tmy3'
    AMY2018 = 'amy2018'

class Version:
    __DEFAULT_YEAR_OF_PUBLICATION = 2021

    def __init__(self, dataset_type: Union[str, VersionDatasetType] = None, weather_data: Union[str, VersionWeatherData] = None, year_of_publication: int = None, release: int = None, cache: bool = None):
        self.root_url = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/'
        self.dataset_type = dataset_type
        self.weather_data = weather_data
        self.year_of_publication = year_of_publication
        self.release = release
        self.cache = cache

    @property
    def cache_directory(self) -> str:
        cache_directory = user_cache_dir(
            appname=__appname__,
            version=f'v{__version__}',
        )
        directory = os.path.join(
            cache_directory,
            'end_use_load_profiles', 
            str(self.year_of_publication), 
            f'{self.dataset_type}_{self.weather_data}_release_{self.release}'
        )
        os.makedirs(directory, exist_ok=True)

        return directory

    @property
    def url(self) -> str:
        return urllib.parse.urljoin(self.root_url, f'{self.year_of_publication}/{self.dataset_type}_{self.weather_data}_release_{self.release}/')

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
    
    @property
    def cache(self) -> bool:
        return self.__cache
    
    @dataset_type.setter
    def dataset_type(self, value: Union[str, VersionDatasetType]):
        if value is None:
            value = VersionDatasetType.RESSTOCK.value

        elif isinstance(value, VersionDatasetType):
            value = value.value

        else:
            valid_values = [v.value for v in VersionDatasetType]
            assert value in valid_values,\
                f'\'{value}\' is not a valid value for dataset_type. Valid values are {valid_values}' 
        
        self.__dataset_type = value

    @weather_data.setter
    def weather_data(self, value: Union[str, VersionWeatherData]):
        if value is None:
            value = VersionWeatherData.TMY3.value

        elif isinstance(value, VersionWeatherData):
            value = value.value

        else:
            valid_values = [v.value for v in VersionWeatherData]
            assert value in valid_values,\
                f'\'{value}\' is not a valid value for weather_data. Valid values are {valid_values}' 
        
        self.__weather_data = value

    @year_of_publication.setter
    def year_of_publication(self, value: int):
        if value is None:
            value = self.__DEFAULT_YEAR_OF_PUBLICATION

        else:
            assert value >= self.__DEFAULT_YEAR_OF_PUBLICATION,\
                f'year_of_publication must be >= {self.__DEFAULT_YEAR_OF_PUBLICATION}'
            
            if value > self.__DEFAULT_YEAR_OF_PUBLICATION:
                LOGGER.warning(f'Setting year_of_publication to {value} may have URL endpoint'\
                    f' issues when retrieving data from {self.root_url} as this library has only been'\
                        f' tested for {self.__DEFAULT_YEAR_OF_PUBLICATION} year_of_publication.')

        self.__year_of_publication = value

    @release.setter
    def release(self, value: int):
        self.__release = 1 if value is None else value

    @cache.setter
    def cache(self, value: bool):
        self.__cache = False if value is None else value

    def clear_cache(self):
        if os.path.isdir(self.cache_directory):
            shutil.rmtree(self.cache_directory)
        
        else:
            pass

    def __str__(self) -> str:
        return f'{self.dataset_type}_{self.year_of_publication}_{self.weather_data}_release_{self.release}'

class Data:
    def __init__(self, name: str = None, relative_path: str = None, version: Version = None, cache: bool = None):
        self.name = name
        self.relative_path = relative_path
        self.version = version
        self.cache = cache

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def relative_path(self) -> str:
        return self.__relative_path
    
    @property
    def version(self) -> Version:
        return self.__version
    
    @property
    def cache(self) -> bool:
        return self.__cache
    
    @name.setter
    def name(self, value: str):
        self.__name = value

    @relative_path.setter
    def relative_path(self, value: str):
        self.__relative_path = value

    @version.setter
    def version(self, value: Version):
        self.__version = Version() if value is None else value

    @cache.setter
    def cache(self, value: bool):
        self.__cache = self.version.cache if value is None else value

    @property
    def cache_path(self) -> str:
        assert self.relative_path is not None, 'set relative_path to get a cache_path'
        path = os.path.join(self.version.cache_directory, self.relative_path)
        os.makedirs(os.path.split(path)[0], exist_ok=True)

        return  path

    def get(self) -> Any:
        raise NotImplementedError
    
class TabularData(Data):
    def __init__(self, name: str = None, relative_path: str = None, reader: Any = None, reader_kwargs: Mapping[str, Any] = None, version: Version = None, cache: bool = None):
        super().__init__(name=name, relative_path=relative_path, version=version, cache=cache)
        self.reader = reader
        self.reader_kwargs = reader_kwargs

    @Data.cache_path.getter
    def cache_path(self) -> str:
        path = super().cache_path
        path = os.path.splitext(path)[0] + '.parquet'

        return path

    @property
    def reader(self) -> Any:
        return self.__reader
    
    @property
    def reader_kwargs(self) -> Mapping[str, Any]:
        return self.__reader_kwargs
    
    @reader.setter
    def reader(self, value: Any):
        self.__reader = value

    @reader_kwargs.setter
    def reader_kwargs(self, value: Mapping[str, Any]):
        self.__reader_kwargs = {} if value is None else value

    def get(self, filters: Mapping[str, List[Any]] = None) -> pd.DataFrame:
        data: pd.DataFrame

        if os.path.isfile(self.cache_path):
            if filters is not None:
                reader_kwargs = {'filters': [(c, 'in', v) for c, v in filters.items()]}
            
            else:
                reader_kwargs = {}

            data = pd.read_parquet(self.cache_path, **reader_kwargs)

        else:
            path = urllib.parse.urljoin(self.version.url, self.relative_path)
            data = self.reader(path, **self.reader_kwargs)

            if self.cache:
                data.to_parquet(self.cache_path)

            else:
                pass

            if filters is not None:
                for column, values in filters.items():
                    data = data[data[column].isin(values)].copy()
        
            else:
                pass

        return data
    
class ParquetData(TabularData):
    def __init__(self, name: str = None, relative_path: str = None, reader_kwargs: Mapping[str, Any] = None, version: Version = None, cache: bool = None):
        super().__init__(name=name, relative_path=relative_path, reader=pd.read_parquet, reader_kwargs=reader_kwargs, version=version, cache=cache)
    
class CSVData(TabularData):
    def __init__(self, name: str = None, relative_path: str = None, version: Version = None, cache: bool = None):
        super().__init__(name=name, relative_path=relative_path, reader=pd.read_csv, reader_kwargs={'sep': ','}, version=version, cache=cache)
    
class TSVData(TabularData):
    def __init__(self, name: str = None, relative_path: str = None, version: Version = None, cache: bool = None):
        super().__init__(name=name, relative_path=relative_path, reader=pd.read_csv, reader_kwargs={'sep': '\t'}, version=version, cache=cache)

class Metadata(ParquetData):
    def __init__(self, version: Version = None, cache: bool = None):
        cache = True if cache is None else cache
        super().__init__(name='building_metadata', relative_path='metadata/metadata.parquet', version=version, cache=cache)

class DataDictionary(TSVData):
    def __init__(self, version: Version = None, cache: bool = None):
        super().__init__(name='data_dictionary', relative_path='data_dictionary.tsv', version=version, cache=cache)

class EnumerationDictionary(TSVData):
    def __init__(self, version: Version = None, cache: bool = None):
        super().__init__(name='enumeration_dictionary', relative_path='enumeration_dictionary.tsv', version=version, cache=cache)

class UpgradeDictionary(TSVData):
    def __init__(self, version: Version = None, cache: bool = None):
        super().__init__(name='upgrade_dictionary', relative_path='upgrade_dictionary.tsv', version=version, cache=cache)

class SpatialTract(CSVData):
    def __init__(self, version: Version = None, cache: bool = None):
        super().__init__(name='spatial_tract', relative_path='geographic_information/spatial_tract_lookup_table.csv', version=version, cache=cache)

class BuildingData(Data):
    def __init__(self, bldg_id: int = None, name: str = None, relative_path: str = None, version: Version = None, cache: bool = None):
        super().__init__(name=name, relative_path=relative_path, version=version, cache=cache)
        self.bldg_id = bldg_id

    @property
    def bldg_id(self) -> int:
        return self.__bldg_id
    
    @bldg_id.setter
    def bldg_id(self, value: int):
        self.__bldg_id = 1 if value is None else value

    def get_metadata(self) -> Mapping[str, Union[float, int, str]]:
        metadata = Metadata(version=self.version).get()
        metadata = metadata.loc[self.bldg_id].to_dict()
        metadata = {'bldg_id': self.bldg_id, **metadata}
        
        return metadata

class TimeSeries(BuildingData, ParquetData):
    def __init__(self, bldg_id: int = None, version: Version = None, cache: bool = None):
        super().__init__(bldg_id=bldg_id, name='building_time_series', version=version, cache=cache)

    @BuildingData.relative_path.getter
    def relative_path(self) -> str:
        metadata = self.get_metadata()
        upgrade = metadata['upgrade']
        upgrade = int(upgrade)
        county = metadata['in.county']
        path = f'timeseries_individual_buildings/by_county/upgrade={upgrade}/county={county}/{self.bldg_id}-{upgrade}.parquet' 

        return path

class Schedules(BuildingData, CSVData):
    def __init__(self, bldg_id: int = None, version: Version = None, cache: bool = None):
        super().__init__(bldg_id=bldg_id, name='building_schedules', version=version, cache=cache)

    @BuildingData.relative_path.getter
    def relative_path(self) -> str:
        metadata = self.get_metadata()
        upgrade = int(metadata['upgrade'])
        path = f'occupancy_schedules/bldg{self.bldg_id:07d}-up{upgrade:02d}.csv.gz'

        return path

class OpenStudioModel(BuildingData):
    def __init__(self, bldg_id: int = None, version: Version = None, cache: bool = None):
        super().__init__(bldg_id=bldg_id, name='building_open_studio_model', version=version, cache=cache)

    @BuildingData.relative_path.getter
    def relative_path(self) -> str:
        metadata = self.get_metadata()
        upgrade = int(metadata['upgrade'])
        path = f'building_energy_models/bldg{int(self.bldg_id):07d}-up{upgrade:02d}.osm.gz'

        return path

    def get(self) -> str:
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, 'r') as f:
                data = f.read()

        else:
            url = urllib.parse.urljoin(self.version.url, self.relative_path)
            response = requests.get(url)
            compressed_file = io.BytesIO(response.content)
            decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode='rb')
            data = decompressed_file.read().decode()

            if self.cache:
                with open(self.cache_path, 'w') as f:
                    f.write(data)

            else:
                pass

        return data
    
class Weather(BuildingData):
    def __init__(self, bldg_id: int = None, county: str = None, version: Version = None, cache: bool = None):
        cache = True if cache is None else cache
        super().__init__(bldg_id=bldg_id, name='building_weather', version=version, cache=cache)
        self.county = county
        self.energy_plus_weather_url = 'https://raw.githubusercontent.com/NREL/EnergyPlus/develop/weather/master.geojson'
        self.nasa_power_weather_url = 'https://power.larc.nasa.gov/api/temporal/hourly/point'

    @property
    def county(self) -> str:
        return self.get_metadata()['in.county'] if self.__county is None else self.__county

    @property
    def cache_directory(self) -> str:
        directory = os.path.join(self.version.cache_directory, 'weather')
        os.makedirs(directory, exist_ok=True)

        return directory
    
    @county.setter
    def county(self, value: str):
        self.__county = value

    def get(self, year: int = None) -> Tuple[str, str]:
        county = self.county
        epw = None
        ddy = None

        if year is not None or self.version.weather_data.startswith('amy'):
            year = self.version.weather_data.replace('amy', '') if year is None else year
            year = int(year)
            epw_cache_filepath = os.path.join(self.cache_directory, f'{county.lower()}_amy{year}.epw')
            
            if os.path.isfile(epw_cache_filepath):
                with open(epw_cache_filepath, 'r') as f:
                    epw = f.read()
            
            else:
                epw = self.__get_amy_weather(year)

                if self.cache:
                    with open(epw_cache_filepath, 'w') as f:
                        f.write(epw)

        elif self.version.weather_data.startswith('tmy3'):
            epw_cache_filepath = os.path.join(self.cache_directory, f'{county.lower()}_tmy3.epw')
            ddy_cache_filepath = os.path.join(self.cache_directory, f'{county.lower()}_tmy3.ddy')

            if os.path.isfile(epw_cache_filepath) and os.path.isfile(ddy_cache_filepath):
                with open(epw_cache_filepath, 'r') as f:
                    epw = f.read()

                with open(ddy_cache_filepath, 'r') as f:
                    ddy = f.read()
            
            else:
                epw, ddy = self.__get_tmy3_weather()

                if self.cache:
                    with open(epw_cache_filepath, 'w') as f:
                        f.write(epw)

                    with open(ddy_cache_filepath, 'w') as f:
                        f.write(ddy)   

        return epw, ddy
    
    def convert_epw_to_csv(self, epw: str) -> pd.DataFrame:
        columns = [
            'Year',
            'Month',
            'Day',
            'Hour',
            'Minute',
            'Data Source and Uncertainty Flags',
            'Dry Bulb Temperature (C)',
            'Dew Point Temperature (C)',
            'Relative Humidity (%)',
            'Atmospheric Station Pressure (Pa)',
            'Extraterrestrial Horizontal Radiation (Wh/m2)',
            'Extraterrestrial Direct Normal Radiation (Wh/m2)',
            'Horizontal Infrared Radiation Intensity (Wh/m2)',
            'Global Horizontal Radiation (Wh/m2)',
            'Direct Normal Radiation (Wh/m2)',
            'Diffuse Horizontal Radiation (Wh/m2)',
            'Global Horizontal Illuminance (lux)',
            'Direct Normal Illuminance (lux)',
            'Diffuse Horizontal Illuminance (lux)',
            'Zenith Luminance (Cd/m2)',
            'Wind Direction (Degrees)',
            'Wind Speed (m/s)',
            'Total Sky Cover (Tenths of Coverage. 1 is 1/10 coverage. 10 is full coverage)',
            'Opaque Sky Cover (Tenths of Coverage. 1 is 1/10 coverage. 10 is full coverage)',
            'Visibility (km)',
            'Ceiling Height (m)',
            'Present Weather Observation',
            'Present Weather Codes',
            'Precipitable Water (mm)',
            'Aerosol Optical Depth (thousandths)', 'Snow Depth (cm)',
            'Days Since Last Snowfall (Days)', 'Albedo (unit less)',
            'Liquid Precipitation Depth (mm)',
            'Liquid Precipitation Quantity (hr)'
        ]
        data = pd.read_csv(io.StringIO(epw), skiprows=8, header=None, names=columns)
        data.loc[data['Hour']==24, 'Hour'] = 0
        data.loc[data['Minute']==60, 'Minute'] = 0
        years = data['Year'].unique()
        year = (2019 if data.shape[0] == 8760 else 2020) if len(years) > 1 else years[0]
        data['Timestamp'] = data.apply(lambda x: f'{year}-{x["Month"]}-{x["Day"]} {x["Hour"]}:{x["Minute"]}',axis=1)
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        data = data.set_index('Timestamp', drop=True, verify_integrity=True)

        return data

    def get_version_csv(self) -> pd.DataFrame:
        suffix = self.version.weather_data.strip('amy')
        relative_path = os.path.join('weather', self.version.weather_data, f"{self.county}_{suffix}.csv")
        csvd = CSVData(relative_path=relative_path, version=self.version, cache=self.cache)
        data = csvd.get()
        data['date_time'] = pd.to_datetime(data['date_time'])

        return data
    
    def __get_amy_weather(self, year: int = None) -> str:
        metadata = self.get_metadata()
        params = {
            'start': f'{year}0101',
            'end': f'{year}1231',
            'latitude': metadata['in.weather_file_latitude'],
            'longitude': metadata['in.weather_file_longitude'],
            'community': 're',
            'parameters': ['ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_SW_DIFF', 'ALLSKY_SFC_SW_DWN', 'T2M', 'RH2M', 'WS2M'],
            'format': 'epw',
            'time-standard': 'lst',
            # 'site-elevation': FileHandler.get_settings()['general']['location']['elevation'],
        }
        response = requests.get(self.nasa_power_weather_url, params=params)
        epw = response.text
        
        return epw
    
    def __get_tmy3_weather(self) -> Tuple[str, str]:
        weather_metadata = self.__get_energyplus_weather_metadata()
        weather_metadata = weather_metadata[weather_metadata['provider']=='TMY3'].copy()
        metadata = self.get_metadata()

        if self.version.dataset_type == VersionDatasetType.RESSTOCK.value:
            longitude = metadata['in.weather_file_longitude']
            latitude = metadata['in.weather_file_latitude']

        else:
            resstock_metadata = Metadata(Version(
                VersionDatasetType.RESSTOCK.value,
                self.version.weather_data,
                self.version.year_of_publication,
                self.version.release,
            )).get({'in.resstock_county_id': [metadata['in.resstock_county_id']]})
            longitude = resstock_metadata.iloc[0]['in.weather_file_longitude']
            latitude = resstock_metadata.iloc[0]['in.weather_file_latitude']

        weather_metadata = weather_metadata[
            (weather_metadata['longitude'].astype(str)==longitude) 
            & (weather_metadata['latitude'].astype(str)==latitude)
        ].copy()
        
        if weather_metadata.shape[0] == 0:
            raise Exception(f'TMY3 data not found for bldg_id: {self.bldg_id} with expected weather file: {metadata["in.weather_file_tmy3"]}')

        elif weather_metadata.shape[0] > 1:
            found = sorted(weather_metadata["title"].unique().tolist())
            raise Exception(f'TMY3 data for bldg_id: {self.bldg_id} is ambiguous. Expected weather file: {metadata["in.weather_file_tmy3"]}, found: {found}')
        
        else:
            weather_metadata = weather_metadata.iloc[0].to_dict()
            session = requests.Session()
            retries = Retry(total=5, backoff_factor=1)
            session.mount('http://', HTTPAdapter(max_retries=retries))
            urllib3.disable_warnings()
            epw = session.get(weather_metadata['epw_url']).content.decode()
            ddy = session.get(weather_metadata['ddy_url']).content.decode(encoding='windows-1252')

        return epw, ddy
    
    def __get_energyplus_weather_metadata(self) -> pd.DataFrame:
        response = requests.get(self.energy_plus_weather_url)
        response = response.json()
        features = response['features']
        records = []

        for feature in features:
            title = feature['properties']['title']
            epw_url = BeautifulSoup(feature['properties']['epw'], 'html.parser').find('a')['href']
            ddy_url = BeautifulSoup(feature['properties']['ddy'], 'html.parser').find('a')['href']
            longitude, latitude = tuple(feature['geometry']['coordinates'])
            region = epw_url.split('/')[3]
            country = epw_url.split('/')[4]
            state = epw_url.split('/')[5]
            station_id = title.split('.')[-1].split('_')[0]
            provider = title.split('.')[-1].split('_')[-1]

            records.append({
                'title': title,
                'region': region,
                'country': country,
                'state': state,
                'station_id': station_id,
                'provider': provider,
                'epw_url': epw_url,
                'ddy_url': ddy_url,
                'longitude': longitude,
                'latitude': latitude,
            })

        data = pd.DataFrame(records)
        
        return data