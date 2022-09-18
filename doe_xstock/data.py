from datetime import datetime, timedelta
import ssl
from meteostat import Stations
from meteostat import Daily, Hourly, Monthly, units
import pandas as pd

class MeteostatWeather:
    def __init__(self, station_ids, weather_variables=None, resolution=None, earliest_start_timestamp=None, latest_end_timestamp=None, model=None, gap_limit=None):
        self.__unit_system = units.scientific
        self.station_ids = station_ids
        self.weather_variables = weather_variables
        self.__resolution_constructor = None
        self.resolution = resolution
        self.earliest_start_timestamp = earliest_start_timestamp
        self.latest_end_timestamp = latest_end_timestamp
        self.model = model
        self.gap_limit = gap_limit
        self.__set_ssl()

    @property
    def station_ids(self):
        return self.__station_ids

    @property
    def weather_variables(self):
        return self.__weather_variables

    @property
    def resolution(self):
        return self.__resolution

    @property
    def earliest_start_timestamp(self):
        return self.__earliest_start_timestamp

    @property
    def latest_end_timestamp(self):
        if self.__latest_end_timestamp is None:
            timestamp = pd.to_datetime(datetime.now() - timedelta(days=1))
            timestamp = timestamp.normalize()
        else:
            timestamp = pd.to_datetime(self.__latest_end_timestamp)

        return timestamp

    @property
    def model(self):
        return self.__model

    @property
    def gap_limit(self):
        return self.__gap_limit

    @station_ids.setter
    def station_ids(self, station_ids):
        self.__station_ids = station_ids

    @weather_variables.setter
    def weather_variables(self, weather_variables):
        self.__weather_variables = weather_variables
    
    @resolution.setter
    def resolution(self, resolution):
        self.__resolution = 'hourly' if resolution is None else resolution
        self.__resolution_constructor = {'hourly':Hourly, 'daily':Daily, 'monthly':Monthly}[self.resolution]
        self.__resolution_constructor.max_age = 0 # no caching

    @earliest_start_timestamp.setter
    def earliest_start_timestamp(self, earliest_start_timestamp):
        if earliest_start_timestamp is None:
            self.__earliest_start_timestamp = pd.to_datetime('2009-01-01')
        else:
            self.__earliest_start_timestamp = pd.to_datetime(earliest_start_timestamp)

    @latest_end_timestamp.setter
    def latest_end_timestamp(self, latest_end_timestamp):
        self.__latest_end_timestamp = latest_end_timestamp

    @model.setter
    def model(self, model):
        self.__model = False if model is None else model

    @gap_limit.setter
    def gap_limit(self, gap_limit):
        self.__gap_limit = 3 if gap_limit is None else gap_limit

    def download(self,**kwargs):
        data_list = []

        for station_id in self.station_ids:
            kwargs = {
                'loc':station_id,
                'start':self.earliest_start_timestamp,
                'end':self.latest_end_timestamp,
                **kwargs
            }
            data = self.__resolution_constructor(**kwargs)

            if self.model:
                data.normalize()
                data.interpolate()
            else:
                pass

            data = data.convert(self.__unit_system)
            data = data.fetch()
            weather_variables = data.columns.tolist() if self.weather_variables is None else self.weather_variables
            data = data[weather_variables].copy()

            if data.shape[1] > 0:
                data_list.append(self.__preprocess(data,station_id))
            else:
                pass

        if len(data_list) > 0:
            return pd.concat(data_list,ignore_index=True)
        else:
            return None
        
    def __preprocess(self, data, station_id):
        data['timestamp'] = pd.to_datetime(data.index).strftime('%Y-%m-%d %H:%M:%S')
        data['station_id'] = station_id
        data['resolution'] = self.resolution
        return data

    @classmethod
    def get_station_from_coordinates(cls, latitude, longitude, radius=None, count=None):
        count = 1 if count is None else count
        stations = Stations()
        data = stations.nearby(latitude, longitude, radius=radius).fetch(count)
        return data

    @classmethod
    def convert_sea_level_pressure_to_station_pressure(cls, sea_level_pressure, elevation):
        # this method was copied from:
        # https://github.com/IMMM-SFA/diyepw/blob/f1904bcde05f29d63b91b6c2c73a4cc7fdf9103a/diyepw/create_amy_epw_file.py#L404

        # convert (or keep) pressure and elevation inputs as floats
        sea_level_pressure = float(sea_level_pressure)
        elevation = float(elevation)

        # convert from hectopascals to inHg
        Pa_inHg = sea_level_pressure * 0.029529983071445

        # calculate station pressure according to formula from https://www.weather.gov/epz/wxcalc_stationpressure
        Pstn_inHg = Pa_inHg * ((288 - 0.0065*elevation)/288)**5.2561

        # convert from inHg to Pa
        Pstn = Pstn_inHg * 3386.389

        return Pstn

    def __set_ssl(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context