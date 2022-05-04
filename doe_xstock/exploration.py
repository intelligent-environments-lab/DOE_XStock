import concurrent.futures
from datetime import datetime
import logging
import logging.config
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytz
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from doe_xstock.database import SQLiteDatabase
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class MetadataClustering:
    def __init__(self,database_filepath,name,maximum_n_clusters,minimum_n_clusters=None,filters=None,seed=None):
        self.database_filepath = database_filepath
        self.name = name
        self.maximum_n_clusters = maximum_n_clusters
        self.minimum_n_clusters = minimum_n_clusters
        self.filters = filters
        self.seed = seed
        self.__database = None
        self.__set_database()

    @property
    def database_filepath(self):
        return self.__database_filepath

    @property
    def name(self):
        return self.__name

    @property
    def maximum_n_clusters(self):
        return self.__maximum_n_clusters

    @property
    def minimum_n_clusters(self):
        return self.__minimum_n_clusters

    @property
    def seed(self):
        return self.__seed

    @property
    def filters(self):
        return self.__filters

    @database_filepath.setter
    def database_filepath(self,database_filepath):
        self.__database_filepath = database_filepath

    @name.setter
    def name(self,name):
        self.__name = name

    @maximum_n_clusters.setter
    def maximum_n_clusters(self,maximum_n_clusters):
        self.__maximum_n_clusters = maximum_n_clusters

    @minimum_n_clusters.setter
    def minimum_n_clusters(self,minimum_n_clusters):
        self.__minimum_n_clusters = 2 if minimum_n_clusters is None else minimum_n_clusters

    @seed.setter
    def seed(self,seed):
        self.__seed = 0 if seed is None else seed

    @filters.setter
    def filters(self,filters):
        self.__filters = filters

    def cluster(self):
        query = f"""
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS metadata_clustering (
            id INTEGER NOT NULL,
            name TEXT NOT NULL,
            reference_timestamp TEXT NOT NULL,
            n_clusters INTEGER NOT NULL,
            sse REAL NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (name, n_clusters)
        );
        CREATE TABLE IF NOT EXISTS metadata_clustering_label (
            id INTEGER NOT NULL,
            clustering_id INTEGER NOT NULL,
            metadata_id INTEGER NOT NULL,
            label INTEGER NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY (clustering_id) REFERENCES metadata_clustering (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            UNIQUE (clustering_id, metadata_id)
        );
        DELETE FROM metadata_clustering WHERE name = '{self.name}';
        """
        self.__database.query(query)
        metadata = self.get_metadata()
        scaler = MinMaxScaler()
        scaler = scaler.fit(metadata.values)
        metadata[metadata.columns.tolist()] = scaler.transform(metadata.values)
        x = metadata.values
        n_clusters_list = list(range(self.minimum_n_clusters,self.maximum_n_clusters + 1))
        x_list = [x for _ in n_clusters_list]
        work_order = [x_list,n_clusters_list]
        reference_timestamp = datetime.now(tz=pytz.utc).replace(microsecond=0)
        LOGGER.debug(f"Started clustering for name:{self.name} and reference_timestamp:{reference_timestamp}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.fit_kmeans,*work_order)

            for i, r in enumerate(results):
                n_clusters, labels, sse = r
                query1 = f"""
                INSERT INTO metadata_clustering (name, reference_timestamp, n_clusters, sse)
                VALUES (
                    '{self.name}', '{reference_timestamp}', {n_clusters}, {sse}
                );
                """
                values = [{'metadata_id':m,'label':l} for m, l in zip(metadata.index, labels)]
                query2 = f"""
                INSERT INTO metadata_clustering_label (clustering_id, metadata_id, label)
                VALUES (
                    (SELECT id FROM metadata_clustering WHERE name  = '{self.name}' AND n_clusters = {n_clusters}),
                    :metadata_id,
                    :label
                );
                """
                self.__database.query(query1)
                self.__database.insert_batch([query2],[values])
                LOGGER.debug(f"Clustered for name:{self.name}, reference_timestamp:{reference_timestamp}, n_clusters:{n_clusters}")

        LOGGER.debug(f"Ended clustering for name:{self.name} and reference_timestamp:{reference_timestamp}")

    def fit_kmeans(self,x,n_clusters):
        result =KMeans(n_clusters,random_state=self.seed).fit(x)
        return n_clusters, result.labels_.tolist(), result.inertia_

    def get_metadata(self):
        categorical_fields = [
            'in_vintage',
            'in_orientation',
            'in_occupants',
            'in_infiltration',
            'in_insulation_ceiling',
            'in_insulation_slab',
            'in_insulation_wall',
        ]
        numeric_fields = [
            'in_sqft',
            ('in_window_area_ft_2','/in_wall_area_above_grade_exterior_ft_2','wwr'),
            'out_electricity_cooling_energy_consumption_intensity',  
            'out_electricity_heating_energy_consumption_intensity',
            'out_electricity_water_systems_energy_consumption_intensity',
            'out_site_energy_total_energy_consumption_intensity',
        ]
        metadata_fields = categorical_fields + numeric_fields
        metadata_fields_query = [
            c if isinstance(c,str) else f'{c[0]}{c[1]} AS {c[2] if len(c) == 3 else c[0]}' 
            for c in metadata_fields
        ]
        metadata_fields = [c if isinstance(c,str) else c[2] if len(c) == 3 else c[0] for c in metadata_fields]
        separator = ',\n'
        where_clause = None if self.filters is None else ' AND '.join([f'{k} IN {tuple(v)}' for k,v in self.filters.items()])
        where_clause = '' if where_clause is None else f'WHERE {where_clause}'
        query = f"""
        SELECT
            bldg_id,
            {separator.join(metadata_fields_query)}
        FROM metadata 
        {where_clause}
        """
        metadata = self.__database.query_table(query).set_index('bldg_id')
        return self.preprocess_metadata(metadata)

    def preprocess_metadata(self,metadata):
        # convert discrete variables to continuous
        # vintage
        # - drop buildings pre-1940s and convert to integer
        metadata = metadata[metadata['in_vintage']!='<1940'].copy()
        metadata['in_vintage'] = metadata['in_vintage'].str[0:-1].astype(int)

        # orientation
        # - cosine transformation
        order = ['North','Northeast','East','Southeast','South','Southwest','West','Northwest']
        metadata['in_orientation'] = metadata['in_orientation'].map(lambda x: order.index(x))
        metadata['in_orientation_hour_sin'] = np.sin(2 * np.pi * metadata['in_orientation']/(len(order)-1))
        metadata['in_orientation_hour_cos'] = np.cos(2 * np.pi * metadata['in_orientation']/(len(order)-1))
        metatada = metadata.drop(columns=['in_orientation'])

        # occupants
        metadata['in_occupants'] = metadata['in_occupants'].astype(int)

        # infiltration
        metadata['in_infiltration'] = metadata['in_infiltration'].replace(regex=r' ACH50', value='')
        metadata['in_infiltration'] = pd.to_numeric(metadata['in_infiltration'],errors='coerce')
        metadata['in_infiltration'] = metadata['in_infiltration'].fillna(0)

        # insulation ceiling
        metadata['in_insulation_ceiling'] = metadata['in_insulation_ceiling'].replace(regex=r'R-', value='')
        metadata['in_insulation_ceiling'] = pd.to_numeric(metadata['in_insulation_ceiling'],errors='coerce')
        metadata['in_insulation_ceiling'] = metadata['in_insulation_ceiling'].fillna(0)

        # insulation slab
        metadata['in_insulation_slab'] = metadata['in_insulation_slab'].str.split(' ',expand=True)[1]
        metadata['in_insulation_slab'] = metadata['in_insulation_slab'].replace(regex=r'R', value='')
        metadata['in_insulation_slab'] = pd.to_numeric(metadata['in_insulation_slab'],errors='coerce')
        metadata['in_insulation_slab'] = metadata['in_insulation_slab'].fillna(0)

        # insulation wall
        metadata['in_insulation_wall'] = metadata['in_insulation_wall'].replace(regex=r'.+ R-', value='')
        metadata['in_insulation_wall'] = pd.to_numeric(metadata['in_insulation_wall'],errors='coerce')
        metadata['in_insulation_wall'] = metadata['in_insulation_wall'].fillna(0)

        return metadata

    def __set_database(self):
        self.__database = SQLiteDatabase(self.database_filepath)

    @classmethod
    def run(cls,filepath,name,maximum_n_clusters,minimum_n_clusters=None,filters=None,seed=None):
        mc = cls(filepath,name,maximum_n_clusters,minimum_n_clusters=minimum_n_clusters,filters=filters,seed=seed)
        mc.cluster()