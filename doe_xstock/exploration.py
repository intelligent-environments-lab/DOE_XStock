import concurrent.futures
from datetime import datetime
import logging
import logging.config
import math
import os
from pathlib import Path
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytz
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from doe_xstock.database import SQLiteDatabase
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),Path('misc/logging_config.json')))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('doe_xstock_a')

class MetadataClustering:
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    
    def __init__(self,database_filepath,dataset,name,maximum_n_clusters,figure_filepath=None,minimum_n_clusters=None,filters=None,sample_count=None,seed=None):
        self.database_filepath = database_filepath
        self.dataset = dataset
        self.name = name
        self.maximum_n_clusters = maximum_n_clusters
        self.minimum_n_clusters = minimum_n_clusters
        self.figure_filepath = figure_filepath
        self.filters = filters
        self.sample_count = sample_count
        self.seed = seed
        self.__database = self.__get_database(self.database_filepath)

    @property
    def database_filepath(self):
        return self.__database_filepath

    @property
    def dataset(self):
        return self.__dataset

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
    def figure_filepath(self):
        return self.__figure_filepath

    @property
    def sample_count(self):
        return self.__sample_count

    @property
    def seed(self):
        return self.__seed

    @property
    def filters(self):
        return self.__filters

    @database_filepath.setter
    def database_filepath(self,database_filepath):
        self.__database_filepath = database_filepath

    @dataset.setter
    def dataset(self,dataset):
        self.__dataset = dataset

    @name.setter
    def name(self,name):
        self.__name = name

    @maximum_n_clusters.setter
    def maximum_n_clusters(self,maximum_n_clusters):
        self.__maximum_n_clusters = maximum_n_clusters

    @minimum_n_clusters.setter
    def minimum_n_clusters(self,minimum_n_clusters):
        minimum_minimum_n_cluusters = 2
        self.__minimum_n_clusters = minimum_minimum_n_cluusters if minimum_n_clusters is None\
            or minimum_n_clusters < minimum_minimum_n_cluusters else minimum_n_clusters

    @figure_filepath.setter
    def figure_filepath(self,figure_filepath):
        self.__figure_filepath = 'figures' if figure_filepath is None else figure_filepath
        os.makedirs(self.__figure_filepath, exist_ok=True)

    @sample_count.setter
    def sample_count(self,sample_count):
        self.__sample_count = 100 if sample_count is None else sample_count

    @seed.setter
    def seed(self,seed):
        self.__seed = 0 if seed is None else seed

    @filters.setter
    def filters(self,filters):
        self.__filters = filters

    def cluster(self):
        query = f"""
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS metadata_clustering_name (
            id INTEGER NOT NULL,
            name TEXT NOT NULL,
            PRIMARY KEY (id),
            UNIQUE (name)
        );
        
        CREATE TABLE IF NOT EXISTS metadata_clustering (
            id INTEGER NOT NULL,
            name_id INTEGER NOT NULL,
            reference_timestamp TEXT NOT NULL,
            n_clusters INTEGER NOT NULL,
            sse REAL NOT NULL,
            silhouette_score REAL NOT NULL,
            calinski_harabasz_score REAL NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY (name_id) REFERENCES metadata_clustering_name (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            UNIQUE (name_id, n_clusters)
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

        CREATE TABLE IF NOT EXISTS optimal_metadata_clustering (
            id INTEGER NOT NULL,
            name_id INTEGER NOT NULL,
            clustering_id INTEGER NOT NULL,
            score_name TEXT NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY (name_id) REFERENCES metadata_clustering_name (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            FOREIGN KEY (clustering_id) REFERENCES metadata_clustering (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            UNIQUE (name_id),
            CHECK (score_name IN ('sse', 'silhouette_score', 'calinski_harabasz_score'))
        );
        DELETE FROM metadata_clustering_name WHERE name = '{self.name}';
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
                n_clusters, labels, sse, silhouette_score, calinski_harabasz_score  = r
                query1 = f"""
                INSERT INTO metadata_clustering_name (name)
                VALUES ('{self.name}') 
                ON CONFLICT (name) DO NOTHING;

                INSERT INTO metadata_clustering (name_id, reference_timestamp, n_clusters, sse, silhouette_score, calinski_harabasz_score)
                VALUES ((SELECT id FROM metadata_clustering_name WHERE name = '{self.name}'), '{reference_timestamp}', {n_clusters}, {sse}, {silhouette_score}, {calinski_harabasz_score});
                """
                values = [{'metadata_id':m,'label':l} for m, l in zip(metadata.index, labels)]
                query2 = f"""
                INSERT INTO metadata_clustering_label (clustering_id, metadata_id, label)
                VALUES (
                    (SELECT id FROM metadata_clustering WHERE 
                        name_id = (SELECT id FROM metadata_clustering_name WHERE name = '{self.name}') AND n_clusters = {n_clusters}
                    ),
                    :metadata_id,
                    :label
                );
                """
                self.__database.query(query1)
                self.__database.insert_batch([query2],[values])
                LOGGER.debug(f"Clustered for name:{self.name}, reference_timestamp:{reference_timestamp}, n_clusters:{n_clusters}")
        
        # set optimal clustering
        LOGGER.debug(f"Inserting optimal clustering data for name:{self.name} and reference_timestamp:{reference_timestamp}")
        score_name = 'silhouette_score'
        query = f"""
        INSERT INTO optimal_metadata_clustering (name_id, clustering_id, score_name)
        SELECT
            n.id AS name_id,
            m.id AS clustering_id,
            '{score_name}' AS score_name
        FROM metadata_clustering m
        LEFT JOIN (
            SELECT
                name_id,
                MAX({score_name}) AS {score_name}
            FROM metadata_clustering
            GROUP BY
                name_id
        ) s ON s.name_id = m.name_id
        LEFT JOIN metadata_clustering_name n ON n.id = m.name_id
        WHERE m.{score_name} = s.{score_name} AND n.name = '{self.name}'
        ;"""
        self.__database.query(query)

        # sample buildings for E+ simulation
        LOGGER.debug(f"Sampling {self.sample_count} buildings for name:{self.name} and reference_timestamp:{reference_timestamp}")
        self.set_buildings_to_simulate(self.name, self.database_filepath, self.sample_count, self.seed)

        # plot figures
        LOGGER.debug(f"Plotting figures for name:{self.name} and reference_timestamp:{reference_timestamp}")
        n_clusters = self.__database.query_table(f"""
        SELECT
            c.n_clusters
        FROM metadata_clustering c
        WHERE id = (
            SELECT 
                clustering_id 
            FROM optimal_metadata_clustering 
            WHERE name_id = (SELECT id FROM metadata_clustering_name WHERE name = '{self.name}')
        )
        """).iloc[0]['n_clusters']
        filename = self.name.lower().replace(' ','_').replace(',','')
        self.plot_scores(self.name,self.database_filepath,figure_filepath=os.path.join(self.figure_filepath,f'{filename}_metadata_clustering_scores.png'))
        self.plot_sample_count(self.name,n_clusters,self.database_filepath,figure_filepath=os.path.join(self.figure_filepath,f'{filename}_metadata_clustering_sample_count.png'))
        self.plot_ground_truth(self.name,n_clusters,self.database_filepath,figure_filepath=os.path.join(self.figure_filepath,f'{filename}_metadata_clustering_ground_truth.png'))

        LOGGER.debug(f"Ended clustering for name:{self.name} and reference_timestamp:{reference_timestamp}")

    @classmethod
    def set_buildings_to_simulate(cls, name, database_filepath, count=100, seed=0):
        data = cls.__get_database(database_filepath).query_table(f"""
        SELECT
            r.metadata_id,
            r.label
        FROM optimal_metadata_clustering o
        LEFT JOIN metadata_clustering_label r ON r.clustering_id = o.clustering_id
        LEFT JOIN metadata_clustering_name n ON n.id = o.name_id
        WHERE n.name = '{name}'
        ORDER BY
            r.label ASC,
            r.metadata_id ASC
        """)
        data['count'] = data.groupby('label')['label'].transform('count')
        count = min(data.shape[0], count)
        data = data.sample(n=count, weights=data['count'], random_state=seed)
        cls.__get_database(database_filepath).query(f"""
        CREATE TABLE IF NOT EXISTS building_to_simulate_in_energyplus (
            id INTEGER NOT NULL,
            name_id INTEGER NOT NULL,
            metadata_id INTEGER NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY (name_id) REFERENCES metadata_clustering_name (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            FOREIGN KEY (metadata_id) REFERENCES metadata (id)
                ON DELETE CASCADE
                ON UPDATE CASCADE,
            UNIQUE (name_id, metadata_id)
        );
        DELETE FROM building_to_simulate_in_energyplus WHERE name_id = (SELECT id FROM metadata_clustering_name WHERE name = '{name}');
        """)
        query = f"""
        INSERT INTO building_to_simulate_in_energyplus (name_id, metadata_id)
            VALUES ((SELECT id FROM metadata_clustering_name WHERE name = '{name}'), :metadata_id)
        ;
        """
        cls.__get_database(database_filepath).insert_batch([query], [data.to_dict('records')])

    @classmethod
    def plot_ground_truth(cls,name,n_clusters,database_filepath,figure_filepath=None):
        categorical_fields, numeric_fields = cls.__get_fields()
        metadata_fields = categorical_fields + numeric_fields
        metadata_fields_query, metadata_fields = cls.__transform_field_names(metadata_fields)
        ground_truth_fields = [
            'in_bedrooms', 'in_clothes_dryer', 'in_clothes_washer', 'in_cooking_range',
            'in_dishwasher', 'in_ducts', 
            'in_geometry_floor_area', 'in_geometry_foundation_type', 'in_geometry_stories', 
            'in_geometry_wall_exterior_finish', 'in_geometry_wall_type', 
            'in_geometry_wall_type_and_exterior_finish', 'in_heating_fuel',
            'in_hvac_cooling_efficiency', 'in_hvac_cooling_type', 'in_hvac_heating_efficiency',
            'in_hvac_heating_type', 'in_hvac_heating_type_and_fuel', 
            'in_hvac_secondary_heating_efficiency', 'in_lighting',
            'in_misc_extra_refrigerator',
            'in_plug_load_diversity',  'in_refrigerator', 'in_resstock_puma_id',
            'in_usage_level', 'in_water_heater_efficiency', 
            'in_water_heater_fuel', 'in_window_areas', 'in_windows',
        ]
        ground_truth_query = metadata_fields_query + ground_truth_fields
        separator = ',\n'
        data = cls.__get_database(database_filepath).query_table(f"""
        SELECT
            e.id,
            c.label,
            {separator.join(ground_truth_query)}
        FROM metadata e
        INNER JOIN metadata_clustering_label c ON c.metadata_id = e.id
        LEFT JOIN metadata_clustering n ON n.id = c.clustering_id
        LEFT JOIN metadata_clustering_name m ON m.id = n.name_id
        WHERE m.name = '{name}' AND n.n_clusters = {n_clusters}
        """).set_index('id')
        column_limit = 3
        fields = metadata_fields + ground_truth_fields
        cmaps = ['RdBu' for _ in metadata_fields] + ['RdBu_r' for _ in ground_truth_fields]
        row_count = math.ceil(len(fields)/column_limit)
        column_count = min(column_limit,len(fields))
        fig, _ = plt.subplots(row_count,column_count,figsize=(6*column_count,2.5*row_count))
        divnorm = colors.TwoSlopeNorm(vcenter=0)

        for ax, cmap, field in zip(fig.axes, cmaps, fields):
            plot_data = data[['label',field]].copy()

            if pd.api.types.is_numeric_dtype(plot_data[field]):
                plot_data[field] = pd.cut(plot_data[field],6)
                plot_data[field] = plot_data[field].astype(str)
            else:
                pass

            plot_data = plot_data[['label',field]].groupby(['label',field]).size().reset_index(name='count')
            plot_data = plot_data.pivot(index=field,columns='label',values='count')
            plot_data = plot_data.fillna(0)
            x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap=cmap,edgecolors='white',linewidth=0)
            _ = fig.colorbar(cm.ScalarMappable(cmap=cmap,norm=divnorm),ax=ax,orientation='vertical',label='Count',fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=0)
            ax.set_ylabel('Option')
            ax.set_xlabel('Cluster')
            ax.set_title(field)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        
        plt.tight_layout()
        figure_filepath = f'{name}_metadata_clustering_ground_truth.png' if figure_filepath is None else figure_filepath
        plt.savefig(figure_filepath,facecolor='white',bbox_inches='tight')
        plt.close()

    @classmethod
    def plot_sample_count(cls,name,n_clusters,database_filepath,figure_filepath=None):
        data = cls.__get_database(database_filepath).query_table(f"""
        SELECT
            c.label,
            COUNT(c.metadata_id) AS count
        FROM metadata_clustering_label c
        LEFT JOIN metadata_clustering n ON n.id = c.clustering_id
        LEFT JOIN metadata_clustering_name m ON m.id = n.name_id
        WHERE m.name = '{name}' AND n.n_clusters = {n_clusters}
        GROUP BY c.label
        ORDER BY c.label
        """)
        _, ax = plt.subplots(1,1,figsize=(6,2))
        x, y = data['label'], data['count']
        ax.bar(x,y)
        ax.set_xlabel('cluster label')
        ax.set_ylabel('count')
        ax.set_title(name)
        figure_filepath = f'{name}_metadata_clustering_sample_count.png' if figure_filepath is None else figure_filepath
        plt.savefig(figure_filepath,facecolor='white',bbox_inches='tight')
        plt.close()

    @classmethod
    def plot_scores(cls,name,database_filepath,figure_filepath=None):
        data = cls.__get_database(database_filepath).query_table(f"""
        SELECT
            n.name,
            c.n_clusters,
            c.sse,
            c.silhouette_score,
            c.calinski_harabasz_score
        FROM metadata_clustering c
        LEFT JOIN metadata_clustering_name n ON n.id = c.name_id
        WHERE name = '{name}'
        ORDER BY c.n_clusters
        """)
        scores = data.columns.tolist()[2:]
        row_count = 1
        columns_count = len(scores)
        fig, ax = plt.subplots(row_count,columns_count,figsize=(4*columns_count,2*row_count))

        for ax, score in zip(fig.axes, scores):
            x, y = data['n_clusters'], data[score]
            ax.plot(x,y)
            ax.set_title(name)
            ax.set_xlabel('n_clusters')
            ax.set_ylabel(score)

        plt.tight_layout()
        figure_filepath = f'{name}_metadata_clustering_scores.png' if figure_filepath is None else figure_filepath
        plt.savefig(figure_filepath,facecolor='white',bbox_inches='tight')
        plt.close()

    def fit_kmeans(self,x,n_clusters):
        result =KMeans(n_clusters,random_state=self.seed).fit(x)
        return n_clusters, result.labels_.tolist(), result.inertia_, silhouette_score(x,result.labels_), calinski_harabasz_score(x,result.labels_)

    def get_metadata(self):
        categorical_fields, numeric_fields = self.__get_fields()
        metadata_fields = categorical_fields + numeric_fields
        metadata_fields_query, metadata_fields = self.__transform_field_names(metadata_fields)
        separator = ',\n'
        where_clause = f"""
        WHERE dataset_id = (
            SELECT 
                id 
            FROM dataset 
            WHERE 
                dataset_type = '{self.dataset['dataset_type']}'
                AND weather_data = '{self.dataset['weather_data']}'
                AND year_of_publication = {self.dataset['year_of_publication']}
                AND release = {self.dataset['release']}
        )
        """
        where_clause += '' if self.filters is None else ' AND ' + ' AND '.join([f'{k} IN {tuple(v)}' for k,v in self.filters.items()])
        query = f"""
        SELECT
            id,
            {separator.join(metadata_fields_query)}
        FROM metadata 
        {where_clause}
        """
        metadata = self.__database.query_table(query).set_index('id')
        metadata = self.preprocess_metadata(metadata)

        return metadata

    @classmethod
    def __transform_field_names(cls,fields):
        fields_query = [
            c if isinstance(c,str) else f'{c[0]}{c[1]} AS {c[2] if len(c) == 3 else c[0]}' 
            for c in fields
        ]
        fields = [c if isinstance(c,str) else c[2] if len(c) == 3 else c[0] for c in fields]
        return fields_query, fields

    @classmethod
    def __get_fields(cls):
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
            ('in_window_area_ft_2','/in_wall_area_above_grade_exterior_ft_2','wwr'),
            'out_site_energy_total_energy_consumption_intensity',
        ]
        return categorical_fields, numeric_fields

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
        metadata = metadata.drop(columns=['in_orientation'])

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

    @classmethod
    def __get_database(cls,filepath):
        return SQLiteDatabase(filepath)