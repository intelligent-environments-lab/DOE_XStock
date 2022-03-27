import concurrent.futures
import itertools
import logging
import logging.handlers
import os
import math
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
from saxpy.sax import ts_to_string
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, minmax_scale
from doe_xstock.database import SQLiteDatabase
from doe_xstock.utilities import read_json, split_lines, write_json

 # logger
LOGGER = logging.getLogger()

'''OBJECTIVE: 
The objective is to identify building energy use diversity in the RESSTOCK dataset in a data-driven way to help with developing 
neighborhoods that can be used as simulation datasets in CityLearn. Building load and occupancy schedules/profiles drive the energy use of a building.
The building envelope also affects the sensible and latent heat gain/loss from/to outdoor environment which in turn affects energy use. If we can find out
buildings with similar occupancy schedules and similar susceptibility to heat gains and losses, then we can use such building groups as selection pools
for diversified energy use buildings. Thus the approach here is to use unsupervised modeling techniques to identify the clusters of buildings with similar energy use characteristics. 

The NREL RESSTOCK dataset provides load and occupancy schedules for a period of 1 year at 15 mins intervals. Here we cluster those each schedule using high-dimensional clustering techniques e.g. dimensionality reduction using SAX and PCA such that each building's schedule is a datapoint that can fall
within a schedule cluster. When repeated for all load schedules, we form a cluster label matrix for each building such that each column is the building's 
cluster assignment for the clustering structure of a specific schedule. 

The NREL RESSTOCK dataset also provides building envelope metadata which are factors that influence heat gain and loss. We identify useful metadata that can
be used as variables in a categorical clustering analysis of the buildings.

The cluster matrix gotten from schedule clustering can then be merged with the 1-D cluster matrix gotten from clustering the metadata to create a complete cluster matrix that represents the load and envelope factors that affect a building's energy use. A final round of clustering is then carried out on
the cluster label matrix to form building clusters that are used as selection pools for neighborhoods.'''

class Analysis:
    def __init__(self):
        # CONSTANTS & VARIABLES ****************************************************************
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        self.DATABASE_FILEPATH = '/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db'
        self.DATABASE = SQLiteDatabase(self.DATABASE_FILEPATH)
        self.FIGURES_DIRECTORY = '../occupants_analysis'
        self.MAX_FIGURE_WIDTH = 10
        self.SCHEDULE_DATA_DIRECTORY = '../occupants_schedule_data'
        self.SCHEDULE_PCA_DATA_DIRECTORY = self.FIGURES_DIRECTORY
        self.SCHEDULE_SAX_DATA_DIRECTORY = self.FIGURES_DIRECTORY
        self.SCHEDULE_CLUSTER_DATA_DIRECTORY = self.FIGURES_DIRECTORY
        self.schedule_data_filepaths = [
            os.path.join(self.SCHEDULE_DATA_DIRECTORY,f) for f in os.listdir(self.SCHEDULE_DATA_DIRECTORY) if f.endswith('.pkl')
        ]
        self.schedule_data_filepaths = sorted(self.schedule_data_filepaths)
        self.SEASONS = {
            1:'winter',2:'winter',12:'winter',
            3:'spring',4:'spring',5:'spring',
            6:'summer',7:'summer',8:'summer',
            9:'fall',10:'fall',11:'fall',
        }
        self.PCA_CUMMULATIVE_VARIANCE_THRESHOLD = 0.95
        self.SAX_A = list(range(2,11))
        self.SAX_CUTS = {a:cuts_for_asize(a) for a in self.SAX_A}
        daily_timesteps = int(24*4)
        self.SAX_W = [w for w in list(range(2,daily_timesteps+1)) if daily_timesteps%w == 0]
        self.DATE_RANGE = pd.DataFrame({'timestamp':pd.date_range('2017-01-01','2017-12-31 23:45:00', freq=f'{int(60*15)}S')})
        self.DATE_RANGE['timestep'] = self.DATE_RANGE.index
        self.DATE_RANGE['month'] = self.DATE_RANGE['timestamp'].dt.month
        self.DATE_RANGE['week'] = self.DATE_RANGE['timestamp'].dt.isocalendar().week
        self.DATE_RANGE['date'] = self.DATE_RANGE['timestamp'].dt.normalize()
        self.DATE_RANGE['day_of_week'] = self.DATE_RANGE['timestamp'].dt.weekday
        self.DATE_RANGE.loc[self.DATE_RANGE[
            'day_of_week'] == 6, 'week_of'
        ] = self.DATE_RANGE.loc[self.DATE_RANGE['day_of_week'] == 6]['timestamp'].dt.normalize()
        self.DATE_RANGE['week_of'] = self.DATE_RANGE['week_of'].ffill()
        self.DATE_RANGE['season'] = self.DATE_RANGE['month'].map(lambda x: self.SEASONS[x])
        self.LOG_FILEPATH = '../output.log'
        self.LOG_MAX_BYTES = 1E8
        self.LOG_BACKUP_COUNT = 0
        self.set_logger()
       
    def run(self):
        try:
            LOGGER.info('Started analysis.')
            self.run_schedule_analysis()
            # self.run_building_metadata_analysis()
        except Exception as e:
            LOGGER.exception(e)
        finally:
            LOGGER.info('Finished analysis.')

    def run_building_metadata_analysis(self):
        # FIGURE ****************************************************************
        '''DESCRIPTION: What are the catagories and distribution of values in the metadata? Helps form initial understanding of diversity 
        across buildings.'''
        LOGGER.debug('Plotting building metadata.')
        self.plot_building_metadata()

    def plot_building_metadata(self):
        # # numeric metadata
        metadata = self.DATABASE.get_table('metadata')
        columns_to_exclude = [
            'id', 'bldg_id', 'dataset_id','upgrade','metadata_index','in_county','in_puma','in_ashrae_iecc_climate_zone_2004',
            'in_building_america_climate_zone', 'in_iso_rto_region','applicability', 'in_ahs_region','in_applicable','in_cec_climate_zone',
            'in_census_division','in_census_division_recs','in_census_region','in_geometry_building_type_acs','in_geometry_building_type_height',
            'in_geometry_building_type_recs','in_state','in_weather_file_longitude','in_weather_file_latitude','in_weather_file_city',
            'in_nhgis_county_gisjoin','in_state_name','in_american_housing_survey_region','in_weather_file_2018','in_weather_file_tmy3',
            'in_resstock_county_id','in_vacancy_status'
         ]
        columns = [c for c in metadata.columns if c not in columns_to_exclude and pd.api.types.is_numeric_dtype(metadata[c])]
        column_count = 6
        row_count = math.ceil(len(columns)/column_count)
        fig, _ = plt.subplots(row_count, column_count, figsize=(min(4,self.MAX_FIGURE_WIDTH)*column_count,3*row_count))

        for ax, column in zip(fig.axes, columns):
            ax.hist(metadata[column])
            ax.set_title(split_lines(column, line_character_limit=28,delimiter='_'))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,'building_numeric_metadata_histogram.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # non numeric metadata
        columns = [c for c in metadata.columns if c not in columns_to_exclude and not pd.api.types.is_numeric_dtype(metadata[c])]
        column_count = 4
        row_count = math.ceil(len(columns)/column_count)
        fig, _ = plt.subplots(row_count, column_count, figsize=(min(7,self.MAX_FIGURE_WIDTH)*column_count,6*row_count))

        for ax, column in zip(fig.axes, columns):
            plot_data = metadata.groupby(column).size().reset_index(name='count')
            x, y = list(range(plot_data.shape[0])), plot_data['count']
            ax.barh(x,y)
            ax.set_yticks(x)
            ax.set_yticklabels(plot_data[column].to_list())
            ax.set_title(split_lines(column, line_character_limit=28,delimiter='_'))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,'building_non_numeric_metadata_histogram.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def run_schedule_analysis(self):        
        # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION: Retrieve schedule data from database and write to filepath for faster I/O.'''
        # LOGGER.debug('Reading schedule data from database and saving to local file.')
        # self.save_schedule_data()

        # # FIGURE ****************************************************************
        # '''DESCRIPTION: schedule hourly box plots categorized by day of week and season to investigate variance w.r.t. the categories.'''
        # LOGGER.debug('Plotting schedule hourly boxplot.')
        # self.plot_schedule_hourly_box_plot()
        # '''NOTE:
        # __Hourly variance when categorized by day-of-week__
        # - There are distinct hourly shapes i.e. all schedules change from hour to hour.
        # - there are day-of-week variations in the schedules w.r.t. to hour in some cases.
        # - For most schedules, weekdays have similar distribution.
        # - Friday and Saturday seem to be typically distinct from other days of the week (possible that E+ sims started with Monday as first day of week not Sunday as shown in EPW file).
        # - Some hours have no variance irrespective of the day of week i.e. 1 unique value for records hence, dimensionality reduction of the daily data
        # to a smaller dimension will be feasible e.g. instead of 24 hours of data to represent a day, maybe just use hours within 8 AM - 10 PM or so.
        # Maybe PCA can help with this dimension reduction.

        # __Hourly variance when categorized by season__
        # - There are indeed seasonal variations and particularly winter season seems to have higher values than other seasons.
        # - However, within a season, there may be little to no variance.
        # - Some schedules like occupancy show the same distribution irrespective of the season but show variance when categorized by day-of-week.'''

        # # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION: exclude the following schedules as they do not provide any variance.'''
        # schedules_to_exclude = ['plug_loads_vehicle','vacancy','lighting_exterior_holiday']
        # LOGGER.debug(f'Excluding the following schedules that do not provide variance temporally nor spatially: {schedules_to_exclude}.')
        # self.schedule_data_filepaths = [f for f in self.schedule_data_filepaths if not self.schedule_name(f) in schedules_to_exclude]

        # # FIGURE ****************************************************************
        # '''DESCRIPTION: building schedule standard deviation hourly box plots categorized by day of week and season to investigate 
        # variance per building w.r.t. the categories.'''
        # LOGGER.debug('Plotting building schedule hourly std boxplot')
        # self.plot_schedule_building_hourly_std_box_plot()
        # '''NOTE:
        # __Hourly variance when categorized by day-of-week__
        # - Distinct buildings have hourly variance when hours are categorized by day-of-week as indicated by non-zero standard deviation distributions.
        # - Standard deviation tends to increase later in the day for most schedules.

        # __Hourly variance when categorized by season__
        # - Distinct buildings have hourly variance when hours are categorized by season as indicated by non-zero standard deviation distributions.
        # - Some schedules like fuel_loads_grill only have > 0 std for all buildings during fall because they have constant non-zero value during other seasons.'''

        # # FIGURE ****************************************************************
        # '''DESCRIPTION: building schedule standard deviation from general average hourly box plots categorized by day of week and season to investigate 
        # variance per building w.r.t. the categories.'''
        # LOGGER.debug('Plotting building schedule hourly std_wrt general avg boxplot')
        # self.plot_schedule_building_hourly_std_wrt_general_avg_box_plot()
        # '''NOTE:
        # - There are schedules that can be excluded from analyssi becasue there is no variance across buildings when their values are evaluated 
        # w.r.t. to a common mean value.'''

        # # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION: exclude the following schedules as they vary temporally but not spatially i.e. across buildings.'''
        # schedules_to_exclude = ['fuel_loads_fireplace','fuel_loads_grill','fuel_loads_lighting','lighting_exterior','lighting_garage','plug_loads_well_pump']
        # LOGGER.debug(f'Excluding the following schedules that vary temporally but not spatially i.e. across buildings: {schedules_to_exclude}.')
        # self.schedule_data_filepaths = [f for f in self.schedule_data_filepaths if not self.schedule_name(f) in schedules_to_exclude]

        # # FIGURE ****************************************************************
        # '''DESCRIPTION: building schedule correlation to further narrow down the feature space.'''
        # LOGGER.debug(f'Saving schedule correlation data.')
        # self.save_schedule_corr()
        # LOGGER.debug(f'Ploting schedule correlation data.')
        # self.plot_schedule_corr()
        # '''NOTE:
        # - Clothes dryer exhaust and clothes dryer have Pearson correlation = 1. 
        # - Also ligthing, plug load and ceiling fan are highly correlated. Lighting has higher correlation with the other 2 than the otehr 2 have with themselves and lighting so lighting may be used to represent all 3.'''

        # # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION: exclude the following schedules as they are highly correlated with other schedules.'''
        # schedules_to_exclude = ['clothes_dryer_exhaust','ceiling_fan','plug_loads']
        # LOGGER.debug(f'Excluding the following schedules that are highly correlated with other schedules: {schedules_to_exclude}.')
        # self.schedule_data_filepaths = [f for f in self.schedule_data_filepaths if not self.schedule_name(f) in schedules_to_exclude]

        # DATA MANIPULATION & FIGURE ****************************************************************
        '''DESCRIPTION:
        SAX transformation of the schedule time series using a window size of SAX_W for PAA and alphabet size of SAX_A for time series to word conversion.
        The idea is that SAX compresses the data to discrete words describing the scehdule profiles that can be aggregated by counts of unique words 
        per building 
        and used as clustering dataset. Essentially this reduces the width of the clustering data from the length of the time series to the
        number of unique words.'''
        LOGGER.debug('Applying SAX transformation to schedules and saving transformed data.')
        # self.save_normalized_pre_schedule_sax_data()
        # self.save_schedule_sax_data()
        # LOGGER.debug('Plotting SAX data.')
        # self.plot_schedule_sax_data()
        '''NOTE:
        - The schedules with wide variance e.g. appliance loads have about 16 unique words. 
            Occupancy and some others have the larger number of unique words but still under 100 which is much less than using the entire 
            timeseries as a selfuter datapoint.
            Maybe, PCA can be applied as a second compression to further reduce the dimensionality?
        - The heat map that shows the normalized word counts per building shows variance across the building which is a good sign that the clustering
        might pick out building groups
            (fingers crossed!).
        '''

        # # DATA MANIPULATION & FIGURE ****************************************************************
        # '''DESCRIPTION:
        # Perform PCA transformation on the per building unqiue word counts data set to reduce the feature space of
        # those schedules that are large and extract the variables that best represent the variance in the word counts.
        # The features are used in the clustering analysis to extract the different building classes.'''
        # LOGGER.debug('Applying PCA to SAX data to further reduce dimensionality.')
        # self.save_schedule_pca_data()
        # LOGGER.debug('Plotting PCA data.')
        # self.plot_schedule_pca_data()
        # '''NOTE:
        # - Able to reduce the schedules with over 60 words to about 50 words when 95% of variance is desired from PCA components.
        # - In general all schedules could use lower dimension than discovered in SAX to explain variance and use as clustering dataset.
        # '''

        # # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION:
        # Perform clustering on the PCA components extracted from SAX dataset to identify characteristic schedules.'''
        # LOGGER.debug(f'Saving PCA data that explains {self.PCA_CUMMULATIVE_VARIANCE_THRESHOLD} variance as cluster data.')
        # self.save_schedule_cluster_data()
        # LOGGER.debug(f'Plotting cluster data.')
        # self.plot_schedule_cluster_data()
        # LOGGER.debug(f'Running KMeans cluster and saving results.')
        # self.run_kmeans()
        # LOGGER.debug(f'Plotting KMeans data.')
        # self.plot_kmeans()
        # LOGGER.debug(f'Running DBSCAN cluster and saving results.')
        # self.run_dbscan()
        # LOGGER.debug(f'Plotting DBSCAN data.')
        # self.plot_dbscan()

        # # DATA MANIPULATION ****************************************************************
        # '''DESCRIPTION:
        # Building cluster grid from the results of the independent schedule clustering.'''
        # LOGGER.debug(f'Saving building cluster grid data.')
        # # self.save_building_cluster_grid()
        # LOGGER.debug(f'Plotting building cluster grid data.')
        # self.plot_building_cluster_grid()

    def plot_building_cluster_grid(self):
        plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,'building_cluster_grid.pkl'))
        plot_data = plot_data.sort_values(plot_data.columns.tolist()).reset_index(drop=True)
        
        for c in plot_data.columns:
            plot_data.loc[plot_data[c]>=0,c] += 1

        fig, ax = plt.subplots(1,1,figsize=(min(10,self.MAX_FIGURE_WIDTH),10))
        x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
        divnorm = colors.TwoSlopeNorm(vcenter=0)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.0025,clip_on=False)
        _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)
        ax.tick_params('x',which='both',rotation=45)
        ax.set_ylabel('Building')
        ax.set_xlabel('Schedule')
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'building_cluster_grid_heatmap'), facecolor='white', bbox_inches='tight')
        plt.close()

    def save_building_cluster_grid(self):
        dbscan_result = read_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'))
        labels = {}
        
        for i, (schedule, schedule_data) in enumerate(dbscan_result['schedules'].items()):
            scores = schedule_data['scores']['calinski_harabasz']
            max_score_index = scores.index(max(scores))
            labels[schedule] = schedule_data['labels'][max_score_index]

        labels['metadata_id'] = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule}.pkl')).index.tolist()
        labels = pd.DataFrame(labels).set_index('metadata_id')
        labels.to_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,'building_cluster_grid.pkl'))

    def plot_dbscan(self):          
        knn_result = read_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json'))
        dbscan_result = read_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'))

        # KNN distance
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, _ = plt.subplots(row_count, column_count, figsize=(min(6,self.MAX_FIGURE_WIDTH),3*row_count))

        for i, (ax, (schedule, schedule_data)) in enumerate(zip(fig.axes, knn_result['schedules'].items())):
            y = sorted(schedule_data['distance'])
            x = list(range(len(y)))
            ax.plot(x,y)
            ax.set_ylabel(f'distance to\nnearest neighbor')
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_xlabel('sorted index')
            ax.grid(True)
            ax.set_title(schedule)

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_knn_sorted_distance.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # DBSCAN  scores
        row_count = len(self.schedule_data_filepaths)
        column_count = len(list(dbscan_result['schedules'].values())[0]['scores'])
        fig, axs = plt.subplots(row_count, column_count, figsize=(min(5,self.MAX_FIGURE_WIDTH)*column_count,2*row_count))

        for i, (schedule, schedule_data) in enumerate(dbscan_result['schedules'].items()):
            x = schedule_data['eps']
            y2 = [len(set(l)) for l in schedule_data['labels']]

            for j, (score, y1) in enumerate(schedule_data['scores'].items()):
                axs[i,j].plot(x,y1,color='blue')
                axs[i,j].set_title(f'{schedule}')
                axs[i,j].set_xlabel('eps')
                axs[i,j].set_ylabel(score,color='blue')
                ax2 = axs[i,j].twinx()
                ax2.plot(x,y2,color='red')
                ax2.set_ylabel('clusters',color='red')
        
        plt.tight_layout()
        fig.align_ylabels()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_dbscan_scores.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # DBSCAN label against timeseries
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, axs = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            plot_data = plot_data.pivot(index='metadata_id',columns='timestep',values=self.schedule_name(filepath))
            scores = dbscan_result['schedules'][self.schedule_name(filepath)]['scores']['calinski_harabasz']
            max_score_index = scores.index(max(scores))
            labels = dbscan_result['schedules'][self.schedule_name(filepath)]['labels'][max_score_index]
            columns = plot_data.columns.tolist()
            plot_data['label'] = labels
            plot_data = plot_data.sort_values(['label']+columns).reset_index(drop=True)
            indices = plot_data.groupby(['label']).size().reset_index(name='count')['count'].tolist()
            indices = [sum(indices[0:i + 1]) for i in range(len(indices))]
            plot_data = plot_data.drop(columns=['label'])
            x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0,clip_on=False)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)

            for label, index in zip(sorted(set(labels)), indices):
                ax.axhline(index + 0.5,color='black',linestyle='--',linewidth=4,clip_on=False)
                ax.text(-80,index,f'C:{label}',color='black',fontsize=12,ha='right',va='center',fontweight='medium')

            ax.tick_params('x',which='both',rotation=0)
            ax.set_ylabel('Building')
            ax.set_xlabel('Timestep')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_building_dbscan_cluster_timeseries_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # DBSCAN label against clusted pca data
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, axs = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            scores = dbscan_result['schedules'][self.schedule_name(filepath)]['scores']['calinski_harabasz']
            max_score_index = scores.index(max(scores))
            labels = dbscan_result['schedules'][self.schedule_name(filepath)]['labels'][max_score_index]
            columns = plot_data.columns.tolist()
            plot_data['label'] = labels
            plot_data = plot_data.sort_values(['label']+columns).reset_index(drop=True)
            indices = plot_data.groupby(['label']).size().reset_index(name='count')['count'].tolist()
            indices = [sum(indices[0:i + 1]) for i in range(len(indices))]
            plot_data = plot_data.drop(columns=['label'])
            x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0,clip_on=False)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)

            for label, index in zip(sorted(set(labels)), indices):
                ax.axhline(index + 0.5,color='black',linestyle='--',linewidth=4,clip_on=False)
                ax.text(-0.5,index,f'C:{label}',color='black',fontsize=12,ha='right',va='center',fontweight='medium')

            ax.tick_params('x',which='both',rotation=0)
            ax.set_ylabel('Building')
            ax.set_xlabel('Timestep')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_building_dbscan_cluster_pca_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def run_dbscan(self):
        # Use KNN to determine eps
        with concurrent.futures.ThreadPoolExecutor() as executor:
            knn_result = {'n_neighbors':2,'schedules':{}}
            work_order = [self.schedule_data_filepaths,[knn_result['n_neighbors']]*len(self.schedule_data_filepaths)]
            results = executor.map(self.fit_knn,*work_order)

            for r in results:
                LOGGER.debug(f'finished fitting schedule: {r[0]}')
                knn_result['schedules'][r[0]] = {}
                knn_result['schedules'][r[0]]['distance'] = r[1]
        
            write_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json'),knn_result)

        # Now apply DBSCAN
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dbscan_eps = read_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_eps.json'))
            work_order = [[],[]]
            dbscan_result = {'schedules':{}}
            step = 0.1

            for filepath in dbscan_eps:
                eps = dbscan_eps[filepath]
                distance = read_json(
                    os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json')
                )['schedules'][self.schedule_name(filepath)]['distance']
                mini, maxi = max(round(min(distance),1),step), round(max(distance),1)
                eps = np.arange(mini,maxi,step).tolist()
                work_order[1] += eps
                work_order[0] += [filepath]*len(eps)
                dbscan_result['schedules'][self.schedule_name(filepath)] = {
                    'eps':[],
                    'min_samples':[],
                    'scores':{'sse':[],'calinski_harabasz':[],'silhouette':[]},
                    'labels':[]
                }
        
            results = executor.map(self.fit_dbscan,*work_order)

            for r in results:
                if r[3] is None:
                    LOGGER.debug(f'UNSUCCESSFUL fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
                else:
                    LOGGER.debug(f'finished fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
                    dbscan_result['schedules'][r[0]]['eps'].append(r[1])
                    dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
                    dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
                    dbscan_result['schedules'][r[0]]['labels'].append(r[3])
                    dbscan_result['schedules'][r[0]]['scores']['sse'].append(r[4]['sse'])
                    dbscan_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[4]['calinski_harabasz'])
                    dbscan_result['schedules'][r[0]]['scores']['silhouette'].append(r[4]['silhouette'])
                
            write_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'),dbscan_result)

    def fit_dbscan(self,filepath,eps,min_samples=None):
        x = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl')).values
        min_samples = int(x.shape[1]*2) if min_samples is None else min_samples
        result = DBSCAN(eps=eps,min_samples=min_samples).fit(x)
        try:
            scores = {
                'sse':self.get_sse(x,result.labels_.tolist()),
                'calinski_harabasz':calinski_harabasz_score(x,result.labels_),
                'silhouette':silhouette_score(x,result.labels_)
            }
            return self.schedule_name(filepath), eps, min_samples, result.labels_.tolist(), scores
        except ValueError as e:
            return self.schedule_name(filepath), eps, min_samples, None

    def fit_knn(self,filepath,n_neighbors):
        x = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl')).values
        distance, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
        return self.schedule_name(filepath), distance[:,1].tolist()

    def plot_kmeans(self):
        kmeans_result = read_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'kmeans_result.json'))
        
        # KMeans scores
        row_count = len(self.schedule_data_filepaths)
        column_count = len(list(kmeans_result['schedules'].values())[0]['scores'])
        fig, axs = plt.subplots(row_count, column_count, figsize=(min(5,self.MAX_FIGURE_WIDTH)*column_count,2*row_count))
        x = kmeans_result['n_clusters']

        for i, (schedule, schedule_data) in enumerate(kmeans_result['schedules'].items()):
            for j, (score, y) in enumerate(schedule_data['scores'].items()):
                axs[i,j].plot(x,y)
                axs[i,j].set_title(f'{schedule}')
                axs[i,j].set_xlabel('n_clusters')
                axs[i,j].set_ylabel(score)

        plt.tight_layout()
        fig.align_ylabels()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_kmeans_scores.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def run_kmeans(self): 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            kmeans_result = {'n_clusters':list(range(2,101)),'schedules': {}}
            work_order = [[],kmeans_result['n_clusters']*len(self.schedule_data_filepaths)]

            for f in self.schedule_data_filepaths:
                work_order[0] += [f]*len(kmeans_result['n_clusters'])
                kmeans_result['schedules'][self.schedule_name(f)] = {'scores':{'sse':[],'calinski_harabasz':[],'silhouette':[]},'labels':[]}
            
            results = executor.map(self.fit_kmeans,*work_order)

            for r in results:
                LOGGER.debug(f'finished fitting schedule: {r[0]}, n_clusters: {r[1]}')
                kmeans_result['schedules'][r[0]]['labels'].append(r[2])
                kmeans_result['schedules'][r[0]]['scores']['sse'].append(r[3]['sse'])
                kmeans_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[3]['calinski_harabasz'])
                kmeans_result['schedules'][r[0]]['scores']['silhouette'].append(r[3]['silhouette'])
        
            write_json(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'kmeans_result.json'),kmeans_result)

    def fit_kmeans(self,filepath,n_clusters):
        x = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl')).values
        result = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
        scores = {
            'sse':result.inertia_,
            'calinski_harabasz':calinski_harabasz_score(x,result.labels_),
            'silhouette':silhouette_score(x,result.labels_)
        }
        return self.schedule_name(filepath), n_clusters, result.labels_.tolist(), scores

    def get_sse(self,x,labels):
        df = pd.DataFrame(x)
        df['label'] = labels
        df = df.groupby('label').apply(lambda gr:
            (gr.iloc[:,0:-1] - gr.iloc[:,0:-1].mean())**2
        )
        sse = df.sum().sum()
        return sse
    
    def plot_schedule_cluster_data(self):
        # cluster dataset
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, ax = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
            x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count (Normalized)',fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel('Building')
            ax.set_xlabel('PCA Component')
            ax.set_yticks([])
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_pca_cluster_dataset_building_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def save_schedule_cluster_data(self):
        explained_variance_ratio_data = read_json(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'))

        # prep cluster data
        for i, filepath in enumerate(self.schedule_data_filepaths):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            variance_ratio_data = explained_variance_ratio_data[self.schedule_name(filepath)]
            index = self.pca_threshold_satisfaction_index(variance_ratio_data)
            pca_columns = list(range(0,index + 1))
            plot_data = plot_data[['metadata_id'] + pca_columns].copy()
            plot_data = plot_data.set_index('metadata_id').sort_index()
            plot_data.to_pickle(os.path.join(self.SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))

    def plot_schedule_pca_data(self):    
        explained_variance_ratio_data = read_json(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'))
        # components per schedule
        _, ax = plt.subplots(1,1,figsize=(min(4,self.MAX_FIGURE_WIDTH),8))
        y1 = [self.pca_threshold_satisfaction_index(v) + 1 for _, v in explained_variance_ratio_data.items()]
        y2 = [len(v) for _, v in explained_variance_ratio_data.items()]
        x = list(range(len(y1)))
        ax.barh(x,y1,color=['blue'],label=f'{self.PCA_CUMMULATIVE_VARIANCE_THRESHOLD*100}%')
        ax.barh(x,y2,color=['blue'],alpha=0.2,label=f'100%')
        ax.set_yticks(x)
        ax.set_yticklabels(list(explained_variance_ratio_data.keys()))
        ax.legend(title='Explained variance')
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_pca_threshold_satisfaction_components.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # explained variance ratio
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, _ = plt.subplots(row_count, column_count, figsize=(min(18,self.MAX_FIGURE_WIDTH),2.5*row_count))

        for i, (ax, (column, y2)) in enumerate(zip(fig.axes, explained_variance_ratio_data.items())):
            y1 = [sum(y2[0:i]) for i in range(len(y2))]
            x = list(range(0,len(y1)))
            ax.bar(x,y1,color='grey',label='Previous Sum')
            ax.bar(x,y2,bottom=y1,color='blue',label='Current')
            ax.axhline(self.PCA_CUMMULATIVE_VARIANCE_THRESHOLD,color='red',linestyle='--',label='Threshold')
            index = self.pca_threshold_satisfaction_index(y2) + 0.5
            ax.axvline(index,color='red',linestyle='--',)
            ax.set_ylabel('Cummulative explained\nvariance ratio')
            ax.set_xlabel('Component')
            ax.set_xticks(x)
            ax.set_xticklabels(x)
            ax.set_title(column)
            ax.legend()

        plt.tight_layout()
        fig.align_ylabels()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_pca_explained_variance_ratio.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # distribution of components
        fig, _ = plt.subplots(row_count, column_count, figsize=(min(18,self.MAX_FIGURE_WIDTH),2.5*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            variance_ratio_data = explained_variance_ratio_data[self.schedule_name(filepath)]
            index = self.pca_threshold_satisfaction_index(variance_ratio_data) + 1.5
            columns = list(range(0,len(variance_ratio_data)))
            ax.boxplot(plot_data[columns].values)
            ax.set_ylabel('Value')
            ax.set_xlabel('Component')
            ax.axvline(index, color='red', linestyle='--')
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        fig.align_ylabels()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_pca_component_distribution.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # correlation between components and original variables
        fig, ax = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_csv(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'{self.schedule_name(filepath)}_correlation.csv'))
            plot_data = plot_data.set_index('component')
            x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)
            explained_variance_data = explained_variance_ratio_data[self.schedule_name(filepath)]
            ax.axhline(self.pca_threshold_satisfaction_index(explained_variance_data) + 0.5,color='red',linestyle='--')

            for k in range(len(y)):
                for j in range(len(x)):
                    value = z[k,j]
                    value = round(value,2) if abs(value) >= 0.4 else None
                    _ = ax.text(j,k,value,ha='center',va='center',color='black')

            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel('Component')
            ax.set_xlabel('Hour')
            ax.set_xticks(x)
            ax.set_yticks(y)
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_pca_correlation_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def save_schedule_pca_data(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            work_order = self.schedule_data_filepaths
            explained_variance_ratio_data = {}
            results = [executor.submit(self.preprocess_pca,*[w]) for w in work_order]

            for i, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    r = future.result()
                    LOGGER.debug(f'finished pca: {r[0]}')
                    explained_variance_ratio_data[r[0]] = r[1]
                except Exception as e:
                    LOGGER.exception(e)
            
            write_json(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'),explained_variance_ratio_data)

    def preprocess_pca(self,filepath):
        plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
        plot_data = plot_data.groupby(['metadata_id','sax_word']).size().reset_index(name='count')
        plot_data = plot_data.pivot(index='metadata_id',columns='sax_word',values='count')
        plot_data = plot_data.fillna(0)
        # standrdize
        scaler = StandardScaler()
        x = plot_data.values
        scaler = scaler.fit(x)
        x = scaler.transform(x)
        pca = PCA(n_components=None)
        pca = pca.fit(x)
        component_data = pd.DataFrame(pca.transform(x), columns=range(0,plot_data.shape[1]))
        index_data = plot_data.reset_index(drop=False)[['metadata_id']].copy()
        index_data = pd.concat([index_data,component_data],axis=1)
        index_data.to_pickle(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
        component_data.columns = [str(c) for c in component_data.columns]
        keys = plot_data.columns.tolist()
        plot_data = pd.concat([plot_data.reset_index(drop=False),component_data],axis=1)
        plot_data = plot_data.corr('pearson')
        plot_data = plot_data.loc[component_data.columns.tolist()][keys].copy()
        plot_data = plot_data.reset_index(drop=True)
        plot_data['component'] = plot_data.index
        plot_data.to_csv(os.path.join(self.SCHEDULE_PCA_DATA_DIRECTORY,f'{self.schedule_name(filepath)}_correlation.csv'),index=False)
        return self.schedule_name(filepath), list(pca.explained_variance_ratio_)
         
    def plot_schedule_sax_data(self):
        sax_result = read_json(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'sax_result.json'))
        row_count = len(sax_result['schedules'])
        column_count = 2
        fig, axs = plt.subplots(row_count,column_count,figsize=(min(18,self.MAX_FIGURE_WIDTH),2.5*row_count))
        
        for i, (schedule, schedule_data) in enumerate(sax_result['schedules'].items()):
            plot_data = pd.DataFrame({'a':schedule_data['a'],'w':schedule_data['w'],'entropy':schedule_data['entropy']})
            plot_data = plot_data.drop_duplicates(['a','w'])
            plot_data = plot_data.explode('entropy')
            plot_data = plot_data.groupby(['a','w'])[['entropy']].mean().reset_index()

            
            # constant a
            xy = plot_data[plot_data['a']==max(self.SAX_A)].copy()
            xy = xy.sort_values('w')
            x,y = xy['w'], xy['entropy']
            axs[0].plot(x,y)
            axs[0].set_xlabel('w')
            axs[0].set_ylabel('entropy')
            axs[0].set_title(f'a={max(self.SAX_A)}')
            axs[0].set_xticks(x)

            # constant w
            xy = plot_data[plot_data['w']==max(self.SAX_W)].copy()
            xy = xy.sort_values('a')
            x,y = xy['a'], xy['entropy']
            axs[1].plot(x,y)
            axs[1].set_xlabel('a')
            axs[1].set_ylabel('entropy')
            axs[1].set_title(f'w={max(self.SAX_W)}')
            axs[1].set_xticks(x)

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_sax_entropy.png'), facecolor='white', bbox_inches='tight')
        plt.close()
        assert False


        # collective word frequency
        row_count = len(self.schedule_data_filepaths)
        column_count = 1
        fig, _ = plt.subplots(row_count,column_count,figsize=(min(18,self.MAX_FIGURE_WIDTH),2.5*row_count))
        
        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            plot_data = plot_data.groupby(['sax_word']).size().reset_index(name='count')
            plot_data = plot_data.sort_values('count',ascending=False)
            y = plot_data['count']
            x = list(range(len(y)))
            ax.bar(x,y)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_data['sax_word'].tolist())
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel('Count')
            ax.set_xlabel('Word')
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_sax_word_frequency_count_bar_graph.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # word frequency w.r.t. temporal features heatmap
        for classification in ['day_of_week','month']:
            fig, ax = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),6*row_count))

            for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
                plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
                plot_data = pd.merge(plot_data,self.DATE_RANGE,on='date',how='left')
                plot_data = plot_data.groupby(['sax_word',classification]).size().reset_index(name='count')
                plot_data = plot_data.pivot(index='sax_word',columns=classification,values='count')
                plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False)
                plot_data = plot_data.fillna(0)
                plot_data = plot_data.T
                x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
                divnorm = colors.TwoSlopeNorm(vcenter=0)
                _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)
                _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count',fraction=0.025,pad=0.01)
                ax.tick_params('x',which='both',rotation=90)
                ax.set_ylabel(classification)
                ax.set_xlabel('Word')
                ax.set_title(self.schedule_name(filepath))

            plt.tight_layout()
            plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_sax_word_{classification}_frequency_count_heatmap.png'), facecolor='white', bbox_inches='tight')
            plt.close()

        # word occurence heatmpa across building to pick out groups of buildings on a temporal scale
        fig, ax = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            words = plot_data.groupby(['sax_word']).size().reset_index()[['sax_word']].copy()
            words['word_id'] = words.index + 1
            plot_data = pd.merge(plot_data,words,on='sax_word',how='left')
            plot_data = plot_data.pivot(index='metadata_id',columns='date',values='word_id')
            plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
            x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Word ID',fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel('Building')
            ax.set_xlabel('Timestep')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_sax_word_norm_count_building_timeseries_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

        # word count per building heatmap
        fig, ax = plt.subplots(row_count,column_count,figsize=(min(10,self.MAX_FIGURE_WIDTH),10*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
            plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'{self.schedule_name(filepath)}.pkl'))
            plot_data = plot_data.groupby(['metadata_id','sax_word']).size().reset_index(name='count')
            plot_data = plot_data.pivot(index='metadata_id',columns='sax_word',values='count')
            plot_data = plot_data.fillna(0)
            plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
            x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
            z = minmax_scale(z)
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count (Normalized)',fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel('Building')
            ax.set_xlabel('Word')
            ax.set_yticks([])
            ax.set_title(self.schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_sax_word_norm_count_building_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def save_schedule_sax_data(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(len(self.schedule_data_filepaths),10)) as executor:
            schedules = [self.schedule_name(s) for s in self.schedule_data_filepaths]
            varied_a_work_order = list(itertools.product(*[schedules,self.SAX_A,[max(self.SAX_W)]]))
            varied_w_work_order = list(itertools.product(*[schedules,[max(self.SAX_A)],self.SAX_W]))
            work_order = varied_a_work_order + varied_w_work_order
            results = [executor.submit(self.preprocess_sax,*w) for w in work_order]
            sax_result = {'schedules':{s:{'a':[],'w':[],'entropy':[],'word':[]} for s in schedules}}

            for i, future in enumerate(concurrent.futures.as_completed(results)):
                try:
                    r = future.result()
                    LOGGER.debug(f'finished sax: {r[0]}, a: {r[1]}, w: {r[2]}')
                    sax_result['schedules'][r[0]]['a'].append(r[1])
                    sax_result['schedules'][r[0]]['w'].append(r[2])
                    sax_result['schedules'][r[0]]['word'].append(r[4])
                    sax_result['schedules'][r[0]]['entropy'].append(r[5])

                except Exception as e:
                    LOGGER.exception(e)

            write_json(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY,f'sax_result.json'),sax_result)

    def preprocess_sax(self,schedule,a,w):
        plot_data = pd.read_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY, f'{schedule}_pre_sax_norm.pkl'))

        # use if evenly spaced windows and want to speed up paa calc
        timestep_count = len(plot_data['timestep'].unique())
        plot_data['day'] = (plot_data['timestep']/(timestep_count)).astype(int)
        plot_data['day_group'] = (plot_data['timestep']/(timestep_count/w)).astype(int)
        plot_data = plot_data.groupby(['metadata_id','day','day_group'])[[schedule]].mean()
        sax_paa = plot_data[schedule].tolist()

        # # use for saxpy paa function
        # w *= len(plot_data['metadata_id'].unique())
        # w = int(w)
        # sax_paa = paa(plot_data[schedule].tolist(),w)

        sax_word = ts_to_string(sax_paa,self.SAX_CUTS[a])
        entropy = [self.get_entropy([0]*w,list(sax_word[i:i+w]))[0] for i in range(0,len(sax_paa),w)]
        return schedule, a, w, sax_paa, sax_word, entropy

    def get_entropy(self,labels,ground_truth):
        df = pd.DataFrame({'label':labels,'ground_truth':ground_truth})
        
        label_sizes = df[['label']].groupby('label').size().reset_index(name='size')
        label_sizes = label_sizes.set_index('label').to_dict('index')
        
        df = df.groupby(['label','ground_truth']).apply(lambda gr:
            len(gr) / label_sizes[gr.iloc[0]['label']]['size']
        ).reset_index(name='probability')

        purity = df.groupby('label').apply(lambda gr:
            gr['probability'].max() * label_sizes[gr.iloc[0]['label']]['size'] / len(labels)
        ).reset_index(name='purity')
        purity = purity['purity'].sum()
        
        df = df.groupby(['label','ground_truth']).apply(lambda gr:
            gr.iloc[0]['probability']*np.log2(gr.iloc[0]['probability'])
        ).reset_index(name='entropy')

        df['entropy'] = df['entropy'].abs()
        df = df.groupby('label')[['entropy']].sum().reset_index(drop=False)
        
        df['entropy'] = df.apply(lambda x:
            x['entropy']*label_sizes[x['label']]['size'] / len(labels),axis=1
        )
        entropy = df['entropy'].sum()

        return entropy, purity

    def save_normalized_pre_schedule_sax_data(self):
        for i, filepath in enumerate(self.schedule_data_filepaths):
            plot_data = pd.read_pickle(filepath)

            # normalize building values between 0 and 1 to remove effect of magnitude across buildings with
            # highly correlated schedules
            mini = plot_data.groupby(['metadata_id'])[[self.schedule_name(filepath)]].min().reset_index()
            mini = mini.rename(columns={self.schedule_name(filepath):'min'})
            maxi = plot_data.groupby(['metadata_id'])[[self.schedule_name(filepath)]].max().reset_index()
            maxi = maxi.rename(columns={self.schedule_name(filepath):'max'})
            plot_data = pd.merge(plot_data,mini,on='metadata_id',how='left')
            plot_data = pd.merge(plot_data,maxi,on='metadata_id',how='left')
            plot_data[self.schedule_name(filepath)] = (plot_data[self.schedule_name(filepath)] - plot_data['min'])/(plot_data['max'] - plot_data['min'])
            
            # then standardize for SAX discretization purposes
            scaler = StandardScaler()
            x = plot_data[[self.schedule_name(filepath)]].values
            scaler = scaler.fit(x)
            plot_data[self.schedule_name(filepath)] = scaler.transform(x)

            plot_data = plot_data.drop(columns=['min','max'])
            plot_data.to_pickle(os.path.join(self.SCHEDULE_SAX_DATA_DIRECTORY, f'{self.schedule_name(filepath)}_pre_sax_norm.pkl'))

    def plot_schedule_corr(self):
        # correlation
        fig, ax = plt.subplots(1,1,figsize=(min(10,self.MAX_FIGURE_WIDTH),10))
        plot_data = pd.read_csv(os.path.join(self.SCHEDULE_DATA_DIRECTORY,'filtered_schedule_correlation.csv'))
        plot_data = plot_data.set_index(plot_data.columns)
        x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
        divnorm = colors.TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0)

        for k in range(len(y)):
            for j in range(len(x)):
                value = z[k,j]
                value = round(value,2) if k != j and abs(value) >= 0.5 else None
                _ = ax.text(j,k,value,ha='center',va='center',color='black')

        _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)
        ax.tick_params('x',which='both',rotation=90)
        ax.set_ylabel(None)
        ax.set_xlabel(None)

        plt.tight_layout()
        plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_filtered_correlation_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    def save_schedule_corr(self):
        columns = [self.schedule_name(f) for f in self.schedule_data_filepaths]
        data_list = []

        for i, row_filepath in enumerate(self.schedule_data_filepaths):
            row_corr = []
            row_data = pd.read_pickle(row_filepath)[['metadata_id','timestep',self.schedule_name(row_filepath)]].copy()

            for j, column_filepath in enumerate(self.schedule_data_filepaths):
                if i == j:
                    row_corr.append(1)
                elif i > 0 and j < len(data_list):
                    row_corr.append(data_list[j][i])
                else:
                    column_data = pd.read_pickle(column_filepath)[['metadata_id','timestep',self.schedule_name(column_filepath)]].copy()
                    corr_data = pd.merge(row_data, column_data, on=['metadata_id','timestep'], how='left')
                    corr_data = corr_data[[self.schedule_name(row_filepath),self.schedule_name(column_filepath)]].corr('pearson').iloc[0,1]
                    row_corr.append(corr_data)

            data_list.append(row_corr)

        plot_data = pd.DataFrame(data_list,columns=columns)
        plot_data = plot_data.set_index(columns)
        plot_data.to_csv(os.path.join(self.SCHEDULE_DATA_DIRECTORY,'filtered_schedule_correlation.csv'),index=True)

    def plot_schedule_building_hourly_std_wrt_general_avg_box_plot(self):
        for hue in ['day_of_week','season']:
            row_count = len(self.schedule_data_filepaths)
            column_count = 1
            fig, _ = plt.subplots(row_count, column_count, figsize=(min(18,self.MAX_FIGURE_WIDTH),3*row_count))
            
            for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
                plot_data = pd.merge(pd.read_pickle(filepath),self.DATE_RANGE,on='timestep')
                mean_data = plot_data.groupby(['hour',hue])[[self.schedule_name(filepath)]].mean().reset_index()
                mean_data = mean_data.rename(columns={self.schedule_name(filepath):'mean'})
                count_data = plot_data.groupby(['metadata_id','hour',hue])[[self.schedule_name(filepath)]].size().reset_index(name='count')
                plot_data = pd.merge(plot_data,mean_data,on=['hour',hue],how='left')
                plot_data = pd.merge(plot_data,count_data,on=['metadata_id','hour',hue],how='left')
                plot_data['std'] = (plot_data[self.schedule_name(filepath)] - plot_data['mean'])**2/plot_data['count']
                plot_data = plot_data.groupby(['metadata_id','hour',hue])[['std']].sum().reset_index()
                plot_data['std'] = plot_data['std']**(1/2)
                sns.boxplot(x='hour',y='std',hue=hue,data=plot_data,ax=ax)
                ax.set_ylabel(None)
                ax.set_title(self.schedule_name(filepath))

            plt.tight_layout()
            plt.savefig(
                os.path.join(self.FIGURES_DIRECTORY,f'schedule_hourly_{hue}_building_std_from_collective_mean_box_plot.png'),
                facecolor='white', bbox_inches='tight'
            )
            plt.close()

    def plot_schedule_building_hourly_std_box_plot(self):
        for hue in ['day_of_week','season']:
            row_count = len(self.schedule_data_filepaths)
            column_count = 1
            fig, _ = plt.subplots(row_count, column_count, figsize=(min(18,self.MAX_FIGURE_WIDTH),3*row_count))
            
            for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
                plot_data = pd.merge(pd.read_pickle(filepath),self.DATE_RANGE,on='timestep')
                plot_data = plot_data.groupby(['metadata_id','hour',hue])[[self.schedule_name(filepath)]].std().reset_index()
                sns.boxplot(x='hour',y=self.schedule_name(filepath),hue=hue,data=plot_data,ax=ax)
                ax.set_ylabel(None)
                ax.set_title(self.schedule_name(filepath))

            plt.tight_layout()
            plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_hourly_{hue}_building_std_box_plot.png'), facecolor='white', bbox_inches='tight')
            plt.close()

    def plot_schedule_hourly_box_plot(self):
        for hue in ['day_of_week','season']:
            row_count = len(self.schedule_data_filepaths)
            column_count = 1
            fig, _ = plt.subplots(row_count, column_count, figsize=(min(10,self.MAX_FIGURE_WIDTH),3*row_count))
            
            for i, (ax, filepath) in enumerate(zip(fig.axes, self.schedule_data_filepaths)):
                plot_data = pd.merge(pd.read_pickle(filepath),self.DATE_RANGE,on='timestep')
                sns.boxplot(x='hour',y=self.schedule_name(filepath),hue=hue,data=plot_data,ax=ax)
                ax.set_ylabel(None)
                ax.set_title(self.schedule_name(filepath))

            plt.tight_layout()
            plt.savefig(os.path.join(self.FIGURES_DIRECTORY,f'schedule_hourly_{hue}_box_plot.png'), facecolor='white', bbox_inches='tight')
            plt.close()

    def pca_threshold_satisfaction_index(self,variance_ratio):
        return [
            i for i, v in enumerate([sum(variance_ratio[0:i + 1]) for i in range(len(variance_ratio))]) 
            if v >= self.PCA_CUMMULATIVE_VARIANCE_THRESHOLD
        ][0]

    def schedule_name(self,filepath):
        return filepath.split('/')[-1].split('.')[0]
    
    def save_schedule_data(self):
        # schedules
        schedule_columns = self.DATABASE.query_table("""PRAGMA table_info(schedule)""")
        schedule_columns = schedule_columns[~schedule_columns['name'].isin(['metadata_id','timestep'])]['name'].tolist()
        schedule_columns = ['occupants']

        for i, column in enumerate(schedule_columns):
            self.DATABASE.query_table(f"""
            SELECT
                metadata_id,
                timestep,
                {column}
            FROM schedule
            """).to_pickle(os.path.join(self.SCHEDULE_DATA_DIRECTORY,f'{column}.pkl'))
    
    def set_logger(self):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        LOGGER.setLevel(logging.DEBUG)
        handler = logging.handlers.RotatingFileHandler(self.LOG_FILEPATH,mode='a',maxBytes=self.LOG_MAX_BYTES,backupCount=self.LOG_BACKUP_COUNT)
        formatter = logging.Formatter('%(levelname)s - %(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)

if __name__ == '__main__':
    Analysis().run()