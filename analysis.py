import concurrent.futures
import os
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
from doe_xstock.utilities import read_json, write_json

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

# CONSTANTS & VARIABLES ****************************************************************
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
DATABASE_FILEPATH = '/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db'
FIGURES_DIRECTORY = 'figures/'
SCHEDULE_DATA_DIRECTORY = '../schedule_data'
SCHEDULE_PCA_DATA_DIRECTORY = '../schedule_pca_data'
SCHEDULE_SAX_DATA_DIRECTORY = '../schedule_sax_data'
SCHEDULE_CLUSTER_DATA_DIRECTORY = '../schedule_cluster_data'
SCHEDULE_DATA_FILEPATHS = [os.path.join(SCHEDULE_DATA_DIRECTORY,f) for f in os.listdir(SCHEDULE_DATA_DIRECTORY) if f.endswith('.pkl')]
SCHEDULE_DATA_FILEPATHS = sorted(SCHEDULE_DATA_FILEPATHS)
SEASONS = {
    1:'winter',2:'winter',12:'winter',
    3:'spring',4:'spring',5:'spring',
    6:'summer',7:'summer',8:'summer',
    9:'fall',10:'fall',11:'fall',
}
PCA_CUMMULATIVE_VARIANCE_THRESHOLD = 0.95
SAX_A = 3
SAX_W = 4
DATE_RANGE = pd.DataFrame({'timestamp':pd.date_range('2017-01-01','2017-12-31 23:00:00', freq='H')})
DATE_RANGE['timestep'] = DATE_RANGE.index
DATE_RANGE['month'] = DATE_RANGE['timestamp'].dt.month
DATE_RANGE['week'] = DATE_RANGE['timestamp'].dt.isocalendar().week
DATE_RANGE['date'] = DATE_RANGE['timestamp'].dt.normalize()
DATE_RANGE['day_of_week'] = DATE_RANGE['timestamp'].dt.weekday
DATE_RANGE.loc[DATE_RANGE['day_of_week'] == 6, 'week_of'] = DATE_RANGE.loc[DATE_RANGE['day_of_week'] == 6]['timestamp'].dt.normalize()
DATE_RANGE['week_of'] = DATE_RANGE['week_of'].ffill()
DATE_RANGE['season'] = DATE_RANGE['month'].map(lambda x: SEASONS[x])
# lambda functions
schedule_name = lambda x: x.split('/')[-1].split('.')[0]
threshold_satisfaction_index = lambda x: [
    i for i, v in enumerate([sum(x[0:i + 1]) for i in range(len(x))]) if v >= PCA_CUMMULATIVE_VARIANCE_THRESHOLD
][0]

# FIGURE ****************************************************************
'''DESCRIPTION: schedule hourly box plots categorized by day of week and season to investigate variance w.r.t. the categories.'''

def plot():
    for hue in ['day_of_week','season']:
        row_count = len(SCHEDULE_DATA_FILEPATHS)
        column_count = 1
        fig, _ = plt.subplots(row_count, column_count, figsize=(18,3*row_count))
        
        for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
            print(f'{hue} - {i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
            plot_data = pd.merge(pd.read_pickle(filepath),DATE_RANGE,on='timestep')
            sns.boxplot(x='hour',y=schedule_name(filepath),hue=hue,data=plot_data,ax=ax)
            ax.set_ylabel(None)
            ax.set_title(schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_{hue}_schedule_box_plot.png'), facecolor='white', bbox_inches='tight')
        plt.close()
# plot()

'''NOTE:
__Hourly variance when categorized by day-of-week__
- There are distinct hourly shapes i.e. all schedules change from hour to hour.
- there are day-of-week variations in the schedules w.r.t. to hour in some cases.
- For most schedules, weekdays have similar distribution.
- Friday and Saturday seem to be typically distinct from other days of the week (possible that E+ sims started with Monday as first day of week not Sunday as shown in EPW file).
- Some hours have no variance irrespective of the day of week i.e. 1 unique value for records hence, dimensionality reduction of the daily data
to a smaller dimension will be feasible e.g. instead of 24 hours of data to represent a day, maybe just use hours within 8 AM - 10 PM or so.
Maybe PCA can help with this dimension reduction.

__Hourly variance when categorized by season__
- There are indeed seasonal variations and particularly winter season seems to have higher values than other seasons.
- However, within a season, there may be little to no variance.
- Some schedules like occupancy show the same distribution irrespective of the season but show variance when categorized by day-of-week.'''
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION: exclude the following schedules as they do not provide any variance.'''

schedules_to_exclude = ['plug_loads_vehicle','vacancy','lighting_exterior_holiday']
SCHEDULE_DATA_FILEPATHS = [f for f in SCHEDULE_DATA_FILEPATHS if not schedule_name(f) in schedules_to_exclude]
# END

# FIGURE ****************************************************************
'''DESCRIPTION: building schedule standard deviation hourly box plots categorized by day of week and season to investigate 
variance per building w.r.t. the categories.'''

def plot():
    for hue in ['day_of_week','season']:
        row_count = len(SCHEDULE_DATA_FILEPATHS)
        column_count = 1
        fig, _ = plt.subplots(row_count, column_count, figsize=(18,3*row_count))
        
        for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
            print(f'{hue} - {i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
            plot_data = pd.merge(pd.read_pickle(filepath),DATE_RANGE,on='timestep')
            plot_data = plot_data.groupby(['metadata_id','hour',hue])[[schedule_name(filepath)]].std().reset_index()
            sns.boxplot(x='hour',y=schedule_name(filepath),hue=hue,data=plot_data,ax=ax)
            ax.set_ylabel(None)
            ax.set_title(schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_{hue}_schedule_building_std_box_plot.png'), facecolor='white', bbox_inches='tight')
        plt.close()
# plot()

'''NOTE:
__Hourly variance when categorized by day-of-week__
- Distinct buildings have hourly variance when hours are categorized by day-of-week as indicated by non-zero standard deviation distributions.
- Standard deviation tends to increase later in the day for most schedules.

__Hourly variance when categorized by season__
- Distinct buildings have hourly variance when hours are categorized by season as indicated by non-zero standard deviation distributions.
- Some schedules like fuel_loads_grill only have > 0 std for all buildings during fall because they have constant non-zero value during other seasons.'''
# END

# FIGURE ****************************************************************
'''DESCRIPTION: building schedule standard deviation from general average hourly box plots categorized by day of week and season to investigate 
variance per building w.r.t. the categories.'''

def plot():
    for hue in ['day_of_week','season']:
        row_count = len(SCHEDULE_DATA_FILEPATHS)
        column_count = 1
        fig, _ = plt.subplots(row_count, column_count, figsize=(18,3*row_count))
        
        for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
            print(f'{hue} - {i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
            plot_data = pd.merge(pd.read_pickle(filepath),DATE_RANGE,on='timestep')
            mean_data = plot_data.groupby(['hour',hue])[[schedule_name(filepath)]].mean().reset_index()
            mean_data = mean_data.rename(columns={schedule_name(filepath):'mean'})
            count_data = plot_data.groupby(['metadata_id','hour',hue])[[schedule_name(filepath)]].size().reset_index(name='count')
            plot_data = pd.merge(plot_data,mean_data,on=['hour',hue],how='left')
            plot_data = pd.merge(plot_data,count_data,on=['metadata_id','hour',hue],how='left')
            plot_data['std'] = (plot_data[schedule_name(filepath)] - plot_data['mean'])**2/plot_data['count']
            plot_data = plot_data.groupby(['metadata_id','hour',hue])[['std']].sum().reset_index()
            plot_data['std'] = plot_data['std']**(1/2)
            sns.boxplot(x='hour',y='std',hue=hue,data=plot_data,ax=ax)
            ax.set_ylabel(None)
            ax.set_title(schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_{hue}_schedule_building_std_from_collective_mean_box_plot.png'), facecolor='white', bbox_inches='tight')
        plt.close()
# plot()

'''NOTE:
- There are schedules that can be excluded from analyssi becasue there is no variance across buildings when their values are evaluated w.r.t. to a common mean value.'''
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION: exclude the following schedules as they vary temporally but not spatially i.e. across buildings.'''

schedules_to_exclude = ['fuel_loads_fireplace','fuel_loads_grill','fuel_loads_lighting','lighting_exterior','lighting_garage','plug_loads_well_pump']
SCHEDULE_DATA_FILEPATHS = [f for f in SCHEDULE_DATA_FILEPATHS if not schedule_name(f) in schedules_to_exclude]
# END

# FIGURE ****************************************************************
'''DESCRIPTION: building schedule correlation to further narrow down the feature space.'''

def plot():
        columns = [schedule_name(f) for f in SCHEDULE_DATA_FILEPATHS]
        data_list = []

        for i, row_filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
            row_corr = []
            row_data = pd.read_pickle(row_filepath)[['metadata_id','timestep',schedule_name(row_filepath)]].copy()

            for j, column_filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
                print('i:',i,'j:',j)
                if i == j:
                    row_corr.append(1)
                elif i > 0 and j < len(data_list):
                    row_corr.append(data_list[j][i])
                else:
                    column_data = pd.read_pickle(column_filepath)[['metadata_id','timestep',schedule_name(column_filepath)]].copy()
                    corr_data = pd.merge(row_data, column_data, on=['metadata_id','timestep'], how='left')
                    corr_data = corr_data[[schedule_name(row_filepath),schedule_name(column_filepath)]].corr('pearson').iloc[0,1]
                    row_corr.append(corr_data)

            data_list.append(row_corr)

        plot_data = pd.DataFrame(data_list,columns=columns)
        plot_data = plot_data.set_index(columns)
        plot_data.to_csv(os.path.join(SCHEDULE_DATA_DIRECTORY,'filtered_schedule_correlation.csv'),index=True)
        
        # correlation
        fig, ax = plt.subplots(1,1,figsize=(12,12))
        plot_data = pd.read_csv(os.path.join(SCHEDULE_DATA_DIRECTORY,'filtered_schedule_correlation.csv'))
        plot_data = plot_data.set_index(plot_data.columns)
        x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
        divnorm = colors.TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)

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
        plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_filtered_schedule_correlation_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

# plot()

'''NOTE:
- Clothes dryer exhaust and clothes dryer have Pearson correlation = 1. 
- Also ligthing, plug load and ceiling fan are highly correlated. Lighting has higher correlation with the other 2 than the otehr 2 have with themselves and lighting so lighting may be used to represent all 3.'''
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION: exclude the following schedules as they vary temporally but not spatially i.e. across buildings.'''

schedules_to_exclude = ['clothes_dryer_exhaust','ceiling_fan','plug_loads']
SCHEDULE_DATA_FILEPATHS = [f for f in SCHEDULE_DATA_FILEPATHS if not schedule_name(f) in schedules_to_exclude]
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION:
 SAX transformation of the schedule time series using a window size of SAX_W for PAA and alphabet size of SAX_A for time series to word conversion.
 The idea is that SAX compresses the data to discrete words describing the scehdule profiles that can be aggregated by counts of unique words per building 
 and used as clustering dataset. Essentially this reduces the width of the clustering data from the length of the time series to the number of unique words.'''

def manipulate():
    # set data
    for i, filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.merge(pd.read_pickle(filepath),DATE_RANGE,on='timestep')
        # normalize first
        scaler = StandardScaler()
        x = plot_data[[schedule_name(filepath)]].values
        scaler = scaler.fit(x)
        plot_data[schedule_name(filepath)] = scaler.transform(x)
        plot_data = plot_data.pivot(index=['metadata_id','date'],columns='hour',values=schedule_name(filepath))
        plot_data['sax_paa'] = plot_data[plot_data.columns.tolist()].apply(lambda x: paa(x.values,SAX_W),axis=1)
        plot_data['sax_word'] = plot_data['sax_paa'].map(lambda x: ts_to_string(x,cuts_for_asize(SAX_A)))
        plot_data = plot_data.reset_index(drop=False)
        plot_data[['metadata_id','date','sax_paa','sax_word']].to_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        print(f'{schedule_name(filepath)} unique count:',len(plot_data['sax_word'].unique()))

    # plot
    row_count = len(SCHEDULE_DATA_FILEPATHS)
    column_count = 1
    # collective word frequency
    fig, _ = plt.subplots(row_count,column_count,figsize=(18,2.5*row_count))
    
    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
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
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_sax_word_frequency_count_bar_graph.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # word frequency w.r.t. temporal features heatmap
    for classification in ['day_of_week','month']:
        fig, ax = plt.subplots(row_count,column_count,figsize=(18,6*row_count))

        for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
            print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
            plot_data = pd.read_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
            plot_data = pd.merge(plot_data,DATE_RANGE,on='date',how='left')
            plot_data = plot_data.groupby(['sax_word',classification]).size().reset_index(name='count')
            plot_data = plot_data.pivot(index='sax_word',columns=classification,values='count')
            plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False)
            plot_data = plot_data.fillna(0)
            plot_data = plot_data.T
            x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
            divnorm = colors.TwoSlopeNorm(vcenter=0)
            _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
            _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count',fraction=0.025,pad=0.01)
            ax.tick_params('x',which='both',rotation=90)
            ax.set_ylabel(classification)
            ax.set_xlabel('Word')
            ax.set_title(schedule_name(filepath))

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_sax_word_{classification}_frequency_count_heatmap.png'), facecolor='white', bbox_inches='tight')
        plt.close()

    # word occurence heatmpa across building to pick out groups of buildings on a temporal scale
    fig, ax = plt.subplots(row_count,column_count,figsize=(18,25*row_count))

    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        words = plot_data.groupby(['sax_word']).size().reset_index()[['sax_word']].copy()
        words['word_id'] = words.index + 1
        plot_data = pd.merge(plot_data,words,on='sax_word',how='left')
        plot_data = plot_data.pivot(index='metadata_id',columns='date',values='word_id')
        plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
        x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
        divnorm = colors.TwoSlopeNorm(vcenter=0)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
        _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Word ID',fraction=0.025,pad=0.01)
        ax.tick_params('x',which='both',rotation=90)
        ax.set_ylabel('Building')
        ax.set_xlabel('Timestep')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_sax_word_norm_count_building_timeseries_heatmap.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # word count per building heatmap becasue it's what we want to cluster init so why not!
    fig, ax = plt.subplots(row_count,column_count,figsize=(18,25*row_count))

    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        plot_data = plot_data.groupby(['metadata_id','sax_word']).size().reset_index(name='count')
        plot_data = plot_data.pivot(index='metadata_id',columns='sax_word',values='count')
        plot_data = plot_data.fillna(0)
        plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
        x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
        z = minmax_scale(z)
        divnorm = colors.TwoSlopeNorm(vcenter=0)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
        _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count (Normalized)',fraction=0.025,pad=0.01)
        ax.tick_params('x',which='both',rotation=90)
        ax.set_ylabel('Building')
        ax.set_xlabel('Word')
        ax.set_yticks([])
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_sax_word_norm_count_building_heatmap.png'), facecolor='white', bbox_inches='tight')
    plt.close()

# manipulate()

'''NOTE:
- The schedules with wide variance e.g. appliance loads have about 16 unique words. 
    Occupancy and some others have the larger number of unique words but still under 100 which is much less than using the entire timeseries as a clsuter datapoint.
    Maybe, PCA can be applied as a second compression to further reduce the dimensionality?
- The heat map that shows the normalized word counts per building shows variance across the building which is a good sign that the clusterin might pick out building groups
    (fingers crossed!).
'''
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION:
 Perform PCA transformation on the per building unqiue word counts data set to reduce the feature space of
 those schedules that are large and extract the variables that best represent the variance in the word counts.
 The features are used in the clustering analysis to extract the different building classes.'''
 
def manipulate():
    explained_variance_ratio_data = {}

    for i, filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_SAX_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        plot_data = plot_data.groupby(['metadata_id','sax_word']).size().reset_index(name='count')
        plot_data = plot_data.pivot(index='metadata_id',columns='sax_word',values='count')
        plot_data = plot_data.fillna(0)
        scaler = StandardScaler()
        x = plot_data.values
        scaler = scaler.fit(x)
        x = scaler.transform(x)
        pca = PCA(n_components=None)
        pca = pca.fit(x)
        explained_variance_ratio_data[schedule_name(filepath)] = list(pca.explained_variance_ratio_)
        component_data = pd.DataFrame(pca.transform(x), columns=range(0,plot_data.shape[1]))
        index_data = plot_data.reset_index(drop=False)[['metadata_id']].copy()
        index_data = pd.concat([index_data,component_data],axis=1)
        index_data.to_pickle(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        component_data.columns = [str(c) for c in component_data.columns]
        keys = plot_data.columns.tolist()
        plot_data = pd.concat([plot_data.reset_index(drop=False),component_data],axis=1)
        plot_data = plot_data.corr('pearson')
        plot_data = plot_data.loc[component_data.columns.tolist()][keys].copy()
        plot_data = plot_data.reset_index(drop=True)
        plot_data['component'] = plot_data.index
        plot_data.to_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}_correlation.csv'),index=False)
    
    write_json(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'),explained_variance_ratio_data)
    explained_variance_ratio_data = read_json(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'))

    # plots
    # components per schedule
    _, ax = plt.subplots(1,1,figsize=(4,8))
    y1 = [threshold_satisfaction_index(v) + 1 for _, v in explained_variance_ratio_data.items()]
    y2 = [len(v) for _, v in explained_variance_ratio_data.items()]
    x = list(range(len(y1)))
    ax.barh(x,y1,color=['blue'],label=f'{PCA_CUMMULATIVE_VARIANCE_THRESHOLD*100}%')
    ax.barh(x,y2,color=['blue'],alpha=0.2,label=f'100%')
    ax.set_yticks(x)
    ax.set_yticklabels(list(explained_variance_ratio_data.keys()))
    ax.legend(title='Explained variance')
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_threshold_satisfaction_components.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # explained variance ratio
    row_count = len(SCHEDULE_DATA_FILEPATHS)
    column_count = 1
    fig, _ = plt.subplots(row_count, column_count, figsize=(18,2.5*row_count))

    for i, (ax, (column, y2)) in enumerate(zip(fig.axes, explained_variance_ratio_data.items())):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        y1 = [sum(y2[0:i]) for i in range(len(y2))]
        x = list(range(0,len(y1)))
        ax.bar(x,y1,color='grey',label='Previous Sum')
        ax.bar(x,y2,bottom=y1,color='blue',label='Current')
        ax.axhline(PCA_CUMMULATIVE_VARIANCE_THRESHOLD,color='red',linestyle='--',label='Threshold')
        index = threshold_satisfaction_index(y2) + 0.5
        ax.axvline(index,color='red',linestyle='--',)
        ax.set_ylabel('Cummulative explained\nvariance ratio')
        ax.set_xlabel('Component')
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_title(column)
        ax.legend()

    plt.tight_layout()
    fig.align_ylabels()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_explained_variance_ratio.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # distribution of components
    fig, _ = plt.subplots(row_count, column_count, figsize=(18,2.5*row_count))

    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        variance_ratio_data = explained_variance_ratio_data[schedule_name(filepath)]
        index = threshold_satisfaction_index(variance_ratio_data) + 1.5
        columns = list(range(0,len(variance_ratio_data)))
        ax.boxplot(plot_data[columns].values)
        ax.set_ylabel('Value')
        ax.set_xlabel('Component')
        ax.axvline(index, color='red', linestyle='--')
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    fig.align_ylabels()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_component_distribution.png'), facecolor='white', bbox_inches='tight')
    plt.close()

    # correlation
    fig, ax = plt.subplots(row_count,column_count,figsize=(12,12*row_count))

    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}_correlation.csv'))
        plot_data = plot_data.set_index('component')
        x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
        divnorm = colors.TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1)
        _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
        explained_variance_data = explained_variance_ratio_data[schedule_name(filepath)]
        ax.axhline(threshold_satisfaction_index(explained_variance_data) + 0.5,color='red',linestyle='--')

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
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_correlation_heatmap.png'), facecolor='white', bbox_inches='tight')
    plt.close()

# manipulate()

'''NOTE:
- Able to reduce the schedules with over 60 words to about 50 words when 95% of variance is desired from PCA components.
- In general all schedules could use lower dimension than discovered in SAX to explain variance and use as clustering dataset.
'''
# END

# DATA MANIPULATION ****************************************************************
'''DESCRIPTION:
 Perform clustering on the PCA components extracted from SAX dataset to identify characteristic schedules.'''

def manipulate():
    # explained_variance_ratio_data = read_json(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.json'))

    # # prep cluster data
    # for i, filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
    #     print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
    #     plot_data = pd.read_pickle(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
    #     variance_ratio_data = explained_variance_ratio_data[schedule_name(filepath)]
    #     index = threshold_satisfaction_index(variance_ratio_data)
    #     pca_columns = list(range(0,index + 1))
    #     plot_data = plot_data[['metadata_id'] + pca_columns].copy()
    #     plot_data = plot_data.set_index('metadata_id').sort_index()
        
    #     # # normalize before clustering
    #     # scaler = StandardScaler()
    #     # scaler = scaler.fit(plot_data.values)
    #     # plot_data[pca_columns] = scaler.transform(plot_data.values)
    #     # # plot_data[pca_columns] = minmax_scale(plot_data.values)
        
    #     # save cluster dataset
    #     plot_data.to_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))

    # # KMeans fitting  
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     kmeans_result = {'n_clusters':list(range(2,101)),'schedules': {}}
    #     work_order = [[],kmeans_result['n_clusters']*len(SCHEDULE_DATA_FILEPATHS)]

    #     for f in SCHEDULE_DATA_FILEPATHS:
    #         work_order[0] += [f]*len(kmeans_result['n_clusters'])
    #         kmeans_result['schedules'][schedule_name(f)] = {'scores':{'sse':[],'calinski_harabasz':[],'silhouette':[]},'labels':[]}
    #         results = executor.map(fit_kmeans,*work_order)

    #     for r in results:
    #         print('finished fitting schedule:',r[0],'n_clusters:',r[1])
    #         kmeans_result['schedules'][r[0]]['labels'].append(r[2])
    #         kmeans_result['schedules'][r[0]]['scores']['sse'].append(r[3]['sse'])
    #         kmeans_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[3]['calinski_harabasz'])
    #         kmeans_result['schedules'][r[0]]['scores']['silhouette'].append(r[3]['silhouette'])
    
    #     write_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'kmeans_result_without_norm.json'),kmeans_result)

    # # DBSCAN
    # # Use KNN to determine eps
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     knn_result = {'n_neighbors':2,'schedules':{}}
    #     work_order = [SCHEDULE_DATA_FILEPATHS,[knn_result['n_neighbors']]*len(SCHEDULE_DATA_FILEPATHS)]
    #     results = executor.map(fit_knn,*work_order)

    #     for r in results:
    #         print('finished fitting schedule:',r[0])
    #         knn_result['schedules'][r[0]] = {}
    #         knn_result['schedules'][r[0]]['distance'] = r[1]
       
    #     write_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json'),knn_result)

    # # Now apply DBSCAN
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     dbscan_eps = read_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_eps.json'))
    #     work_order = [[],[]]
    #     dbscan_result = {'schedules':{}}
    #     step = 0.1

    #     for filepath in dbscan_eps:
    #         eps = dbscan_eps[filepath]
    #         distance = read_json(
    #             os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json')
    #         )['schedules'][schedule_name(filepath)]['distance']
    #         mini, maxi = max(round(min(distance),1),step), round(max(distance),1)
    #         eps = np.arange(mini,maxi,step).tolist()
    #         work_order[1] += eps
    #         work_order[0] += [filepath]*len(eps)
    #         dbscan_result['schedules'][schedule_name(filepath)] = {
    #             'eps':[],
    #             'min_samples':[],
    #             'scores':{'sse':[],'calinski_harabasz':[],'silhouette':[]},
    #             'labels':[]
    #         }
    
    #     results = executor.map(fit_dbscan,*work_order)

    #     for r in results:
    #         if r[3] is None:
    #             print('UNSUCCESSFUL fitting schedule:',r[0],'eps:',r[1],'min_samples:',r[2])
    #         else:
    #             print('finished fitting schedule:',r[0],'eps:',r[1],'min_samples:',r[2])
    #             dbscan_result['schedules'][r[0]]['eps'].append(r[1])
    #             dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
    #             dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
    #             dbscan_result['schedules'][r[0]]['labels'].append(r[3])
    #             dbscan_result['schedules'][r[0]]['scores']['sse'].append(r[4]['sse'])
    #             dbscan_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[4]['calinski_harabasz'])
    #             dbscan_result['schedules'][r[0]]['scores']['silhouette'].append(r[4]['silhouette'])
            
    #     write_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'),dbscan_result)
        
    # # plot
    kmeans_result = read_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'kmeans_result_without_norm.json'))
    knn_result = read_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json'))
    dbscan_result = read_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'))

    # # cluster dataset
    # row_count = len(SCHEDULE_DATA_FILEPATHS)
    # column_count = 1
    # fig, ax = plt.subplots(row_count,column_count,figsize=(18,25*row_count))

    # for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
    #     print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
    #     plot_data = pd.read_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))

    #     plot_data = plot_data.sort_values(plot_data.columns.tolist(),ascending=False).reset_index(drop=True)
    #     x, y, z = plot_data.columns.tolist(), plot_data.index.tolist(), plot_data.values
    #     divnorm = colors.TwoSlopeNorm(vcenter=0)
    #     _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
    #     _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label='Count (Normalized)',fraction=0.025,pad=0.01)
    #     ax.tick_params('x',which='both',rotation=90)
    #     ax.set_ylabel('Building')
    #     ax.set_xlabel('PCA Component')
    #     ax.set_yticks([])
    #     ax.set_title(schedule_name(filepath))

    # plt.tight_layout()
    # plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_pca_cluster_dataset_building_heatmap.png'), facecolor='white', bbox_inches='tight')
    # plt.close()

    # # KMeans scores
    # row_count = len(SCHEDULE_DATA_FILEPATHS)
    # column_count = len(list(kmeans_result['schedules'].values())[0]['scores'])
    # fig, axs = plt.subplots(row_count, column_count, figsize=(5*column_count,2*row_count))
    # x = kmeans_result['n_clusters']

    # for i, (schedule, schedule_data) in enumerate(kmeans_result['schedules'].items()):
    #     for j, (score, y) in enumerate(schedule_data['scores'].items()):
    #         axs[i,j].plot(x,y)
    #         axs[i,j].set_title(f'{schedule}')
    #         axs[i,j].set_xlabel('n_clusters')
    #         axs[i,j].set_ylabel(score)

    # # DBSCAN  scores
    # row_count = len(SCHEDULE_DATA_FILEPATHS)
    # column_count = len(list(dbscan_result['schedules'].values())[0]['scores'])
    # fig, axs = plt.subplots(row_count, column_count, figsize=(5*column_count,2*row_count))

    # for i, (schedule, schedule_data) in enumerate(dbscan_result['schedules'].items()):
    #     x = schedule_data['eps']
    #     y2 = [len(set(l)) for l in schedule_data['labels']]

    #     for j, (score, y1) in enumerate(schedule_data['scores'].items()):
    #         axs[i,j].plot(x,y1,color='blue')
    #         axs[i,j].set_title(f'{schedule}')
    #         axs[i,j].set_xlabel('eps')
    #         axs[i,j].set_ylabel(score,color='blue')
    #         ax2 = axs[i,j].twinx()
    #         ax2.plot(x,y2,color='red')
    #         ax2.set_ylabel('clusters',color='red')
    
    # plt.tight_layout()
    # fig.align_ylabels()
    # plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_dbscan_scores.png'), facecolor='white', bbox_inches='tight')
    # plt.close()

    # DBSCAN label against timeseries
    row_count = len(SCHEDULE_DATA_FILEPATHS)
    column_count = 1
    fig, ax = plt.subplots(row_count,column_count,figsize=(8,8*row_count))

    for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
        print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
        plot_data = pd.read_pickle(os.path.join(SCHEDULE_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
        plot_data = plot_data.pivot(index='metadata_id',columns='timestep',values=schedule_name(filepath))
        scores = dbscan_result['schedules'][schedule_name(filepath)]['scores']['calinski_harabasz']
        max_score_index = scores.index(max(scores))
        labels = dbscan_result['schedules'][schedule_name(filepath)]['labels'][max_score_index]
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
        ax.set_title(schedule_name(filepath))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_building_dbscan_cluster_timeseries_heatmap.png'), facecolor='white', bbox_inches='tight')
    plt.close()


    # # KNN distance
    # row_count = len(SCHEDULE_DATA_FILEPATHS)
    # column_count = 1
    # fig, _ = plt.subplots(row_count, column_count, figsize=(6,3*row_count))

    # for i, (ax, (schedule, schedule_data)) in enumerate(zip(fig.axes, knn_result['schedules'].items())):
    #     y = sorted(schedule_data['distance'])
    #     x = list(range(len(y)))
    #     ax.plot(x,y)
    #     ax.set_ylabel(f'distance to\nnearest neighbor')
    #     ax.yaxis.tick_right()
    #     ax.yaxis.set_label_position('right')
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #     ax.set_xlabel('sorted index')
    #     ax.grid(True)
    #     ax.set_title(schedule)

    # plt.tight_layout()
    # plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_knn_sorted_distance.png'), facecolor='white', bbox_inches='tight')
    # plt.close()

def fit_kmeans(filepath,n_clusters):
    x = pd.read_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl')).values
    result = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    scores = {
        'sse':result.inertia_,
        'calinski_harabasz':calinski_harabasz_score(x,result.labels_),
        'silhouette':silhouette_score(x,result.labels_)
    }
    return schedule_name(filepath), n_clusters, result.labels_.tolist(), scores

def fit_knn(filepath,n_neighbors):
    x = pd.read_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl')).values
    distance, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(x)
    return schedule_name(filepath), distance[:,1].tolist()

def fit_dbscan(filepath,eps,min_samples=None):
    x = pd.read_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl')).values
    min_samples = int(x.shape[1]*2) if min_samples is None else min_samples
    result = DBSCAN(eps=eps,min_samples=min_samples).fit(x)
    try:
        scores = {
            'sse':get_sse(x,result.labels_.tolist()),
            'calinski_harabasz':calinski_harabasz_score(x,result.labels_),
            'silhouette':silhouette_score(x,result.labels_)
        }
        return schedule_name(filepath), eps, min_samples, result.labels_.tolist(), scores
    except ValueError as e:
        return schedule_name(filepath), eps, min_samples, None

def get_sse(x,labels):
    df = pd.DataFrame(x)
    df['label'] = labels
    df = df.groupby('label').apply(lambda gr:
        (gr.iloc[:,0:-1] - gr.iloc[:,0:-1].mean())**2
    )
    sse = df.sum().sum()
    return sse

manipulate()

'''NOTE:
'''
# END