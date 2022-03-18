import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import pandas as pd
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
from saxpy.sax import ts_to_string
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
SCHEDULE_DATA_FILEPATHS = [os.path.join(SCHEDULE_DATA_DIRECTORY,f) for f in os.listdir(SCHEDULE_DATA_DIRECTORY) if f.endswith('.pkl')]
SCHEDULE_DATA_FILEPATHS = sorted(SCHEDULE_DATA_FILEPATHS)
SEASONS = {
    1:'winter',2:'winter',12:'winter',
    3:'spring',4:'spring',5:'spring',
    6:'summer',7:'summer',8:'summer',
    9:'fall',10:'fall',11:'fall',
}
PCA_CUMMULATIVE_VARIANCE_THRESHOLD = 0.9
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
 Once that has been done, the data is melted to longform and SAX transformation is applied to a sequence size equalling 
the number of principal components. While we are aware that some schedules don't even show variance day-of-week wise, this approach is more 
generalizable when analyzing data from other locations.'''

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
    # collective word frequency
    

manipulate()

'''NOTE:
- The PCA transformation on the daily profile hourly variables shows that fewer than 24 components are able to
explain the variance in the schedules in most cases and in some cases, just 1 component is needed.
- PCA also reveals what was seen in the box plots that the variance occurs mostly during the day as early hours of the
day have weak correlation with the top components.
'''
# END


# SCRATCH ****************************************************************

# # DATA MANIPULATION ****************************************************************
# '''DESCRIPTION: PCA analysis of daily profiles dataset of m * n where m is the number of days in the dataset (rows) and n is the number of hours
# in a day (columns). This way we try to capture the variables that explain the hourly variance in the data (horizontal) and seasonal variance (vertical).'''

# def manipulate():
#     explained_variance_ratio_data = {}
    
#     # calculate PCA
#     for i, filepath in enumerate(SCHEDULE_DATA_FILEPATHS):
#         print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
#         plot_data = pd.merge(pd.read_pickle(filepath),DATE_RANGE,on='timestep')
#         plot_data = plot_data.pivot(index=['metadata_id','date'],columns='hour',values=schedule_name(filepath))
#         plot_data = plot_data.dropna()
#         x = plot_data.values
#         # normalize
#         scaler = StandardScaler()
#         scaler = scaler.fit(x)
#         x = scaler.transform(x)
#         # ... then apply PCA
#         pca = PCA(n_components=None)
#         pca.fit(x)
#         explained_variance_ratio_data[schedule_name(filepath)] = list(pca.explained_variance_ratio_)
#         component_data = pd.DataFrame(pca.transform(x), columns=range(0,plot_data.shape[1]))
#         index_data = plot_data.reset_index(drop=False)[['metadata_id','date']].copy()
#         pd.concat([index_data,component_data],axis=1).to_pickle(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
#         component_data.columns = [str(c) for c in component_data.columns]
#         keys = plot_data.columns.tolist()
#         plot_data = pd.concat([plot_data.reset_index(drop=False),component_data],axis=1)
#         plot_data = plot_data.corr('pearson')
#         plot_data = plot_data.loc[component_data.columns.tolist()][keys].copy()
#         plot_data = plot_data.reset_index(drop=True)
#         plot_data['component'] = plot_data.index
#         plot_data.to_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}_correlation.csv'),index=False)
    
#     pd.DataFrame(explained_variance_ratio_data).to_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.csv'),index=False)
#     explained_variance_ratio_data = pd.read_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'explained_variance_ratio.csv'))

#     # plots
#     # explained variance ratio
#     row_count = len(SCHEDULE_DATA_FILEPATHS)
#     column_count = 1
#     fig, _ = plt.subplots(row_count, column_count, figsize=(18,2.5*row_count))

#     for i, (ax, column) in enumerate(zip(fig.axes, explained_variance_ratio_data.columns)):
#         print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
#         y2 = explained_variance_ratio_data[column].tolist()
#         y1 = [sum(y2[0:i]) for i in range(len(y2))]
#         x = list(range(0,len(y1)))
#         ax.bar(x,y1,color='grey',label='Previous Sum')
#         ax.bar(x,y2,bottom=y1,color='blue',label='Current')
#         ax.axhline(PCA_CUMMULATIVE_VARIANCE_THRESHOLD,color='red',linestyle='--',label='Threshold')
#         ax.set_ylabel('Cummulative explained\nvariance ratio')
#         ax.set_xlabel('Component')
#         ax.set_xticks(x)
#         ax.set_xticklabels(x)
#         ax.set_title(column)
#         ax.legend()

#     plt.tight_layout()
#     fig.align_ylabels()
#     plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_explained_variance_ratio.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

#     # distribution of components
#     fig, _ = plt.subplots(row_count, column_count, figsize=(18,2.5*row_count))

#     for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
#         print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
#         plot_data = pd.read_pickle(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl'))
#         index = threshold_satisfaction_index(explained_variance_ratio_data[schedule_name(filepath)]) + 1.5
#         columns = list(range(0,24))
#         ax.boxplot(plot_data[columns].values)
#         ax.set_ylabel('Value')
#         ax.set_xlabel('Component')
#         ax.axvline(index, color='red', linestyle='--')
#         ax.set_title(schedule_name(filepath))

#     plt.tight_layout()
#     fig.align_ylabels()
#     plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_component_distribution.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

#     # correlation
#     fig, ax = plt.subplots(row_count,column_count,figsize=(12,12*row_count))

#     for i, (ax, filepath) in enumerate(zip(fig.axes, SCHEDULE_DATA_FILEPATHS)):
#         print(f'{i+1}/{len(SCHEDULE_DATA_FILEPATHS)}')
#         plot_data = pd.read_csv(os.path.join(SCHEDULE_PCA_DATA_DIRECTORY,f'{schedule_name(filepath)}_correlation.csv'))
#         plot_data = plot_data.set_index('component')
#         x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
#         divnorm = colors.TwoSlopeNorm(vcenter=0,vmin=-1,vmax=1)
#         _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.5)
#         ax.axhline(threshold_satisfaction_index(explained_variance_ratio_data[schedule_name(filepath)].tolist()) + 0.5,color='red',linestyle='--')

#         for k in range(len(y)):
#             for j in range(len(x)):
#                 value = z[k,j]
#                 value = round(value,2) if abs(value) >= 0.4 else None
#                 _ = ax.text(j,k,value,ha='center',va='center',color='black')

#         _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)
#         ax.tick_params('x',which='both',rotation=0)
#         ax.set_ylabel('Component')
#         ax.set_xlabel('Hour')
#         ax.set_xticks(x)
#         ax.set_yticks(y)
#         ax.set_title(schedule_name(filepath))

#     plt.tight_layout()
#     plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_correlation_heatmap.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

#     # components per schedule
#     _, ax = plt.subplots(1,1,figsize=(4,8))
#     y = [threshold_satisfaction_index(explained_variance_ratio_data[c].tolist()) + 1 for c in explained_variance_ratio_data.columns]
#     x = list(range(len(y)))
#     ax.barh(x,y)
#     ax.set_yticks(x)
#     ax.set_yticklabels(explained_variance_ratio_data.columns.to_list())
#     plt.savefig(os.path.join(FIGURES_DIRECTORY,f'travis_county_hourly_schedule_pca_threshold_satisfaction_components.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

# # manipulate()

# '''NOTE:
# - The PCA transformation on the daily profile hourly variables shows that fewer than 24 components are able to
# explain the variance in the schedules in most cases and in some cases, just 1 component is needed.
# - PCA also reveals what was seen in the box plots that the variance occurs mostly during the day as early hours of the
# day have weak correlation with the top components.
# '''
# # END