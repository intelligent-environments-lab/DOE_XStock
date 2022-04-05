# import concurrent.futures
# import logging
# import logging.handlers
# import os
# import math
# from pathlib import Path
# from matplotlib import cm
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np
# import pandas as pd
# from saxpy.alphabet import cuts_for_asize
# from saxpy.paa import paa
# from saxpy.sax import ts_to_string
# import seaborn as sns
# from sklearn import cluster
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import calinski_harabasz_score, silhouette_score
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler, minmax_scale
# from doe_xstock.database import SQLiteDatabase
# from doe_xstock.utilities import read_json, split_lines, write_json

# schedules = ['baths','clothes_dryer','clothes_washer','cooking_range','dishwasher','lighting_interior','occupants','showers']
# data_directory = Path('../schedule_data')
# pca_data_directory = Path('../pca_data')
# os.makedirs(pca_data_directory,exist_ok=True)
# cluster_data_directory = Path('../cluster_data')
# os.makedirs(cluster_data_directory,exist_ok=True)
# figures_directory = Path('../figures')
# os.makedirs(figures_directory,exist_ok=True)
# schedule_filepaths = [os.path.join(data_directory,f'{s}.pkl') for s in schedules]
# pca_explained_variance = 0.95
# date_range = pd.DataFrame({'timestamp':pd.date_range('2017-01-01','2017-12-31 23:00:00', freq='H')})
# date_range['timestep'] = date_range.index
# date_range['month'] = date_range['timestamp'].dt.month
# date_range['week'] = date_range['timestamp'].dt.isocalendar().week
# date_range['date'] = date_range['timestamp'].dt.normalize()
# date_range['day_of_week'] = date_range['timestamp'].dt.weekday
# date_range.loc[date_range[
#     'day_of_week'] == 6, 'week_of'
# ] = date_range.loc[date_range['day_of_week'] == 6]['timestamp'].dt.normalize()
# date_range['week_of'] = date_range['week_of'].ffill()

# def main():
#     # save_schedule_pca_data()
#     # save_schedule_cluster_data()
#     # save_schedule_dbscan_data()
#     # plot_schedule_dbscan()
#     save_building_schedule_cluster_data()
#     # plot_building_schedule_cluster()

# def plot_building_schedule_cluster():
#     dbscan_result = read_json(os.path.join(cluster_data_directory,f'building_dbscan_result.json'))

#     # DBSCAN  scores
#     row_count = 1
#     column_count = len(dbscan_result['scores'])
#     fig, _ = plt.subplots(row_count, column_count, figsize=(5*column_count,2*row_count))
#     x = dbscan_result['eps']
#     y2 = [len(set(l)) for l in dbscan_result['labels']]

#     for i, (ax, (score, y1)) in enumerate(zip(fig.axes,dbscan_result['scores'].items())):
#         plot_data = pd.DataFrame({'x':x,'y2':y2,'y1':y1}).sort_values('x')
#         x, y1, y2 = plot_data['x'], plot_data['y1'], plot_data['y2']
#         ax.plot(x,y1,color='blue')
#         ax.set_xlabel('eps')
#         ax.set_ylabel(score,color='blue')
#         ax2 = ax.twinx()
#         ax2.plot(x,y2,color='red')
#         ax2.set_ylabel('clusters',color='red')
    
#     plt.tight_layout()
#     fig.align_ylabels()
#     plt.savefig(os.path.join(figures_directory,f'building_dbscan_scores.png'), facecolor='white', bbox_inches='tight')
#     plt.close()


#     # plot_data = pd.read_pickle(os.path.join(cluster_data_directory,'building_schedule_cluster_with_outliers_grouped.pkl'))
#     # plot_data = plot_data.sort_values(plot_data.columns.tolist()).reset_index(drop=True)
    
#     # for c in plot_data.columns:
#     #     plot_data.loc[plot_data[c]>=0,c] += 1

#     # fig, ax = plt.subplots(1,1,figsize=(10,10))
#     # x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
#     # divnorm = colors.TwoSlopeNorm(vcenter=0)
#     # _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0.0025,clip_on=False)
#     # _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)
#     # ax.tick_params('x',which='both',rotation=45)
#     # ax.set_ylabel('Building')
#     # ax.set_xlabel('Schedule')
#     # ax.set_yticks([])
#     # plt.tight_layout()
#     # plt.savefig(os.path.join(figures_directory,f'building_cluster_grid_heatmap'), facecolor='white', bbox_inches='tight')
#     # plt.close()

# def save_building_schedule_cluster_data():
#     dbscan_result = read_json(os.path.join(cluster_data_directory,f'dbscan_result.json'))
#     data = {}
    
#     for i, (schedule, schedule_data) in enumerate(dbscan_result['schedules'].items()):
#         scores = schedule_data['scores']['calinski_harabasz']
#         max_score_index = scores.index(max(scores))
#         data[schedule] = schedule_data['labels'][max_score_index]

#     # grouped
#     data['metadata_id'] = pd.read_pickle(os.path.join(cluster_data_directory,f'{schedule}.pkl')).index.tolist()
#     data = pd.DataFrame(data).set_index('metadata_id')

#     # for c in data.columns:
#     #     plot_data = data[[c]].copy()
#     #     print(plot_data.groupby(c).size().reset_index())

#     # assert False

#     data.to_pickle(os.path.join(cluster_data_directory,'building_schedule_cluster_with_outliers_grouped.pkl'))

#     # ungrouped
#     maximum_labels = data.max(axis=0).to_dict()
#     minimum_labels = data.min(axis=0).to_dict()

#     for k, v in minimum_labels.items():
#         if v == -1:
#             first_ix = maximum_labels[k] + 1
#             length = data[data[k] == -1].shape[0]
#             last_ix = first_ix + length
#             data.loc[data[k] == -1, k] = list(range(first_ix,last_ix))
#         else:
#             continue
    
#     data.to_pickle(os.path.join(cluster_data_directory,'building_schedule_cluster_with_outliers_ungrouped.pkl'))

#     # dummy variables
#     data = pd.get_dummies(data.astype(str))
#     data.to_pickle(os.path.join(cluster_data_directory,'building_schedule_cluster_with_dummy_vars.pkl'))

#     # clustering
#     cluster_data_filepath = os.path.join(cluster_data_directory,'building_schedule_cluster_with_dummy_vars.pkl')
#     dbscan_result = {'schedules':{}}
#     step = 0.01
#     n_neighbors = 2

#      # fit KNN
#     x = pd.read_pickle(cluster_data_filepath).values
#     distance, _ = NearestNeighbors(n_neighbors=n_neighbors,metric='jaccard',n_jobs=-1).fit(x).kneighbors(x)
#     distance = distance[:,1].tolist()

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # fit DBSCAN
#         work_order = [[],[],[],[],[]]
#         mini, maxi = max(round(min(distance),1),step), round(max(distance),1)
#         eps = np.arange(mini,maxi,step).tolist()
#         work_order[0] += [cluster_data_filepath]*len(eps)
#         work_order[1] += [cluster_data_filepath]*len(eps)
#         work_order[2] += eps
#         work_order[3] += [2]*len(eps)
#         work_order[4] += ['jaccard']*len(eps)
#         dbscan_result = {
#             'eps':[],
#             'min_samples':[],
#             'scores':{'calinski_harabasz':[],'silhouette':[]},
#             'labels':[]
#         }
#         results = executor.map(fit_dbscan,*work_order)

#         for r in results:
#             if r[3] is None:
#                 print(f'UNSUCCESSFUL fitting: eps: {r[1]}, min_samples: {r[2]}')
#             else:
#                 print(f'finished fitting: eps: {r[1]}, min_samples: {r[2]}')
#                 dbscan_result['eps'].append(r[1])
#                 dbscan_result['min_samples'].append(r[2])
#                 dbscan_result['labels'].append(r[3])
#                 dbscan_result['scores']['calinski_harabasz'].append(r[4]['calinski_harabasz'])
#                 dbscan_result['scores']['silhouette'].append(r[4]['silhouette'])
                
#     write_json(os.path.join(cluster_data_directory,f'building_dbscan_result.json'),dbscan_result)

# def plot_schedule_dbscan():          
#     dbscan_result = read_json(os.path.join(cluster_data_directory,f'dbscan_result.json'))

#     # DBSCAN  scores
#     row_count = len(schedules)
#     column_count = len(list(dbscan_result['schedules'].values())[0]['scores'])
#     fig, axs = plt.subplots(row_count, column_count, figsize=(5*column_count,2*row_count))

#     for i, (schedule, schedule_data) in enumerate(dbscan_result['schedules'].items()):
#         x = schedule_data['eps']
#         y2 = [len(set(l)) for l in schedule_data['labels']]

#         for j, (score, y1) in enumerate(schedule_data['scores'].items()):
#             plot_data = pd.DataFrame({'x':x,'y2':y2,'y1':y1}).sort_values('x')
#             x, y1, y2 = plot_data['x'], plot_data['y1'], plot_data['y2']
#             axs[i,j].plot(x,y1,color='blue')
#             axs[i,j].set_title(f'{schedule}')
#             axs[i,j].set_xlabel('eps')
#             axs[i,j].set_ylabel(score,color='blue')
#             ax2 = axs[i,j].twinx()
#             ax2.plot(x,y2,color='red')
#             ax2.set_ylabel('clusters',color='red')
    
#     plt.tight_layout()
#     fig.align_ylabels()
#     plt.savefig(os.path.join(figures_directory,f'schedule_dbscan_scores.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

#     # DBSCAN label against timeseries
#     row_count = len(schedules)
#     column_count = 1
#     fig, axs = plt.subplots(row_count,column_count,figsize=(10,10*row_count))

#     for i, (ax, s) in enumerate(zip(fig.axes, schedules)):
#         plot_data = pd.read_pickle(os.path.join(data_directory,f'{s}.pkl'))
#         plot_data = plot_data.pivot(index='metadata_id',columns='timestep',values=s)
#         scores = dbscan_result['schedules'][s]['scores']['calinski_harabasz']
#         max_score_index = scores.index(max(scores))
#         labels = dbscan_result['schedules'][s]['labels'][max_score_index]
#         columns = plot_data.columns.tolist()
#         plot_data['label'] = labels
#         plot_data = plot_data.sort_values(['label']+columns).reset_index(drop=True)
#         indices = plot_data.groupby(['label']).size().reset_index(name='count')['count'].tolist()
#         indices = [sum(indices[0:i + 1]) for i in range(len(indices))]
#         plot_data = plot_data.drop(columns=['label'])
#         x, y, z = plot_data.columns.tolist(), plot_data.index, plot_data.values
#         divnorm = colors.TwoSlopeNorm(vcenter=0)
#         _ = ax.pcolormesh(x,y,z,shading='nearest',norm=divnorm,cmap='coolwarm',edgecolors='white',linewidth=0,clip_on=False)
#         _ = fig.colorbar(cm.ScalarMappable(cmap='coolwarm',norm=divnorm),ax=ax,orientation='vertical',label=None,fraction=0.025,pad=0.01)

#         for label, index in zip(sorted(set(labels)), indices):
#             ax.axhline(index + 0.5,color='black',linestyle='--',linewidth=4,clip_on=False)
#             ax.text(-80,index,f'C:{label}',color='black',fontsize=12,ha='right',va='center',fontweight='medium')

#         ax.tick_params('x',which='both',rotation=0)
#         ax.set_ylabel('Building')
#         ax.set_xlabel('Timestep')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(s)

#     plt.tight_layout()
#     plt.savefig(os.path.join(figures_directory,f'schedule_building_dbscan_cluster_timeseries_heatmap.png'), facecolor='white', bbox_inches='tight')
#     plt.close()

# def save_schedule_dbscan_data():
#     dbscan_result = {'schedules':{}}
#     knn_result = {}
#     step = 0.1

#     # fit KNN
#     for s in schedules:
#         print(f'fitting KNN: {s}')

#         n_neighbors = 2
#         x = pd.read_pickle(os.path.join(cluster_data_directory,f'{s}.pkl')).values
#         distance, _ = NearestNeighbors(n_neighbors=n_neighbors,n_jobs=-1).fit(x).kneighbors(x)
#         knn_result[s] = distance[:,1].tolist()

#     write_json(os.path.join(cluster_data_directory,f'knn_result.json'),knn_result)

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # fit DBSCAN
#         work_order = [[],[],[],[],[]]

#         for s, distance in knn_result.items():
#             mini, maxi = max(round(min(distance),1),step), round(max(distance),1)
#             eps = np.arange(mini,maxi,step).tolist()
#             work_order[0] += [s]*len(eps)
#             work_order[1] += [os.path.join(cluster_data_directory,f'{s}.pkl')]*len(eps)
#             work_order[2] += eps
#             work_order[3] += [2]*len(eps)
#             work_order[4] += ['euclidean']*len(eps)
#             dbscan_result['schedules'][s] = {
#                 'eps':[],
#                 'min_samples':[],
#                 'scores':{'calinski_harabasz':[],'silhouette':[]},
#                 'labels':[]
#             }
        
#         results = executor.map(fit_dbscan,*work_order)

#         for r in results:
#             if r[3] is None:
#                 print(f'UNSUCCESSFUL fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
#             else:
#                 print(f'finished fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
#                 dbscan_result['schedules'][r[0]]['eps'].append(r[1])
#                 dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
#                 dbscan_result['schedules'][r[0]]['labels'].append(r[3])
#                 dbscan_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[4]['calinski_harabasz'])
#                 dbscan_result['schedules'][r[0]]['scores']['silhouette'].append(r[4]['silhouette'])
                
#     write_json(os.path.join(cluster_data_directory,f'dbscan_result.json'),dbscan_result)

# def fit_dbscan(schedule,filepath,eps,min_samples,metric):
#     x = pd.read_pickle(filepath).values
#     result = DBSCAN(eps=eps,min_samples=min_samples,metric=metric).fit(x)
#     try:
#         scores = {
#             'calinski_harabasz':calinski_harabasz_score(x,result.labels_),
#             'silhouette':silhouette_score(x,result.labels_)
#         }
#         return schedule, eps, min_samples, result.labels_.tolist(), scores
    
#     except ValueError as e:
#         return schedule, eps, min_samples, None

# def save_schedule_cluster_data():
#     for s in schedules:
#         data = pd.read_pickle(os.path.join(pca_data_directory,f'{s}.pkl'))
#         data[data.columns.tolist()] = minmax_scale(data.values)
#         data.to_pickle(os.path.join(cluster_data_directory,f'{s}.pkl'))

# def save_schedule_pca_data():
#     for s, f in zip(schedules,schedule_filepaths):
#         data = pd.read_pickle(f)
#         # data = data.merge(date_range,on='timestep',how='left')
#         # data = data.pivot(index=['metadata_id','date'],columns='hour',values=s)
#         data = data.pivot(index='metadata_id',columns='timestep',values=s)
        
#         # standardize
#         scaler = StandardScaler()
#         x = data.values
#         scaler = scaler.fit(x)
#         x = scaler.transform(x)
#         data[data.columns.tolist()] = x

#         # fit pca
#         pca = PCA(n_components=pca_explained_variance,svd_solver='full')
#         pca = pca.fit(x)

#         # save
#         index = data.index.tolist()
#         data = pd.DataFrame(pca.transform(x),)
#         data['metadata_id'] = index
#         data = data.set_index('metadata_id')
#         print('schedule:',s,'PCA components:',data.shape[1])
#         data.to_pickle(os.path.join(pca_data_directory,f'{s}.pkl'))

# if __name__ == '__main__':
#     main()