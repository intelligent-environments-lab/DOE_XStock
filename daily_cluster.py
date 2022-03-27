import concurrent.futures
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from doe_xstock.utilities import read_json, write_json

exclude = [
    'plug_loads_vehicle','vacancy','lighting_exterior_holiday',
    'fuel_loads_fireplace','fuel_loads_grill','fuel_loads_lighting','lighting_exterior','lighting_garage','plug_loads_well_pump',
    'clothes_dryer_exhaust','ceiling_fan','plug_loads'
]
SCHEDULE_DATA_DIRECTORY = '../schedule_data'
SCHEDULE_CLUSTER_DATA_DIRECTORY = '../schedule_daily_cluster_data'
schedule_data_filepaths = [
    os.path.join(SCHEDULE_DATA_DIRECTORY,f) for f in os.listdir(SCHEDULE_DATA_DIRECTORY) if f.endswith('.pkl') and f.split('.')[0] not in exclude
]
SEASONS = {
    1:'winter',2:'winter',12:'winter',
    3:'spring',4:'spring',5:'spring',
    6:'summer',7:'summer',8:'summer',
    9:'fall',10:'fall',11:'fall',
}
DATE_RANGE = pd.DataFrame({'timestamp':pd.date_range('2017-01-01','2017-12-31 23:00:00', freq='H')})
DATE_RANGE['timestep'] = DATE_RANGE.index
DATE_RANGE['month'] = DATE_RANGE['timestamp'].dt.month
DATE_RANGE['week'] = DATE_RANGE['timestamp'].dt.isocalendar().week
DATE_RANGE['date'] = DATE_RANGE['timestamp'].dt.normalize()
DATE_RANGE['day_of_week'] = DATE_RANGE['timestamp'].dt.weekday
DATE_RANGE.loc[DATE_RANGE[
    'day_of_week'] == 6, 'week_of'
] = DATE_RANGE.loc[DATE_RANGE['day_of_week'] == 6]['timestamp'].dt.normalize()
DATE_RANGE['week_of'] = DATE_RANGE['week_of'].ffill()
DATE_RANGE['season'] = DATE_RANGE['month'].map(lambda x: SEASONS[x])
FIGURES_DIRECTORY = 'figures/'
schedule_name = lambda x: x.split('/')[-1].split('.')[0]

def main():
    # # set up data and save
    # for i, f in enumerate(schedule_data_filepaths):
    #     print(f'{i+1}/{len(schedule_data_filepaths)}')
    #     data = pd.read_pickle(f)
    #     data = pd.merge(data,DATE_RANGE,on='timestep',how='left')
    #     data = data.pivot(index=['metadata_id','date'],columns='hour',values=schedule_name(f))
    #     data = data.sort_index(level=['metadata_id','date'])
    #     data.to_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(f)}.pkl'))

    # run knn 
    run_knn()

    # plot plot



    # run dbscan

    # plot heatmap showing cluster assignments


def run_knn():
    # Use KNN to determine eps
    knn_result = {'n_neighbors':2,'schedules':{}}

    for f, n in zip(schedule_data_filepaths,[knn_result['n_neighbors']]*len(schedule_data_filepaths)):
        r = fit_knn(f,n)
        print(f'finished fitting schedule: {r[0]}')
        knn_result['schedules'][r[0]] = {}
        knn_result['schedules'][r[0]]['distance'] = r[1]
    
        write_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json'),knn_result)

def run_dbscan():
    # Now apply DBSCAN
    with concurrent.futures.ThreadPoolExecutor() as executor:
        dbscan_eps = read_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_eps.json'))
        work_order = [[],[]]
        dbscan_result = {'schedules':{}}
        step = 0.1

        for filepath in dbscan_eps:
            eps = dbscan_eps[filepath]
            distance = read_json(
                os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'knn_result.json')
            )['schedules'][schedule_name(filepath)]['distance']
            mini, maxi = max(round(min(distance),1),step), round(max(distance),1)
            eps = np.arange(mini,maxi,step).tolist()
            work_order[1] += eps
            work_order[0] += [filepath]*len(eps)
            dbscan_result['schedules'][schedule_name(filepath)] = {
                'eps':[],
                'min_samples':[],
                'scores':{'sse':[],'calinski_harabasz':[],'silhouette':[]},
                'labels':[]
            }
    
        results = executor.map(fit_dbscan,*work_order)

        for r in results:
            if r[3] is None:
                print(f'UNSUCCESSFUL fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
            else:
                print(f'finished fitting schedule: {r[0]}, eps: {r[1]}, min_samples: {r[2]}')
                dbscan_result['schedules'][r[0]]['eps'].append(r[1])
                dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
                dbscan_result['schedules'][r[0]]['min_samples'].append(r[2])
                dbscan_result['schedules'][r[0]]['labels'].append(r[3])
                dbscan_result['schedules'][r[0]]['scores']['sse'].append(r[4]['sse'])
                dbscan_result['schedules'][r[0]]['scores']['calinski_harabasz'].append(r[4]['calinski_harabasz'])
                dbscan_result['schedules'][r[0]]['scores']['silhouette'].append(r[4]['silhouette'])
            
        write_json(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'dbscan_result.json'),dbscan_result)

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

def fit_knn(filepath,n_neighbors):
    x = pd.read_pickle(os.path.join(SCHEDULE_CLUSTER_DATA_DIRECTORY,f'{schedule_name(filepath)}.pkl')).values
    distance, _ = NearestNeighbors(n_neighbors=n_neighbors,n_jobs=-1).fit(x).kneighbors(x)
    return schedule_name(filepath), distance[:,1].tolist()

def get_sse(x,labels):
    df = pd.DataFrame(x)
    df['label'] = labels
    df = df.groupby('label').apply(lambda gr:
        (gr.iloc[:,0:-1] - gr.iloc[:,0:-1].mean())**2
    )
    sse = df.sum().sum()
    return sse

if __name__ == '__main__':
    main()