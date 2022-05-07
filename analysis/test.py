import os
from pathlib import Path
import sys
sys.path.insert(0,'../')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
from doe_xstock.database import SQLiteDatabase
from doe_xstock.doe_xstock import DOEXStockDatabase
from doe_xstock.exploration import MetadataClustering

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
DATABASE_FILEPATH = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/database.db')
FIGURES_DIRECTORY = Path('/Users/kingsleyenweye/Desktop/INTELLIGENT_ENVIRONMENT_LAB/doe_xstock/doe_xstock/figures')
DATABASE = SQLiteDatabase(DATABASE_FILEPATH)
data = DATABASE.query_table("""
SELECT
    n.name,
    c.n_clusters
FROM metadata_clustering_name n
LEFT JOIN optimal_metadata_clustering o ON o.name_id = n.id
LEFT JOIN metadata_clustering c ON c.id = o.clustering_id
""").to_records(index=False)

for name, n_clusters in data:
    MetadataClustering.plot_scores(name,DATABASE_FILEPATH,figure_filepath=os.path.join(FIGURES_DIRECTORY,f'{name}_metadata_clustering_scores.png'))
    MetadataClustering.plot_sample_count(name,n_clusters,DATABASE_FILEPATH,figure_filepath=os.path.join(FIGURES_DIRECTORY,f'{name}_metadata_clustering_sample_count.png'))
    MetadataClustering.plot_ground_truth(name,n_clusters,DATABASE_FILEPATH,figure_filepath=os.path.join(FIGURES_DIRECTORY,f'{name}_metadata_clustering_ground_truth.png'))