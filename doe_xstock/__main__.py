import argparse
import inspect
import logging
import logging.config
import os
import sys
import simplejson as json
from doe_xstock.doe_xstock import DOEXStock
from doe_xstock.utilities import read_json

logging_config = read_json(os.path.join(os.path.dirname(__file__),'misc/logging_config.json'))
logging.config.dictConfig(logging_config)
LOGGER = logging.getLogger('main')

def main():
    parser = argparse.ArgumentParser(prog='doe_xstock',description='Manage DOE\'s End Use Load Profiles for the U.S. Building Stock.')
    parser.add_argument('--version', action='version', version='0.0.1')
    parser.add_argument("-v", "--verbosity",action="count",default=0,help='increase output verbosity')
    parser.add_argument('-d','--filepath',type=str,default=DOEXStock.DEFAULT_DATABASE_FILEPATH,dest='filepath',help='Database filepath.')
    parser.add_argument('-g','--figure_directory',type=str,default='figures',dest='figure_directory',help='Figure directory.')
    parser.add_argument('-t','--data_directory',type=str,default='data',dest='data_directory',help='Data directory.')
    parser.add_argument('-u','--energyplus_output_directory',type=str,default='energyplus_output',dest='energyplus_output_directory',help='Directory to store energyplus simulations output to.')
    parser.add_argument('-l','--lstm_train_data_directory',type=str,default='lstm_train_data',dest='lstm_train_data_directory',help='Directory to store LSTM train data to.')
    parser.add_argument('-o','--overwrite',default=False,action='store_true',dest='overwrite',help='Will overwrite database if it exists.')
    parser.add_argument('-a','--apply_changes',default=False,action='store_true',dest='apply_changes',help='Will apply new changes to database schema.')
    parser.add_argument('-i','--idd_filepath',default='/Applications/EnergyPlus-9-6-0/PreProcess/IDFVersionUpdater/V9-6-0-Energy+.idd',type=str,dest='idd_filepath',help='Energyplus IDD filepath.')
    parser.add_argument('-w','--max_workers',default=1,type=int,dest='max_workers',help='Maximum number of threads/processes to utilize.')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # dataset
    subparser_dataset = subparsers.add_parser('dataset',description='Database dataset operations.')
    subparser_dataset.add_argument('dataset_type',type=str,choices=['resstock'],help='Residential or commercial building stock dataset.')
    subparser_dataset.add_argument('weather_data',type=str,choices=['tmy3','amy2018'],help='Weather file used in dataset simulation.')
    subparser_dataset.add_argument('year_of_publication',type=int,choices=[2021],help='Year dataset was published.')
    subparser_dataset.add_argument('release',type=int,choices=[1],help='Dataset release version.')
    subparser_dataset.add_argument('-f','--filters',type=json.loads,dest='filters',help='Metadata table column filters.')
    dataset_subparsers = subparser_dataset.add_subparsers(title='subcommands',required=True,dest='subcommands')

    # dataset -> insert
    subparser_insert = dataset_subparsers.add_parser('insert',description='Insert dataset into database.')
    subparser_insert.set_defaults(func=DOEXStock.insert)
    
    # dataset -> simulate
    subparser_simulate = dataset_subparsers.add_parser('simulate',description='Run building EnergyPlus simulation.')
    subparser_simulate.add_argument('bldg_id',type=int,help='bldg_id field value in metadata table.')
    subparser_simulate.add_argument('-u','--upgrade',type=int,default=0,help='upgrade field value in metadata table.')
    subparser_simulate.add_argument('-o','--root_output_directory',type=str,dest='root_output_directory',help='Root directory to store simulation output directory to.')
    subparser_simulate.set_defaults(func=DOEXStock.simulate)

    # dataset -> metadata_cluster
    subparser_metadata_clustering = dataset_subparsers.add_parser('metadata_clustering',description='Metadata KMeans clustering.')
    subparser_metadata_clustering.add_argument('name',type=str,help='Unique name.')
    subparser_metadata_clustering.add_argument('maximum_n_clusters',type=int,help='Maximum number of clusters.')
    subparser_metadata_clustering.add_argument('-m','--minimum_n_clusters',type=int,dest='minimum_n_clusters',default=2,help='Minimum number of clusters.')
    subparser_metadata_clustering.add_argument('-c','--sample_count',type=int,dest='sample_count',help='Number of buildings to sample for E+ modeling.')
    subparser_metadata_clustering.add_argument('-s','--seed',type=json.loads,dest='seed',help='Random state seed.')
    subparser_metadata_clustering.set_defaults(func=DOEXStock.metadata_clustering)

    # dataset -> get_lstm_train_data
    subparser_set_lstm_train_data = dataset_subparsers.add_parser('set_lstm_train_data',description='Run EnergyPlus simulations with partial load to generate train data for building LSTM model.')
    subparser_set_lstm_train_data.add_argument('bldg_id',type=int,help='bldg_id field value in metadata table.')
    subparser_set_lstm_train_data.add_argument('-s','--seed',default=0,type=int,dest='seed',help='Pseudorandom generator seed.')
    subparser_set_lstm_train_data.add_argument('-r','--iterations',default=4,type=int,dest='iterations',help='Number of partial load simulation iterations.')
    subparser_set_lstm_train_data.add_argument('-m','--simulation_id',type=str,dest='simulation_id',help='Unique simulation ID.')
    subparser_set_lstm_train_data.set_defaults(func=DOEXStock.set_lstm_train_data)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {
        key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }

    try:
        args.func(**kwargs)
    except Exception as e:
        LOGGER.exception(e)
        sys.exit(1)

if __name__ == '__main__':
    sys.exit(main())