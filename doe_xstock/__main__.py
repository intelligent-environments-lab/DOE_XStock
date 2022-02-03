import argparse
import inspect
import logging
import logging.config
import os
import sys
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
    parser.add_argument('-o','--overwrite',default=False,action='store_true',dest='overwrite',help='Will overwrite database if it exists.')
    parser.add_argument('-a','--apply_changes',default=False,action='store_true',dest='apply_changes',help='Will apply new changes to database schema.')
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # dataset
    subparser_dataset = subparsers.add_parser('dataset',description='Database dataset operations.')
    subparser_dataset.add_argument('dataset_type',type=str,choices=['resstock'],help='Residential or commercial building stock dataset.')
    subparser_dataset.add_argument('weather_data',type=str,choices=['tmy3'],help='Weather file used in dataset simulation.')
    subparser_dataset.add_argument('year_of_publication',type=int,choices=[2021],help='Year dataset was published.')
    subparser_dataset.add_argument('release',type=int,choices=[1],help='Dataset release version.')
    dataset_subparsers = subparser_dataset.add_subparsers(title='subcommands',required=True,dest='subcommands')

    # dataset -> insert
    subparser_insert = dataset_subparsers.add_parser('insert',description='Insert dataset into database.')
    subparser_insert.add_argument('-f','--filters_filepath',type=str,dest='filters_filepath',help='Insertion filters filepath where keys are columns in metadata table and values are values found in the columns.')
    subparser_insert.set_defaults(func=DOEXStock.insert)
    
    # dataset -> simulate
    subparser_simulate = dataset_subparsers.add_parser('simulate',description='Run building EnergyPlus simulation.')
    subparser_simulate.add_argument('idd_filepath',type=str,help='Energyplus IDD filepath.')
    subparser_simulate.add_argument('bldg_id',type=int,help='bldg_id field value in metadata table.')
    subparser_simulate.add_argument('-u','--upgrade',type=int,default=0,help='upgrade field value in metadata table.')
    subparser_simulate.add_argument('-o','--root_output_directory',type=str,dest='root_output_directory',help='Root directory to store simulation output directory to.')
    subparser_simulate.set_defaults(func=DOEXStock.simulate)

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