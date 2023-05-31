import argparse
import inspect
import os
from pathlib import Path
import shutil
import sys
import pandas as pd
from citylearn.data import DataSet
from citylearn.utilities import read_json, write_json

def get_combined_data(key):
    settings = get_settings()
    simulation_output_directory = settings['simulation_output_directory']
    data_list = []

    # environment
    for d in os.listdir(simulation_output_directory):
        if 'resstock' in d:
            simulation_id = d
            d = os.path.join(simulation_output_directory, d)
            data = pd.read_csv(os.path.join(d, f'{simulation_id}-{key}.csv'))
            data['neighborhood'] = data['simulation_id'].str.split('_', expand=True)[0]
            data['resstock_bldg_id'] = data['simulation_id'].str.split('-', expand=True)[5]
            data.loc[data['mode']=='test', 'episode'] = data['episode'].max() + 1

            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data['day_of_week'] = data['timestamp'].dt.weekday
                data['hour'] = data['timestamp'].dt.hour
                data['month'] = data['timestamp'].dt.month
            else:
                pass
            
            data_list.append(data)

        else:
            continue

    return pd.concat(data_list, ignore_index=True, sort=False)

def set_sb3_work_order(**kwargs):
    size_equipment(**kwargs)
    set_schema(**kwargs)
    settings = get_settings()
    src_directory = settings['src_directory']
    work_order_directory = settings['work_order_directory']
    schema_filepath = os.path.join(settings['schema_directory'], f'{kwargs["simulation_id"]}.json')
    schema = read_json(os.path.join(settings['schema_directory'], f'{kwargs["simulation_id"]}.json'))

    work_order = []

    for i, b in enumerate(schema['buildings']):
        simulation_id = f"{kwargs['simulation_id']}_{b}"
        command = f'python "{os.path.join(src_directory, "simulate.py")}" simulate "{schema_filepath}" {simulation_id} -b {b}'
        work_order.append(command)

    # write work order and tacc job
    work_order.append('')
    work_order = '\n'.join(work_order)
    work_order_filepath = os.path.join(work_order_directory, f'{kwargs["simulation_id"]}.sh')

    with open(work_order_filepath, 'w') as f:
        f.write(work_order)


def set_schema(**kwargs):
    # general settings
    settings = get_settings()
    timestamps = get_timestamps()
    
    district_directory = os.path.join(settings['neighborhood_directory'], kwargs['district_name'])
    schema = DataSet.get_schema('citylearn_challenge_2021')
    schema['simulation_id'] = kwargs['simulation_id']
    schema['root_directory'] = district_directory
    schema['central_agent'] = settings['central_agent']
    season = kwargs['season']
    schema['season'] = season
    schema['simulation_start_time_step'] = int(timestamps[timestamps['timestamp']==settings['season_timestamps'][season]['train_start_timestamp']].iloc[0].name)
    schema['simulation_end_time_step'] = int(timestamps[timestamps['timestamp']==settings['season_timestamps'][season]['train_end_timestamp']].iloc[0].name)
    schema['episodes'] = settings['episodes']
    time_steps = schema['simulation_end_time_step'] - schema['simulation_start_time_step']

    # set active observations
    for o in schema['observations']:
        if o in settings['active_observations']:
            schema['observations'][o]['active'] = True
        else:
            schema['observations'][o]['active'] = False

    # set all actions to active
    for a in schema['actions']:
        if a in settings['active_actions']:
            schema['actions'][a]['active'] = True
        else:
            schema['actions'][a]['active'] = False

    # set agent
    schema['agent'] = settings['agent']
    schema['agent']['attributes'] = {
        **schema['agent']['attributes'],
        'start_training_time_step': time_steps,
        'end_exploration_time_step': time_steps + 1,
        'deterministic_start_time_step': (time_steps + 1)*(schema['episodes'] - 1),
    }

    # set reward function
    schema['reward_function'] = settings['reward_function']

    # set buildings
    schema['buildings'] = {}
    sizing_directory = os.path.join(settings['sizing_directory'], kwargs['district_name'])
    building_files = sorted([f for f in os.listdir(district_directory) if f.startswith('resstock')])
    battery_sizing = pd.read_csv(os.path.join(sizing_directory, 'battery_sizing.csv'))
    dhw_storage_sizing = pd.read_csv(os.path.join(sizing_directory, 'dhw_storage_sizing.csv'))
    pv_sizing = pd.read_csv(os.path.join(sizing_directory, 'pv_sizing.csv'))

    for f in building_files:
        key = f.split('.')[0]

        # general building attributes
        schema['buildings'][key] = {
            'include': True,
            'energy_simulation': f,
            'weather': 'weather.csv',
            'carbon_intensity': None,
            'pricing': None,
        }

        # cooling device
        schema['buildings'][key]['cooling_device'] = {
            'type': 'citylearn.energy_model.HeatPump',
            'autosize': True,
            'attributes': {
                'nominal_power': None,
                'efficiency': 0.2,
                'target_cooling_temperature': 8.0,
            }
        }

        # heating device
        schema['buildings'][key]['heating_device'] = {
            'type': 'citylearn.energy_model.HeatPump',
            'autosize': True,
            'attributes': {
                'nominal_power': None,
                'efficiency': 0.2,
                'target_heating_temperature': 45.0,
            }
        }

        # dhw device
        schema['buildings'][key]['dhw_device'] = {
            'type': 'citylearn.energy_model.ElectricHeater',
            'autosize': True,
            'attributes': {
                'nominal_power': None,
                'efficiency': 0.9,
            }
        }

        # dhw storage
        schema['buildings'][key]['dhw_storage'] = {
            'type': 'citylearn.energy_model.StorageTank',
            'attributes': {
                'capacity': dhw_storage_sizing[dhw_storage_sizing['bldg_id']==key].iloc[0]['capacity'],
                'max_input_power': dhw_storage_sizing[dhw_storage_sizing['bldg_id']==key].iloc[0]['nominal_power'],
                'max_output_power': dhw_storage_sizing[dhw_storage_sizing['bldg_id']==key].iloc[0]['nominal_power'],
                'loss_coefficient': 0.003
            }
        }

        # electrical storage
        schema['buildings'][key]['electrical_storage'] = {
            'type': 'citylearn.energy_model.Battery',
            'autosize': False,
            'attributes': {
                'capacity': battery_sizing[battery_sizing['bldg_id']==key].iloc[0]['capacity'],
                'efficiency': 0.9,
                'capacity_loss_coefficient': 1e-05,
                'loss_coefficient': None,
                'nominal_power': battery_sizing[battery_sizing['bldg_id']==key].iloc[0]['nominal_power'],
                'power_efficiency_curve': [[0, 0.83], [0.3, 0.83], [0.7, 0.9], [0.8, 0.9], [1, 0.85]],
                'capacity_power_curve': [[0.0, 1.0], [0.8, 0.9], [1.0, 0.27]]
            }
        }

        schema['buildings'][key]['pv'] = {
            'type': 'citylearn.energy_model.PV',
            'autosize': False,
            'attributes': {
                'nominal_power': pv_sizing[pv_sizing['bldg_id']==key].iloc[0]['nominal_power']
            }
        }

    write_json(os.path.join(settings['schema_directory'], f'{kwargs["simulation_id"]}.json'), schema)

def size_equipment(**kwargs):
    settings = get_settings()
    destination_directory = os.path.join(settings['sizing_directory'], kwargs['district_name'])
    os.makedirs(destination_directory, exist_ok=True)
    dhw_storage_sizing = size_dhw_storage(**kwargs)
    battery_sizing = size_battery(**kwargs)
    pv_sizing = size_pv(**kwargs)

    dhw_storage_sizing.to_csv(os.path.join(destination_directory, 'dhw_storage_sizing.csv'), index=False)
    battery_sizing.to_csv(os.path.join(destination_directory, 'battery_sizing.csv'), index=False)
    pv_sizing.to_csv(os.path.join(destination_directory, 'pv_sizing.csv'), index=False)

def size_dhw_storage(**kwargs):
    # size capacity to meet daily peak
    # size power to meet average hourly peak
    settings = get_settings()
    season = kwargs['season']
    district_directory = os.path.join(settings['neighborhood_directory'], kwargs['district_name'])
    timestamps = get_timestamps()
    building_files = sorted([f for f in os.listdir(district_directory) if f.startswith('resstock')])
    sizing = []

    for b in building_files:
        building_id = b.split('.')[0]
        data = pd.read_csv(os.path.join(district_directory, b))
        sizing_data = pd.concat([timestamps, data], axis=1)
        sizing_data = sizing_data[
            (sizing_data['timestamp'] >= settings['season_timestamps'][season]['train_start_timestamp'])
            &(sizing_data['timestamp'] <= settings['season_timestamps'][season]['train_end_timestamp'])
        ]
        nominal_power = sizing_data['DHW Heating (kWh)'].mean()
        sizing_data = sizing_data.groupby('date')[['DHW Heating (kWh)']].sum()
        capacity = sizing_data['DHW Heating (kWh)'].max()
        sizing.append({'bldg_id': building_id, 'capacity': capacity, 'nominal_power': nominal_power})

    return pd.DataFrame(sizing)

def size_battery(**kwargs):
    # size capacity to meet max daily load
    # cap nominal power at 5 kW
    settings = get_settings()
    season = kwargs['season']
    district_directory = os.path.join(settings['neighborhood_directory'], kwargs['district_name'])
    timestamps = get_timestamps()
    building_files = sorted([f for f in os.listdir(district_directory) if f.startswith('resstock')])
    sizing = []

    for b in building_files:
        building_id = b.split('.')[0]
        data = pd.read_csv(os.path.join(district_directory, b))
        sizing_data = pd.concat([timestamps, data], axis=1)
        sizing_data = sizing_data[
            (sizing_data['timestamp'] >= settings['season_timestamps'][season]['train_start_timestamp'])
            &(sizing_data['timestamp'] <= settings['season_timestamps'][season]['train_end_timestamp'])
        ]

        # leave out cooling and heating load
        sizing_data = sizing_data.groupby('date')[[
            'Equipment Electric Power (kWh)',
            'DHW Heating (kWh)',
            # 'Cooling Load (kWh)',
            # 'Heating Load (kWh)'
        ]].sum()
        sizing_data = sizing_data.sum(axis=1)
        capacity = sizing_data.max()
        # sizing.append({'bldg_id': building_id, 'capacity': capacity, 'nominal_power': settings['battery_nominal_power']})
        sizing.append({'bldg_id': building_id, 'capacity': settings['default_battery_capacity'], 'nominal_power': settings['battery_nominal_power']})

    return pd.DataFrame(sizing)

def size_pv(**kwargs):
    # Size for ZNE
    settings = get_settings()
    district_directory = os.path.join(settings['neighborhood_directory'], kwargs['district_name'])
    building_files = sorted([f for f in os.listdir(district_directory) if f.startswith('resstock')])
    sam_data = pd.read_csv(os.path.join(district_directory, 'sam_pv.csv'))
    roof_data = pd.read_csv(os.path.join(district_directory, 'roof_area.csv'))
    sizing = []

    for b in building_files:
        building_id = b.split('.')[0]
        data = pd.read_csv(os.path.join(district_directory, b))
        data['Solar Generation (W/kW)'] = sam_data['System power generated | (kW)']*1000.0
        data.to_csv(os.path.join(district_directory, b), index=False)
        bldg_id = int(building_id.split('-')[-1])
        roof_area = roof_data[roof_data['bldg_id']==bldg_id]['roof_area'].iloc[0]
        total_load = data[[
            'Equipment Electric Power (kWh)',
            'DHW Heating (kWh)',
            'Cooling Load (kWh)',
            'Heating Load (kWh)'
        ]].sum(axis=1).sum()
        nominal_power_limit = roof_area/settings['sam_module_size']
        zne_nominal_power = total_load/sam_data['System power generated | (kW)'].sum()
        zne_nominal_power *= settings['pv_size_limit_proportion']
        nominal_power = min(zne_nominal_power, nominal_power_limit)
        # sizing.append({'bldg_id': building_id, 'nominal_power': nominal_power})
        sizing.append({'bldg_id': building_id, 'nominal_power': settings['default_pv_size']})
        
    return pd.DataFrame(sizing)

def get_timestamps():
    settings = get_settings()
    timestamps = pd.date_range(settings['start_timestamp'], settings['end_timestamp'], freq='H')
    timestamps = pd.DataFrame(timestamps, columns=['timestamp'])
    timestamps['hour'] = timestamps['timestamp'].dt.hour
    timestamps['date'] = timestamps['timestamp'].dt.normalize()

    return timestamps

def get_settings():
    src_directory = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts)
    settings_filepath = os.path.join(src_directory, 'settings.json')
    settings = read_json(settings_filepath)
    settings['root_directory'] = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts[0:-1])
    settings['data_directory'] = os.path.join(settings['root_directory'], 'data')
    settings['src_directory'] = os.path.join(settings['root_directory'], 'src')
    settings['work_order_directory'] = os.path.join(settings['root_directory'], 'workflow', 'work_order')
    settings['simulation_output_directory'] = os.path.join(settings['data_directory'], 'simulation_output')
    settings['figures_directory'] = os.path.join(settings['root_directory'], 'figures')
    settings['schema_directory'] = os.path.join(settings['data_directory'], 'schemas')
    settings['neighborhood_directory'] = os.path.join(settings['data_directory'], 'neighborhoods')
    settings['sizing_directory'] = os.path.join(settings['data_directory'], 'sizing')
    settings['3dem_directory'] = os.path.join(settings['root_directory'], '3dem')

    os.makedirs(settings['work_order_directory'], exist_ok=True)
    os.makedirs(settings['simulation_output_directory'], exist_ok=True)
    os.makedirs(settings['figures_directory'], exist_ok=True)
    os.makedirs(settings['schema_directory'], exist_ok=True)
    os.makedirs(settings['sizing_directory'], exist_ok=True)
    
    return settings

def main():
    parser = argparse.ArgumentParser(prog='bs2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('district_name', type=str, help='Name of district in data directory.')
    parser.add_argument('season', choices=['winter', 'summer'], type=str, help='Name of district in data directory.')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # size equipment
    subparser_size_equipment = subparsers.add_parser('size_equipment')
    subparser_size_equipment.set_defaults(func=size_equipment)

    # set schema
    subparser_set_schema = subparsers.add_parser('set_schema')
    subparser_set_schema.add_argument('simulation_id', type=str)
    subparser_set_schema.set_defaults(func=set_schema)

    # set sb3 work order
    subparser_set_schema = subparsers.add_parser('set_sb3_work_order')
    subparser_set_schema.add_argument('simulation_id', type=str)
    subparser_set_schema.set_defaults(func=set_sb3_work_order)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
