import argparse
import concurrent.futures
from datetime import datetime
import inspect
import logging
import os
from pathlib import Path
import shutil
import subprocess
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import sys
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from citylearn.citylearn import CityLearnEnv, NormalizedObservationWrapper, StableBaselines3Wrapper
from citylearn.utilities import read_json
from preprocess import get_settings, get_timestamps

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def run_work_order(work_order_filepath, virtual_environment_path=None, windows_system=None):
    settings = get_settings()
    work_order_filepath = Path(work_order_filepath)

    if virtual_environment_path is not None:    
        if windows_system:
            virtual_environment_command = f'"{os.path.join(virtual_environment_path, "Scripts", "Activate.ps1")}"'
        else:
            virtual_environment_command = f'source "{os.path.join(virtual_environment_path, "bin", "activate")}"'
    else:
        virtual_environment_command = 'echo "No virtual environment"'

    with open(work_order_filepath,mode='r') as f:
        args = f.read()
    
    args = args.strip('\n').split('\n')
    args = [f'{virtual_environment_command} && {a}' for a in args]
    settings = get_settings()
    max_workers = settings['max_workers'] if settings.get('max_workers',None) is not None else cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a, 'shell':True}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            except Exception as e:
                print(e)

def simulate(**kwargs):
    settings = get_settings()
    timestamps = get_timestamps()
    schema = kwargs['schema']
    schema = read_json(os.path.join(settings['schema_directory'], schema))
    simulation_id = kwargs.get('simulation_id', schema['simulation_id'])

    # set buildings
    if kwargs.get('building', None) is not None:
        for b in schema['buildings']:
            if b == kwargs['building']:
                schema['buildings'][b]['include'] = True
            else:
                schema['buildings'][b]['include'] = False

    else:
        pass
    
    # set simulation output path
    simulation_output_path = os.path.join(settings['simulation_output_directory'], simulation_id)

    if os.path.isdir(simulation_output_path):
        shutil.rmtree(simulation_output_path)
    else:
        pass

    os.makedirs(simulation_output_path, exist_ok=True)
    set_logger(simulation_id, simulation_output_path)

    # set env and agents
    env = CityLearnEnv(schema, central_agent=True)
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    # learn
    model = SAC('MlpPolicy', env, verbose=2, learning_starts=env.unwrapped.time_steps, seed=0)
    episodes = env.schema['episodes']
    callback = SaveDataCallback(schema, env, simulation_id, simulation_output_path, timestamps, episodes)
    model.learn(total_timesteps=(env.unwrapped.time_steps)*episodes, log_interval=1000, callback=callback)

    # evaluate
    season = schema['season']
    schema['simulation_start_time_step'] = int(timestamps[
        timestamps['timestamp']==settings['season_timestamps'][season]['test_start_timestamp']
    ].iloc[0].name)
    schema['simulation_end_time_step'] = int(timestamps[
        timestamps['timestamp']==settings['season_timestamps'][season]['test_end_timestamp']
    ].iloc[0].name)
    eval_env = CityLearnEnv(schema, central_agent=True)
    eval_env = NormalizedObservationWrapper(eval_env)
    eval_env = StableBaselines3Wrapper(eval_env)
    observations = eval_env.reset()
    start_timestamp = datetime.utcnow()

    while not eval_env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = eval_env.step(actions)

    save_data(schema, eval_env, simulation_id, simulation_output_path, timestamps, start_timestamp, 0, 'test')
    

class SaveDataCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, schema, env, simulation_id, simulation_output_path, timestamps, episodes, verbose=0):
        super(SaveDataCallback, self).__init__(verbose)
        self.schema = schema
        self.env = env
        self.simulation_id = simulation_id
        self.simulation_output_path = simulation_output_path
        self.timestamps = timestamps
        self.episodes = episodes
        self.episode = 0
        self.start_timestamp = datetime.utcnow()
        self.mode = 'train'

    def _on_step(self) -> bool:
        # print to log
        LOGGER.debug(
            f'Time step: {self.env.time_step}/{self.env.time_steps - 1},'\
                f' Episode: {self.episode}/{self.episodes - 1},'\
        )

        # save timer data
        if self.env.time_step == self.env.time_steps - 2:
            save_data(
                self.schema, 
                self.env, 
                self.simulation_id, 
                self.simulation_output_path, 
                self.timestamps, 
                self.start_timestamp, 
                self.episode, 
                self.mode
            )
            self.episode += 1
            self.start_timestamp = datetime.utcnow()

        else:
            pass

        return True

def save_data(schema, env, simulation_id, simulation_output_path, timestamps, start_timestamp, episode, mode):
    end_timestamp = datetime.utcnow()
    timer_data = pd.DataFrame([{
        'simulation_id': simulation_id,
        'mode': mode,
        'episode': episode,
        'start_timestamp': start_timestamp, 
        'end_timestamp': end_timestamp
    }])
    timer_filepath = os.path.join(simulation_output_path, f'{simulation_id}-timer.csv')

    if os.path.isfile(timer_filepath):
        existing_data = pd.read_csv(timer_filepath)
        timer_data = pd.concat([existing_data, timer_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    timer_data.to_csv(timer_filepath, index=False)
    del timer_data

    # save environment summary data
    data_list = []

    for i, b in enumerate(env.buildings):
        env_data = pd.DataFrame({
            'solar_generation': b.solar_generation,
            'non_shiftable_load_demand': b.non_shiftable_load_demand,
            'dhw_demand': b.dhw_demand,
            'heating_demand': b.heating_demand,
            'cooling_demand': b.cooling_demand,
            'energy_from_electrical_storage': b.energy_from_electrical_storage,
            'energy_from_dhw_storage': b.energy_from_dhw_storage,
            'energy_from_dhw_device': b.energy_from_dhw_device,
            'energy_from_heating_device': b.energy_from_heating_device,
            'energy_from_cooling_device': b.energy_from_cooling_device,
            'energy_to_electrical_storage': b.energy_to_electrical_storage,
            'energy_from_dhw_device_to_dhw_storage': b.energy_from_dhw_device_to_dhw_storage,
            'electrical_storage_electricity_consumption': b.electrical_storage_electricity_consumption,
            'dhw_storage_electricity_consumption': b.dhw_storage_electricity_consumption,
            'dhw_electricity_consumption': b.dhw_electricity_consumption,
            'heating_electricity_consumption': b.heating_electricity_consumption,
            'cooling_electricity_consumption': b.cooling_electricity_consumption,
            'net_electricity_consumption': b.net_electricity_consumption,
            'net_electricity_consumption_without_storage': b.net_electricity_consumption_without_storage,
            'net_electricity_consumption_without_storage_and_pv': b.net_electricity_consumption_without_storage_and_pv,
            'electrical_storage_soc': np.array(b.electrical_storage.soc)/b.electrical_storage.capacity_history[0],
            'dhw_storage_soc': np.array(b.dhw_storage.soc)/b.dhw_storage.capacity,
        })
        env_data['timestamp'] = timestamps['timestamp'].iloc[
            schema['simulation_start_time_step']:
            schema['simulation_start_time_step'] + env.time_step + 1
        ].tolist()
        env_data['time_step'] = env_data.index
        env_data['mode'] = mode
        env_data['episode'] = episode
        env_data['building_id'] = i
        env_data['building_name'] = b.name
        env_data['simulation_id'] = simulation_id
        data_list.append(env_data)
    
    env_filepath = os.path.join(simulation_output_path, f'{simulation_id}-environment.csv')

    if os.path.isfile(env_filepath):
        existing_data = pd.read_csv(env_filepath)
        data_list = [existing_data] + data_list
        del existing_data
    else:
        pass
    
    env_data = pd.concat(data_list, ignore_index=True, sort=False)
    env_data.to_csv(env_filepath, index=False)
    del data_list
    del env_data
    
    # save reward data
    reward_data = pd.DataFrame(env.rewards, columns=['reward'])
    reward_data['time_step'] = reward_data.index
    reward_data['building_name'] = None
    reward_data['mode'] = mode
    reward_data['episode'] = episode
    reward_data['simulation_id'] = simulation_id
    reward_filepath = os.path.join(simulation_output_path, f'{simulation_id}-reward.csv')

    if os.path.isfile(reward_filepath):
        existing_data = pd.read_csv(reward_filepath)
        reward_data = pd.concat([existing_data, reward_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    reward_data.to_csv(reward_filepath, index=False)
    del reward_data

    # save KPIs
    ## building level
    kpi_data = env.evaluate()
    kpi_data['mode'] = mode
    kpi_data['episode'] = episode
    kpi_data['simulation_id'] = simulation_id
    kpi_filepath = os.path.join(simulation_output_path, f'{simulation_id}-kpi.csv')

    if os.path.isfile(kpi_filepath):
        existing_data = pd.read_csv(kpi_filepath)
        kpi_data = pd.concat([existing_data, kpi_data], ignore_index=True, sort=False)
        del existing_data
    else:
        pass

    kpi_data.to_csv(kpi_filepath, index=False)
    del kpi_data

def set_logger(simulation_id, simulation_output_path):
    os.makedirs(simulation_output_path, exist_ok=True)
    log_filepath = os.path.join(simulation_output_path, f'{simulation_id}.log')

    # set logger
    handler = logging.FileHandler(log_filepath, mode='w')
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(prog='bs2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # simulate
    subparser_simulate = subparsers.add_parser('simulate')
    subparser_simulate.add_argument('schema', type=str)
    subparser_simulate.add_argument('simulation_id', type=str)
    subparser_simulate.add_argument('-b', '--building', dest='building', type=str)
    subparser_simulate.set_defaults(func=simulate)

    # run work order
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.add_argument('work_order_filepath', type=Path)
    subparser_run_work_order.set_defaults(func=run_work_order)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
