import os
import yaml
import pandas as pd
from glob import glob


def load_exp_config(exp_dpath):
    os.listdir(exp_dpath)
    with open(os.path.join(exp_dpath, '.rehydra', 'config.yaml')) as f:
        exp_config = yaml.safe_load(f)
        
    return exp_config


def get_value_from_fields(configs, fields, index=0):
    if index == len(fields) - 1:
        return configs[fields[index]]
    else:
        return get_value_from_fields(configs[fields[index]], fields, index + 1)

def get_df_historical_data(exp_path=None, chosen_name=None, absolute_path=None):
    if absolute_path is not None and absolute_path.endswith(".h5"):
        file_path = absolute_path
    elif exp_path is not None and chosen_name is not None:
        file_path = os.path.join(exp_path, ".callbacks/HistoricalDataCallback", f"{chosen_name}.h5")
    else:
        raise FileNotFoundError()

    return pd.read_hdf(file_path, key="data") 

def get_list_historical_data(exp_path):
    return glob(exp_path + "/*/.callbacks/HistoricalDataCallback/*.h5")
