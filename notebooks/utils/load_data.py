import os
from datetime import datetime
import pathlib
import numpy as np
import sys, traceback
import pandas as pd


from utils.load_config import (
    get_df_historical_data,
    get_list_historical_data
    )


ROOT_DIR = "./regelum_data/outputs/"

def correct_column_name(df):
    replacements = {
        "x_rob"   :     "x [m]", 
        "y_rob"   :    "y [m]",
        "vartheta":       "angle [rad]",
        "v"       : "velocity [m/s]",
        "omega"   :    "angular velocity [rad/s]",
    }

    new_columns = []
    if "x [m]" not in df.columns.values:
        for col in df.columns.values:
            new_columns.append(col if col not in replacements else replacements[col])

        df.columns = new_columns
    
    return df


def cal_obj_df(row, objective_function):
    observation = np.expand_dims(np.array(row.loc[["x [m]", "y [m]", "angle [rad]"]].values, dtype=np.float32), axis=0)
    action = np.expand_dims(np.array(row.loc[["velocity [m/s]", "angular velocity [rad/s]"]].values, dtype=np.float32), axis=0)

    try:
        return objective_function(observation, action)
    except Exception as err:
        print("Error:", err)
        traceback.print_exc(file=sys.stdout)
        
        raise err


def get_df_from_datetime_range(start_datetime_str, 
                               end_datetime_str, 
                               objective_function,
                               date_format='%Y-%m-%d %H-%M-%S', 
                               decay_rate=1,
                               ):
    start_date_time = datetime.strptime(start_datetime_str, date_format)
    end_date_time = datetime.strptime(end_datetime_str, date_format)
    

    backup_file_name = "_".join([c.replace(" ", "_") for c in ["data", start_datetime_str, end_datetime_str]]) + ".pkl"
    bk_path = os.path.join("./backup-data", backup_file_name)

    if os.path.exists(bk_path):
        return pd.read_pickle(bk_path)

    date_folder = os.listdir(ROOT_DIR)

    valid_paths = []
    for d in date_folder:
        for t in os.listdir(os.path.join(ROOT_DIR, d)):
            tmp_datetime = datetime.strptime(f"{d} {t}", date_format)
            if tmp_datetime < start_date_time or end_date_time < tmp_datetime:
                continue

            valid_paths.append(str(pathlib.Path(os.path.join(ROOT_DIR, d, t)).absolute()))

    path_hierachy = {}
    for p in valid_paths:
        path_hierachy[p] = get_list_historical_data(p)

    total_dfs = []
    for exp_path in path_hierachy:
        exp_dfs = []
        for iteration_path in path_hierachy[exp_path]:
            exp_dfs.append(get_df_historical_data(absolute_path=iteration_path))
            exp_dfs[-1]["absolute_path"] = iteration_path
            exp_dfs[-1] = correct_column_name(exp_dfs[-1])
            exp_dfs[-1]["objective_value"] = exp_dfs[-1].apply(lambda x: cal_obj_df(x, objective_function), axis=1)
            # exp_dfs[-1]["accumulative_objective"] = exp_dfs[-1]["objective_value"].apply(lambda x: x*0.1).cumsum()
            exp_dfs[-1]["accumulative_objective"] = exp_dfs[-1].apply(lambda x: x["objective_value"]*0.1*decay_rate**x["time"], axis=1).cumsum()
        
        if len(exp_dfs) == 0:
            continue
        
        exp_df = pd.concat(exp_dfs)
        exp_df.sort_values(by=["iteration_id", "time"], inplace=True)
        exp_df["experiment_path"] = exp_path
        
        total_dfs.append(exp_df)

    total_df = pd.concat(total_dfs)

    # Post process
    total_df = total_df[total_df.iteration_id <= 100]

    os.makedirs("./backup-data", exist_ok=True)
    total_df.to_pickle(bk_path)
    
    return total_df