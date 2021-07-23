import pandas as pd
import numpy as np
import time
import os
import re
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

params_appliance = {
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
        'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
        'test_house': 8,
        'validation_house': 18,
        'test_on_train_house': 5,
    }
}


def load(path, building, appliance, channel):

    # load csv
    file_name = path + 'CLEAN_House' + str(building) + '.csv'
    single_csv = pd.read_csv(file_name,
                             header=0,
                             names=['aggregate', appliance],
                             usecols=[2, channel+2],
                             na_filter=False,
                             parse_dates=True,
                             infer_datetime_format=True,
                             memory_map=True
                             )

    return single_csv


def main():
    # Open ampds2.h5
    aggregate_data = pd.read_csv("D:/Code/data/ampds2/Electricity_WHE.csv",
        na_filter=False,
        parse_dates=True,
        index_col=0,
        infer_datetime_format=True,
        memory_map=True)
    print(aggregate_data.head())
    aggregate_mean = aggregate_data['P'].mean()
    aggregate_std = aggregate_data['P'].std()
    aggregate_data['P'] = (aggregate_data['P'] - aggregate_mean) / aggregate_std
    appliance_data = pd.read_csv("D:/Code/data/ampds2/Electricity_CWE.csv",
        na_filter=False,
        parse_dates=True,
        index_col=0,
        infer_datetime_format=True,
        memory_map=True)
    print(appliance_data.head())
    app_data_zerod = appliance_data['P'].copy()
    app_data_zerod[app_data_zerod==0] = np.NaN
    app_mean = app_data_zerod.mean(skipna=True)
    app_std = app_data_zerod.std(skipna=True)
    appliance_data['P']= (appliance_data['P'] - app_mean) / app_std
    combined_data = pd.DataFrame(data={"agg":aggregate_data['P'], "app":appliance_data['P']})
    df_train, df_val = train_test_split(combined_data, test_size=0.4, random_state=42)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42)
    # combined_data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_training_.csv")), header=False, index=False)
    print(f"train: {df_train.shape}")
    df_train.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_training_.csv")), header=False, index=False)
    print(f"df_val: {df_val.shape}")
    df_val.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_validation_H1.csv")), header=True, index=False)
    print(f"df_test: {df_test.shape}")
    df_test.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_test_H1.csv")), header=True, index=False)
    
if __name__ == '__main__':
    main()
