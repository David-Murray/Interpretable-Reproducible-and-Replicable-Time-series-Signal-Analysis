import pandas as pd
import numpy as np
import time
import os
import sys
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


def main():
    disk_path = '/media/david/Big Disk/' if sys.platform == 'linux' else 'D:/'
    aggregate_data = pd.DataFrame()
    appliance_data = pd.DataFrame()
    date_range = pd.date_range(start="2012-06-01", end="2013-01-31")
    for day in date_range:
        try:
            agg_data = pd.read_csv(f"{disk_path}Code/data/ECO/01_sm_csv/01/{day.strftime('%Y-%m-%d')}.csv",
                na_filter=False,
                header=None,
                prefix='a_',
                memory_map=True)
        except:
            print("Day Missing")
            agg_data = None
            continue
        try:
            app_data = pd.read_csv(f"{disk_path}Code/data/ECO/01_plugs_csv/01/05/{day.strftime('%Y-%m-%d')}.csv",
                na_filter=False,
                header=None,
                prefix='a_',
                memory_map=True)
        except:
            print("Day Missing")
            app_data = None
            continue
        aggregate_data = aggregate_data.append(agg_data, ignore_index=True)
        appliance_data = appliance_data.append(app_data, ignore_index=True)
    aggregate_data = aggregate_data.iloc[::8]
    aggregate_mean = aggregate_data['a_0'].mean()
    aggregate_std = aggregate_data['a_0'].std()
    print(f"M:{aggregate_mean} S:{aggregate_std}")
    aggregate_data['a_0'] = (aggregate_data['a_0'] - aggregate_mean) / aggregate_std

    appliance_data = appliance_data.iloc[::8]
    app_data_zerod = appliance_data['a_0'].copy()
    app_data_zerod[app_data_zerod<=20] = np.NaN
    app_mean = app_data_zerod.mean(skipna=True)
    app_std = app_data_zerod.std(skipna=True)
    print(f"M:{app_mean} S:{app_std}")
    appliance_data['a_0']= (appliance_data['a_0'] - app_mean) / app_std
    combined_data = pd.DataFrame(data={"agg":aggregate_data['a_0'], "app":appliance_data['a_0']})
    df_train, df_val = train_test_split(combined_data, test_size=0.4, random_state=42, shuffle=False)
    df_val, df_test = train_test_split(df_val, test_size=0.5, random_state=42, shuffle=False)
    print(f"train: {df_train.shape}")
    df_train.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_training_.csv")), header=False, index=False)
    print(f"df_val: {df_val.shape}")
    df_val.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_validation_H1.csv")), header=True, index=False)
    # print(f"df_test: {df_test.shape}")
    # df_test.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_test_H1.csv")), header=True, index=False)
    combined_data.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), "./washingmachine/washingmachine_test_H1.csv")), header=True, index=False)
    plt.plot(combined_data)
    plt.show()

if __name__ == '__main__':
    main()
