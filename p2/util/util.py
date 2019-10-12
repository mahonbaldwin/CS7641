import csv
import os
from sklearn.model_selection import train_test_split

import pandas as pd


def default(val, default):
    if val is None:
        return default
    else:
        return val


def myget(dict, key, default=None):
    if dict is None:
        dict = {}
    return dict.get(key, default)


def classify_data(data, column):
    all_values = data[column].unique()
    new_col_name = column+'-prime'
    data[new_col_name] = data[column]
    category = 1
    for v in all_values:
        data.loc[data[column] == v, new_col_name] = category
        category = category + 1
    print('values')
    print(all_values)
    return data


def split_feature(data, column):
    "Will split a single feature into multiple features with boolean values."
    all_values = data[column].unique()
    for v in all_values:
        data.loc[data[column] == v, column+'_'+str(v)] = 1
        data.loc[data[column] != v, column+'_'+str(v)] = 0
    print('values')
    print(all_values)
    return data


def get_data(filename):
    return pd.read_csv(filename)


# def write_csv(dir_name, filename, header, create_if_needed=False):
#     os.makedirs(dir_name, exist_ok=True)
#     with open(filename, 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(fields)

def prep_data(data, y_key, options=None):
    options = default(options, {})
    exclude_keys = myget(options, 'exclude-keys', [])
    test_size=1 - myget(options, 'perc_sample', .67)
    exclude_keys.append(y_key)

    x = data.loc[:, data.columns.difference(exclude_keys)]
    y = data.loc[:, y_key]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test