# Imports
import numpy as np
import pandas as pd
import os
import glob

from collections import Counter


# Functions
def load_files():
    file_loc = os.path.join(MAIN, DATA)
    files = glob.glob1(file_loc, '*.csv')
    file_holder = {}

    print('Loading {} files..'.format(len(files)))

    for f in files:
        file_path = os.path.join(file_loc, f)
        ftemp = pd.read_csv(file_path, low_memory=False)
        ftemp_name = f[:-4]
        file_holder[ftemp_name] = ftemp

    return file_holder


def process_data():

    all_data = load_files()

    train = all_data['train_2016_v2']
    prop = all_data['properties_2016']
    subm = all_data['sample_submission 4']

    train_id = train['parcelid']
    test_id = subm['ParcelId']

    print('Finding features for train set..')

    x_train_raw = train_id.merge(properties_data, how="inner", left_on="parcelid",
                                 right_on="parcelid", sort=False, suffixes=('_x', '_y'),
                                 copy=True, indicator=False)

    x_train_raw.drop(['parcelid', '_merge'], inplace=True)

    if len(train_id) == len(x_train_raw):
        print('Found features for all train IDs..')
    else:
        missing = len(train_id) - len(x_train_raw)

        print('Missing features for {} IDs..'.format(missing))

    x_test_raw = test_id.merge(properties_data, how="inner", left_on="ParcelId",
                                 right_on="parcelid", sort=False, suffixes=('_x', '_y'),
                                 copy=True, indicator=False)

    x_test_raw.drop(['ParcelId', 'parcel_id', '_merge'], inplace=True)

    if len(test_id) == len(x_test_raw):
        print('Found features for all train IDs..')
    else:
        missing = len(test_id) - len(x_test_raw)

        print('Missing features for {} IDs..'.format(missing))

    print('Converting all features to arrays..')

    x_train = np.array(x_train_raw)
    x_test = np.array(x_test_raw)
    y_train = np.array(train['logerror'], dtype=np.float16)

    id_train = np.array(train['parcelid'], dtype=np.str)
    id_test = np.array(train['Parcelid'], dtype=np.str)

    return x_train, y_train, id_train, x_test, id_test

# START TRYOUT SECTION #
MAIN = os.getcwd()
DATA = 'data'
RESULTS = 'results'
SUBS = 'submissions'

all_data = load_files()

xtr, ytr, idtr, xte, idte = process_data()

train_data = all_data['train_2016_v2']
properties_data = all_data['properties_2016']
submission_format = all_data['sample_submission 4']

test_ids = pd.DataFrame(submission_format['ParcelId'])

x_test = test_ids.merge(
    properties_data,
    how="inner",
    left_on="ParcelId",
    right_on="parcelid",
    sort=True,
    suffixes=('_x', '_y'),
    copy=True,
    indicator=False)

x_test = properties_data[test_ids]
transaction_dates = train_data['transactiondate']
unique_dates = len(transaction_dates.unique())

# END TRYOUT SECTION #
if __name__ == "main":

    MAIN = os.getcwd()
    DATA = 'data'
    RESULTS = 'results'
    SUBS = 'submissions'

    all_data = load_files()

    train_data = all_data['train_2016_v2']
    properties_data = all_data['properties_2016']

