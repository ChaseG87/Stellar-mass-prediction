import numpy as np
import pandas as pd


def log_train_val_data(features):
    '''
    inputs:
        - features: list of strings of input features to be investigated
    returns:
        - df_train, df_val: DataFrames of the train and validation data
    '''
    # load train data
    raw_df_train = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq').reset_index()

    # load test data
    raw_df_val = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_test.pq').reset_index()


    # pre-process taking the log of features
    df_train = np.log10(raw_df_train[features])
    df_val = np.log10(raw_df_val[features])

    # pre-process taking the log of the mass (target)
    df_train['M_tot'] = np.log10(raw_df_train['M_tot'])
    df_val['M_tot'] = np.log10(raw_df_val['M_tot'])

    return df_train, df_val
    


