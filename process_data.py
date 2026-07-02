import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load pre-stellar train data
raw_df_train = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq').reset_index()

# load pre-stellar validation data
raw_df_val = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_test.pq').reset_index()

# load full train data
master_df_train = pd.read_parquet('star_data/accreted_gas_properties.pq').reset_index()

# load full validation data
master_df_val = pd.read_parquet('star_data/accreted_gas_properties_test.pq').reset_index()

# load snap table
snap_table_df = pd.read_parquet('star_data/first_snap_table.pq').reset_index()


def merge_snap_table(df, snap_table=snap_table_df):
    '''
    Merges the snap table with the given DataFrame based on the 'sink_id' column.
    
    inputs:
    - df: pandas DataFrame containing the data to be merged.
    - snap_table: pandas DataFrame containing the snap table data.
    
    returns:
    - merged_df: pandas DataFrame resulting from merging df and snap_table on the 'sink_id' column.
    '''

    merged_df = pd.merge(left=df, right=snap_table[['pid', 'mf']], left_on='sink_id', right_on='pid', how='outer')
    return merged_df


def normalize_data(features, df_train, df_val=None):
    '''
    Normalizes input features of data with respect to the training dataset.
    
    inputs:
        -features: list of strings of features to be normalized
        -df_train: DataFrame of training data
        -df_val: DataFrame of validation data (optional)
    '''

    # create scaler and scale training data
    scaler = StandardScaler()
    training_data = df_train[features]
    scaled_training_data = scaler.fit_transform(training_data)
    df_train[features] = scaled_training_data

    # scale validation of called
    if type(df_val) != type(None):

        val_data = df_val[features]
        scaled_val_data = scaler.transform(val_data)
        df_val[features] = scaled_val_data

        return df_train, df_val
    
    else:
        return df_train


def add_standard_error(df):
    '''
    Takes DataFrames and adds the standard error using the standard deviation and number of included particles
    inputs:
        -dfs: DataFrames to be updated
    returns:
        df: DataFrames with standard error statistics
    '''

    sample_dem = np.sqrt(df['N_inc'])

    df['T_std_er'] = df['T_std']/sample_dem
    df['B_std_er'] = df['B_std']/sample_dem
    df['Ne_std_er'] = df['Ne_std']/sample_dem
    df['rho_std_er'] = df['rho_std']/sample_dem

    return df


def log_data(features, train=raw_df_train, val=raw_df_val, train_val=False, normalize=False):
    '''
    inputs:
        - train: DataFrame of the raw training data
        - val: DataFrame of the raw validation data
        - features: list of strings of input features to be investigated
        - train_val: bool that determines if a validation set will also be included
        - normalize: bool that determines if the input features are normalized
    returns:
        - df_train, df_val: DataFrames of the train and validation data
    '''

    # order data by sink index
    train = merge_snap_table(train, snap_table_df)
    train = train.set_index('sink_id').sort_index()

    # add standard error
    train = add_standard_error(train)

    # pre-process taking the log of features
    df_train = np.log10(train[features])

    # pre-process taking the log of the mass (target)
    df_train['mf'] = np.log10(train['mf'])

    if train_val:
        # apply the same process for validation data if train_val is called
        val = merge_snap_table(val, snap_table_df)
        val = add_standard_error(val)
        val = val.set_index('sink_id').sort_index()
        df_val = np.log10(val[features])
        df_val['mf'] = np.log10(val['mf'])

        if normalize:
            return normalize_data(features=features, df_train=df_train, df_val=df_val)
        else:
            return df_train, df_val
    
    if normalize:
        return normalize(features=features, df_train=df_train)
    else:
        return df_train


def n_snaps_general(master_df, pre_df, n_snaps):
    '''
    Generates data of a specific number of snaps before a star is formed in the training dataset
    inputs:
        - n_snaps: int number of snaps before the pre-stellar snap
    returns:
        - df: DataFrame of stellar gas data n_snaps before the pre-stellar snap
    '''

    if n_snaps > 47:
        raise ValueError('Earliest star forms at 47 snaps, n_snaps cannot exceed this value.')

    # modify pre-stellar snap of each star sink_id
    lookup_df = pre_df[['sink_id', 'snap']].copy()
    lookup_df['target_snap'] = lookup_df['snap'] - n_snaps

    # Drop the original snap so it doesn't conflict during the merge
    lookup_df = lookup_df.drop(columns=['snap'])

    # Perform an inner merge to filter master_df
    df = master_df.merge(
        lookup_df, 
        left_on=['sink_id', 'snap'], 
        right_on=['sink_id', 'target_snap'], 
        how='inner'
    )

    # clean up the extra lookup columns brought over by the merge
    df = df.drop(columns=['target_snap'])

    df = merge_snap_table(df, snap_table_df)

    return df


def n_snaps_extract(n_snaps, train_val=False):
    '''
    Returns a train or train val pair of star gas data n_snaps before the pre-stellar snap

    inputs:
        - n_snaps: int number of steps before the pre-stellar snaps
        - train_val: returns a train val pair if called

    returns:
        if train_val returns train and val pair DataFrames
        if not train_val returns just train DataFrame
    '''

    train_df = n_snaps_general(master_df=master_df_train, pre_df=raw_df_train, n_snaps=n_snaps)


    if train_val:

        val_df = n_snaps_general(master_df=master_df_val, pre_df=raw_df_val, n_snaps=n_snaps)


        return train_df, val_df
    
    else:
        return train_df



def artificial_data():
    '''
    Generates and artificial dataset to test internal_gas_model.py methods
    '''

    # set seed for reproducibility 
    np.random.seed(42)

    noise_to_signal = 2

    # define output feature "mass"
    df = pd.DataFrame()
    target = np.random.randn(1000)
    df['M_tot'] = target

    # define random slopes
    input_slopes = np.random.randn(10)

    print(input_slopes)

    # x = 1/m * y - 1/m * noise --> y = m * x + noise
    input_points = 1/(input_slopes.reshape(-1, 1)) * (target - noise_to_signal * np.random.randn(10,1000))


    inputs = ['x' + str(i+1) for i in range(10)]

    df[inputs] = input_points.T

    return df
