import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load pre-stellar train data
raw_df_train = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq').reset_index()

# load pre-stellar validation data
raw_df_val = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_test.pq').reset_index()

# load full train data
master_df_train = pd.read_parquet('star_data/accreted_gas_properties.pq').reset_index()

# load full validation data
master_df_val = pd.read_parquet('star_data/accreted_gas_properties_test.pq').reset_index()




def log_train_val_data(features):
    '''
    inputs:
        - features: list of strings of input features to be investigated
    returns:
        - df_train, df_val: DataFrames of the train and validation data
    '''


    # pre-process taking the log of features
    df_train = np.log10(raw_df_train[features])
    df_val = np.log10(raw_df_val[features])

    # pre-process taking the log of the mass (target)
    df_train['M_tot'] = np.log10(raw_df_train['M_tot'])
    df_val['M_tot'] = np.log10(raw_df_val['M_tot'])

    return df_train, df_val


def n_snaps_dataset(master_df, pre_df, n_snaps):
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

    return df


def n_snaps_data(n_snaps, train_val=False):
    '''
    Returns a train or train val pair of star gas data n_snaps before the pre-stellar snap

    inputs:
        - n_snaps: int number of steps before the pre-stellar snaps
        - train_val: returns a train val pair if called

    returns:
        if train_val returns train and val pair DataFrames
        if not train_val returns just train DataFrame
    '''

    train_df = n_snaps_dataset(master_df=master_df_train, pre_df=raw_df_train, n_snaps=n_snaps)


    if train_val:

        val_df = n_snaps_dataset(master_df=master_df_val, pre_df=raw_df_val, n_snaps=n_snaps)

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

    plt.scatter(df[inputs[0]], df['M_tot'])
    plt.show()
    #print(df.head())
    return df

artificial_data()