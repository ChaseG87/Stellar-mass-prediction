import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_data import log_data


features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84',
            'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84', 'N_inc'] 


train_df, val_df = log_data(features=features, train_val=True, normalize=True)

# define default slices for resampling based on mass ranges
slices = ((-2, -1), (-1, 0), (0, 1), (1, 2))


def monte_carlo_resample(train_df, features=features, slices=slices, n_clones=5, random_state=None):
    """
    Performs Monte Carlo resampling on training data train_df.
    
    Parameters:
    - train_df: DataFrame containing the training data with features and target variable.
    - n_clones: int, number of clones to create of each data point.
    - slices: tuple of tuples determining the ranges of mass buckets to be investigated.
    - random_state: int or None, random seed for reproducibility.
    
    Returns:
    - resampled_mf: numpy array of shape (n_simulations,) containing the resampled target variable values.
    """

    
    # Break up the training data into slices to create covarience matrices for each row
    # For a given row, normalize with respect to the number of samples for each data point
    # sample from the mulivariate normal distribution to get features
    # keep mass the same and add to a new dataframe of resampled data points

    # A list to collect rows across all slices efficiently
    all_cloned_rows = []

    features_no_N = features.copy()
    features_no_N.remove('N_inc')
    print(features_no_N)

    if random_state is not None:
        np.random.seed = random_state

    for slice in slices:
        slice_df = train_df[(train_df['mf'] >= slice[0]) & (train_df['mf'] < slice[1])]
        cov_matrix = slice_df[features_no_N].cov().values


        for index, row in slice_df.iterrows():

            mean_vector = row[features_no_N].values

            cov_matrix_sp = cov_matrix / row['N_inc'] # Normalize covariance matrix by the number of samples

            resampled_features = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix_sp, size=n_clones)


            for i in range(n_clones):
                clone_dict = {}

                # 1. Map the drawn features to their original column names
                for col_name, val in zip(features_no_N, resampled_features[i]):
                    clone_dict[col_name] = val

                # 2. Anchor variables: keep the target variable and identifier completely unchanged
                clone_dict["mf"] = row["mf"]
                clone_dict["N_inc"] = row["N_inc"]

                all_cloned_rows.append(clone_dict)

    # Convert the full list of dictionaries into a clean DataFrame all at once
    df_resampled = pd.DataFrame(all_cloned_rows)

    # Reorder columns to perfectly match the original training DataFrame structure
    df_resampled = df_resampled.reindex(columns=train_df.columns)

    return df_resampled


def sample_from_mc(sub_sample=100):
    # Sample from the produced monte carlo training data set a subset that represents each of the 5 slices

    pass

def main():

    resampled_df = monte_carlo_resample(train_df, n_clones=5, random_state=42)



if __name__ == '__main__':
    main()