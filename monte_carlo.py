import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from process_data import log_data


features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84',
            'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84',
              'T_std_er', 'Ne_std_er', 'B_std_er', 'rho_std_er', 'N_inc'] 


train_df, val_df = log_data(features=features, train_val=True)

# define default slices for resampling based on mass ranges
slices = ((-2, -1), (-1, 0), (0, 1), (1, 2))


def calculate_local_covariance(global_corr, row):
    '''
    Takes a global correlation matrix and the data of a row and generates a specific covariance matrix for that star.
    '''

    # Calculate individual property SEs
    T_std_er, Ne_std_er, B_std_er, rho_std_er = 10**row['T_std_er'], 10**row['Ne_std_er'], 10**row['B_std_er'], 10**row['rho_std_er']
    T, Ne, B, rho = 10**row['T'], 10**row['Ne'], 10**row['B'], 10**row['rho']

    # delta method from log space
    T_std_er *= 0.434/T
    Ne_std_er *= 0.434/Ne
    B_std_er *= 0.434/B
    rho_std_er *= 0.434/rho

    med_cf = np.sqrt(1.253)
    per_cf = np.sqrt(1.709)

    # Build the 16-element Standard Error vector matching input features
    se_vector = np.array([
        # Temperature block
        T_std_er, med_cf * T_std_er, per_cf * T_std_er, per_cf * T_std_er,
        # Electron Density block
        Ne_std_er, med_cf * Ne_std_er, per_cf * Ne_std_er, per_cf * Ne_std_er,
        # Magnetic Field block
        B_std_er, med_cf * B_std_er, per_cf * B_std_er, per_cf * B_std_er,
        # Mass Density block
        rho_std_er, med_cf * rho_std_er, per_cf * rho_std_er, per_cf * rho_std_er
    ]).astype(float)

    # Build the diagonal matrix
    D = np.diag(se_vector)

    # Construct the full 16x16 personalized covariance matrix
    star_cov_matrix = D @ global_corr @ D

    return star_cov_matrix

def monte_carlo_resample(
        train_df, 
        features=features, 
        slices=slices, 
        n_clones=5, 
        random_state=None, 
        std_er_cov = False
        ):
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

    features_clean = features.copy()

    for str in ['N_inc', 'T_std_er', 'Ne_std_er', 'B_std_er', 'rho_std_er']:
        features_clean.remove(str) if str in features else features_clean

    print(features_clean)

    if random_state is not None:
        np.random.seed(random_state)

    for slice in slices:
        slice_df = train_df[(train_df['mf'] >= slice[0]) & (train_df['mf'] < slice[1])]
        cov_matrix = slice_df[features_clean].cov().values
        corr_matrix = slice_df[features_clean].corr().values


        for index, row in slice_df.iterrows():

            mean_vector = row[features_clean].values

            if std_er_cov:
                cov_matrix_sp = calculate_local_covariance(global_corr=corr_matrix, row=row)
            else:
                cov_matrix_sp = cov_matrix / row['N_inc'] # Normalize covariance matrix by the number of samples

            resampled_features = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix_sp, size=n_clones)


            for i in range(n_clones):
                clone_dict = {}

                # 1. Map the drawn features to their original column names
                for col_name, val in zip(features_clean, resampled_features[i]):
                    clone_dict[col_name] = val

                # 2. Anchor variables: keep the target variable and identifier completely unchanged
                clone_dict['mf'] = row['mf'] 


                all_cloned_rows.append(clone_dict)

    # Convert the full list of dictionaries into a clean DataFrame all at once
    df_resampled = pd.DataFrame(all_cloned_rows)

    # Reorder columns to perfectly match the original training DataFrame structure
    features_clean.append('mf')

    df_resampled = df_resampled.reindex(columns=features_clean)

    return df_resampled


def sample_from_mc(sub_sample=100):
    # Sample from the produced monte carlo training data set a subset that represents each of the 5 slices

    pass

def main():

    resampled_df = monte_carlo_resample(train_df, n_clones=5, random_state=42)
    print(resampled_df.head())



if __name__ == '__main__':
    main()