import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
import sys
import argparse
from process_data import n_snaps_extract, log_data, normalize_data, add_standard_error, merge_snap_table
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

'''
Investigation of multiple snapshots of data.

Looking at the dataset 3 pre-stellar vs 1 pre-stellar

Namespace Dependencies:
- seaborn
- pandas
- matplotlib
- scikit-learn
- numpy
- argparse
- pyarrow/fastparquet (for reading parquet files)
'''

# define features of interest, not looking at snap as were looking at changes in internal gas values with respect to time
features = ['T', 'T_med', 'T_16', 'T_84', 'T_std_er', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std_er', 'Ne_std',
        'B', 'B_med', 'B_16', 'B_84', 'B_std_er', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std_er', 'rho_std'] 

# Define data frame to be investigated


def plot_intrinsic_gas_heatmap(snaps=3, save=False):
    '''
    Plots a heatmap of the correlation matrix for the distribution of intrensic gas properties and mass.
    '''

    features.append('mf')
    
    df_t = n_snaps_extract(n_snaps=snaps, train_val=False)
    df_t = add_standard_error(df=df_t)
    df_t = log_data(features=features, train=df_t, train_val=False)

    correlation_matrix = df_t[features].corr()
    plt.figure(figsize=(12, 10))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Heatmap of Intrinsic Gas Properties ({snaps} snaps pre-sink snap)')
    plt.tight_layout()
    if save:
        plt.savefig(f'figures/time_series_heatmaps/internal_gas_heatmap_{snaps}_presink.pdf', dpi=300)
    plt.show()


def distribution_analysis(pre_df, old_df, feature, save_fig=False):
    '''
    inputs:
        -pre_df: DataFrame of pre-stellar data to be compared against
        -old_df: DataFrame of stellar gas data multiple snaps before pre-sink snap
        -feature: string of the feature of interest
    returns:
        None
        visualizes the distribution of differences in a given property
    '''

    pre_vals = pre_df[feature]
    old_vals = old_df[feature]

    # Ratio of increase because log(x) - log(y) = log(x/y)
    x = pre_vals - old_vals


    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # plot distribution histogram
    ax1.set_title(r'$\log_{10}$ of ratio in ' + feature + ' in 3 snaps')
    ax1.set_xlabel('Log of ratio in ' + feature + ' (dex)')
    ax1.set_ylabel('Count')
    ax1.hist(x, bins=100)
    ax1.set_yscale('log')


    # plot new and old values in a log log scatter plot
    ax2.set_ylabel('pre-stellar values of ' + feature + ' [log scale]')
    ax2.set_xlabel('3 snaps pre-stellar values of ' + feature + ' [log scale]')
    ax2.set_title('Pre-Stellar vs 3 snaps Pre-stellar of ' + feature)
    ax2.scatter(old_vals, pre_vals, alpha=0.5, s=10)
    ax2.plot(pre_vals, pre_vals, c='r', label='Constant Value Line')
    ax2.legend()

    # only plot if save_fig isnt called
    if not save_fig:
        plt.show()

    if save_fig:
        plt.savefig('figures/three_snap_vs_pre_stellar/time_comp_' + feature + '.pdf', dpi=300)


def visualize_random_forest(features, train_df, val_df):
    '''
    Takes training and validation DataFrames from the same snapshot and evaluates a random forest
    inputs:
        -features: list of strings of input features to be trained on
        -train_df: DataFrame of the training dataset (pre-normalized and log taken)
        -val_df: DataFrame of the validation dataset (pre-normalized and log taken)
    '''

    # define training data
    X_train = train_df[features].values.reshape(-1, len(features))
    y_train = train_df['mf'].values

    # define validation data
    X_val = val_df[features].values.reshape(-1, len(features))
    y_val = val_df['mf'].values

    # initialize and fit model
    model = RandomForestRegressor()
    model.fit(X=X_train, y=y_train)
    
    # score both train and validation
    r2_train = model.score(X=X_train, y=y_train)
    r2_val = model.score(X=X_val, y=y_val)

    # get predictions
    pred_train = model.predict(X=X_train)
    pred_val = model.predict(X=X_val)

    # create side by side of performance of both models
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(y_train, pred_train, s=10, alpha=0.5)
    ax1.plot(y_train, y_train, c='r')
    ax1.set_title(f'Random Forest Train $R^2$ = {r2_train:.2f}')
    ax1.set_xlabel(r'Ground Truth ($\log_{10}M_\odot$)')
    ax1.set_ylabel(r'Predicted ($\log_{10}M_\odot$)')

    ax2.scatter(y_val, pred_val, s=10, alpha=0.5)
    ax2.plot(y_val, y_val, c='r')
    ax2.set_title(f'Random Forest Validation $R^2$ = {r2_val:.2f}')
    ax2.set_xlabel(r'Ground Truth ($\log_{10}M_\odot$)')
    ax2.set_ylabel(r'Predicted ($\log_{10}M_\odot$)')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_3_snap_comp', '-plot_3sc', action='store_true', help='plots all statistics histograms pre-stellar vs 3 snaps pre-stellar')
    parser.add_argument('--plot_heatmap', '-hm', type=int, default=-1, help='plots a heatmap of properties from a specified number of snaps from the pre-sink snap')
    parser.add_argument('--save', '-s', action='store_true', help='Flag called to save the called figures')
    parser.add_argument('--random_forest', '-rf', type=int, default=-1, help='Evaluates the train and validation of a random forest a given number of snaps presink')
    args = parser.parse_args()

    if args.plot_3_snap_comp:

        # 3 snaps pre-sink snap
        df_train_min3, df_val_min3 = n_snaps_extract(n_snaps=3, train_val=True)
        df_train_min3, df_val_min3 = log_data(features=features, train=df_train_min3, val=df_val_min3, train_val=True, normalize=True)


        # pre-sink snap
        df_train_base = log_data(features=features, train_val=False, normalize=True)

        print('plotting...')

        # plott a comparision for each feature in features
        for feature in features:

            distribution_analysis(pre_df=df_train_base, old_df=df_train_min3, feature=feature, save_fig=args.save)
            plt.close()

        print('plotting complete.')


    if (args.plot_heatmap + 1):
        plot_intrinsic_gas_heatmap(snaps=args.plot_heatmap, save=args.save)


    if (args.random_forest + 1):

        rf_features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84',
        'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84'] 

        # n snaps pre-sink snap
        df_train_min, df_val_min = n_snaps_extract(n_snaps=args.random_forest, train_val=True)


        print(df_train_min.isna().sum(), df_val_min.isna().sum())

        # log and normalize the data
        df_train_min, df_val_min = log_data(features=features, train=df_train_min, val=df_val_min, train_val=True, normalize=True)

        visualize_random_forest(features=rf_features, train_df=df_train_min, val_df=df_val_min)

        plt.suptitle(f'Train-Validation Comparision {args.random_forest} snapshots pre-sink')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()

