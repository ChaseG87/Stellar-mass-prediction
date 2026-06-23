import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
import sys
import argparse
from process_data import n_snaps_extract, log_train_val_data

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
features = ['M_tot', 'T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
        'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std'] 

# Define data frame to be investigated

# 3 snaps pre-sink snap
df_train_min3, df_val_min3 = n_snaps_extract(n_snaps=3, train_val=True)
df_train_min3, df_val_min3 = log_train_val_data(features=features, train=df_train_min3, val=df_val_min3)


# pre-sink snap
df_train_base, df_val_base = log_train_val_data(features=features)


def plot_intrinsic_gas_heatmap(df):
    '''
    Plots a heatmap of the correlation matrix for the distribution of intrensic gas properties and mass.
    '''
        

    correlation_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Intrinsic Gas Properties')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_3_snap_comp', '-plot_3sc', action='store_true', help='plots all statistics histograms pre-stellar vs 3 snaps pre-stellar')
    args = parser.parse_args()

    if args.plot_3_snap_comp:
        print('plotting...')

        # plott a comparision for each feature in features
        for feature in features:

            distribution_analysis(pre_df=df_train_base, old_df=df_train_min3, feature=feature, save_fig=True)
            plt.close()

        print('plotting complete.')


if __name__ == '__main__':
    main()

