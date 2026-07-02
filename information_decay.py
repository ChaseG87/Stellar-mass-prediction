import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
import sys
import argparse
from process_data import n_snaps_extract, log_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84',
        'B', 'B_med', 'B_16', 'B_84',  'rho', 'rho_med', 'rho_16', 'rho_84'] 


def linear_regression_information_decay(features, snaps):
    '''
    Looks at the decay of R^2 across all snapshots pre-sink for linear regression models trained on the features and snapshot of interest.
    inputs:
        - features: list of strings of input features to be fit to
        - snaps: int of the number of snapshots to be investigated
    returns:
        - None
        visualizes the decay of R^2 across all snapshots pre-sink for linear regression models trained on the features and snapshot of interest.
    '''

    r2s_train = []
    r2s_val = []

    for i in range(snaps):

        # extract data
        train_df, val_df = n_snaps_extract(n_snaps=i, train_val=True)
        train_df, val_df = log_data(features=features, train=train_df, val=val_df, train_val=True)

        X_train = train_df[features].values.reshape(-1, len(features))
        y_train = train_df['mf'].values

        model = LinearRegression().fit(X=X_train, y=y_train)

        c_r2 = model.score(X_train, y_train)
        r2s_train.append(c_r2)

        X_val = val_df[features].values.reshape(-1, len(features))
        y_val = val_df['mf'].values
        c_r2_val = model.score(X_val, y_val)
        r2s_val.append(c_r2_val)

    plt.figure(figsize=(15, 8))

    # create a bar chart
    plt.bar(np.arange(snaps) - 0.25, r2s_train, label='Training', width=0.5)
    plt.bar(np.arange(snaps) + 0.25, r2s_val, label='Validation', width=0.5)
    plt.xticks(np.arange(0, snaps, 5))
    plt.xlabel('Snapshots Before Pre-Sink Snapshot')
    plt.ylabel(r'$R^2$ Score')
    plt.title('Information Decay - Linear Regression')

    plt.yscale('log')
    plt.legend()
    plt.savefig('figures/information_decay_linear_regression.pdf', dpi=300)
    plt.show()


def random_forest_grid_search(features, df_train, df_val):
    pass


def random_forest_information_decay(features, snaps):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', '-l', action='store_true', help='Perform linear regression information decay analysis')
    parser.add_argument('--random_forest', '-rf', action='store_true', help='Perform random forest information decay analysis')
    args = parser.parse_args()

    if args.linear:
        linear_regression_information_decay(features=features, snaps=42)
    if args.random_forest:
        pass


if __name__ == '__main__':
    main()