import pandas as pd
import matplotlib.pyplot as plt
from seaborn import heatmap
import numpy as np
import sys
import argparse
from process_data import n_snaps_extract, log_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression



features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84',       
            'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84'] 

features = ['T_med', 'B_med', 'rho_med'] # uncomment to look at reduced model

best_params = {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5} # best for reduced model


def regression_information_decay(features, snaps, model='linear'):
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
        train_df, val_df = log_data(features=features, train=train_df, val=val_df, train_val=True, normalize=True)

        X_train = train_df[features].values.reshape(-1, len(features))
        y_train = train_df['mf'].values

        if model == 'linear':
            model_ = LinearRegression().fit(X=X_train, y=y_train)
        elif model == 'random_forest':
            model_ = RandomForestRegressor(**best_params).fit(X=X_train, y=y_train)
        else:
            raise ValueError("Invalid model type. Choose 'linear' or 'random_forest'.")

        c_r2 = model_.score(X_train, y_train)
        r2s_train.append(c_r2)

        X_val = val_df[features].values.reshape(-1, len(features))
        y_val = val_df['mf'].values
        c_r2_val = model_.score(X_val, y_val)
        r2s_val.append(c_r2_val)

    
    print(pd.DataFrame({'r2_val': np.exp(np.array(r2s_val)), 'snaps': np.arange(snaps)}).corr())
        
    # plotting
    plt.figure(figsize=(15, 8))

    # create a bar chart
    plt.bar(np.arange(snaps) - 0.25, r2s_train, label='Training', width=0.5)
    plt.bar(np.arange(snaps) + 0.25, r2s_val, label='Validation', width=0.5)
    plt.xticks(np.arange(0, snaps, 5))
    plt.xlabel('Snapshots Before Pre-Sink Snapshot')
    plt.ylabel(r'$R^2$ Score')
    plt.title(r'Information Decay of Reduced Model ($\rho_{med}$, $B_{med}$, and $T_{med}$) - ' + model.capitalize() + ' Regression')


    plt.yscale('log')
    plt.legend()
    plt.savefig(f'figures/information_decay_{model}_regression.pdf', dpi=300)
    plt.show()


def random_forest_grid_search(features, df_train):


    X_train = df_train[features].values.reshape(-1, len(features))
    y_train = df_train[['mf', 'max_mult']].values.reshape(-1, 2)

    # Define the cross-validation strategy
    folds = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'max_features': ['sqrt', 0.3, 0.5],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize and run grid search
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all CPU cores
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=folds, scoring='r2', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # print the best parameters and the corresponding R^2 score
    print("Best Parameters found:", grid_search.best_params_)
    print("Best Validation R^2:", grid_search.best_score_)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', '-l', action='store_true', help='Perform linear regression information decay analysis')
    parser.add_argument('--random_forest', '-rf', action='store_true', help='Perform random forest information decay analysis')
    parser.add_argument('--rf_grid_search', '-rf_gs', action='store_true', help='Perform random forest grid search for hyperparameter tuning')
    args = parser.parse_args()

    if args.linear:
        regression_information_decay(features=features, snaps=42, model='linear')
    if args.random_forest:
        regression_information_decay(features=features, snaps=42, model='random_forest')
    if args.rf_grid_search:
        random_forest_grid_search(features=features, df_train=log_data(features=features, train_val=False, normalize=True, multiplicity=True))


if __name__ == '__main__':
    main()