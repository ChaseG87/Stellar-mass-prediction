import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from process_data import log_data, normalize_data
from data_analysis_v1 import plot_intrinsic_gas_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from monte_carlo import monte_carlo_resample


features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'PE_global', 'L_mag'] 

# read file
df_cut = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut.pq').reset_index()
df_cut.drop(columns=['M_tot'])
df_cut['PE_global'] *= -1  # make potential energy positive for analysis

df_first_snap = pd.read_parquet('star_data/first_snap_table.pq').reset_index()
df_cut = pd.merge(df_cut, df_first_snap[['pid', 'mf']], left_on='sink_id', right_on='pid', how='outer')


# same preprocessing for validation set
df_cut_val = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut_test.pq').reset_index()
df_cut_val.drop(columns=['M_tot'])
df_cut_val['PE_global'] *= -1  # make potential energy positive for analysis

df_first_snap_val = pd.read_parquet('star_data/first_snap_table_test.pq').reset_index()
df_cut_val = pd.merge(df_cut_val, df_first_snap_val[['pid', 'mf']], left_on='sink_id', right_on='pid', how='outer')


# add specific angular momentum
df_cut['L_mag'] = np.sqrt(df_cut['L_vec_1']**2 + df_cut['L_vec_2']**2 + df_cut['L_vec_3']**2)
df_cut_val['L_mag'] = np.sqrt(df_cut_val['L_vec_1']**2 + df_cut_val['L_vec_2']**2 + df_cut_val['L_vec_3']**2)

df_clean_cut = df_cut[(df_cut['N_inc'] >= 10)  & (df_cut['mf'] > 0.1)]
df_clean_val = df_cut_val[(df_cut_val['N_inc'] >= 10)  & (df_cut_val['mf'] > 0.1)]

#print(df_clean_val['mf'].count(), df_cut_val['mf'].count())

#print(df_cut['mf'].count() - df_clean_cut['mf'].count())
# log taken within the function
#plot_intrinsic_gas_heatmap(df=df_clean_cut)

# There are 108 stars that have less than 10 particles
# 65 of those have a max_dist_peak less than 0.005 pcs

# take log of both
X = np.log10(df_clean_cut[features].values.reshape(-1, len(features)))
y = np.log10(df_clean_cut['mf'].values)

# apply to validation set
X_val = np.log10(df_clean_val[features].values.reshape(-1, len(features)))
y_val = np.log10(df_clean_val['mf'].values)


# standardize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

def mass_vs_n_inc():
    plt.figure(figsize=(10, 8))
    plt.scatter(df_clean_cut['mf'], (df_clean_cut['N_inc'] * 0.001)/df_clean_cut['mf'], s=10, alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Mass in 0.01 pc sphere pre-sink / Final Mass')
    plt.xlabel(r'Final Mass ($M_\odot$)')
    plt.title(r'Total Mass vs Proportion of Mass in 0.01 pc sphere')
    plt.tight_layout()
    #plt.savefig('mass_vs_mass_proportion.png', dpi=300)
    plt.show()

#mass_vs_n_inc()

def lasso_cv_analysis():

    model = LassoCV(cv=5, max_iter=10000).fit(X, y)

    pred = model.predict(X)

    r2 = model.score(X, y)

    used_features = [features[i] for i in range(len(features)) if model.coef_[i]]

    mse = model.mse_path_[model.alpha_ == model.alphas_].flatten().mean()
    var = np.var(y, ddof=0)

    print(f'Validation average R^2: {1 - (mse/var):.4f}')


    # plot prediction
    plt.figure(figsize=(10, 8))
    plt.scatter(y, pred, s=10, alpha=0.5)
    plt.plot(y, y, c='r')

    plt.title(f'LASSO-CV applied to 0.01 pc cut, $R^2$ = {r2:.2f}, num_features = {len(used_features)}')

    plt.xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    plt.ylabel(r'Prediction ($\log_{10} M_\odot$)')

    #plt.savefig('figures/lasso_cv_0.01_patch.pdf', dpi=300)
    plt.show()

#lasso_cv_analysis()

def lasso_cv_validation():
    '''
    Plots the training and validation scatter plots of a LASSO-CV model.
    '''

    model = LassoCV(cv=5, max_iter=10000).fit(X, y)

    print(model.coef_)

    pred_train = model.predict(X)
    r2_train = model.score(X, y)

    pred_val = model.predict(X_val)
    r2_val = model.score(X_val, y_val)

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(y, pred_train, s=10, alpha=0.5)
    ax1.plot(y, y, c='r')
    ax1.set_xlabel(r'Ground Truth Mass ($\log_{10}M_\odot$)')
    ax1.set_ylabel(r'Predicted Mass ($\log_{10}M_\odot$)')
    ax1.set_title(f'LASSO-CV training set - $R^2 = ${r2_train:.2f}')

    ax2.scatter(y_val, pred_val, s=10, alpha=0.5)
    ax2.plot(y_val, y_val, c='r')
    ax2.set_xlabel(r'Ground Truth Mass ($\log_{10}M_\odot$)')
    ax2.set_ylabel(r'Predicted Mass ($\log_{10}M_\odot$)')
    ax2.set_title(f'LASSO-CV validation set - $R^2 = ${r2_val:.2f}')

    plt.suptitle('LASSO-CV Training and Validation - Patch Data')

    plt.savefig('figures/patch_figures/lasso_cv_patch_validation.pdf', dpi=300)

    plt.show()

#lasso_cv_validation()



def random_forest_analysis():

    best_params = {'max_depth': None, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2}

    model = RandomForestRegressor(**best_params)
    model.fit(X, y)

    pred = model.predict(X_val)

    r2 = model.score(X_val, y_val)

    print(f'Random Forest Validation R^2: {r2:.4f}')

    fig = plt.figure(figsize=(16, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # plot training
    ax1.scatter(y, model.predict(X), s=10, alpha=0.5)
    ax1.plot(y, y, c='r')

    ax1.set_title(f'Random Forest Train $R^2$ = {model.score(X, y):.2f}')
    ax1.set_xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    ax1.set_ylabel(r'Prediction ($\log_{10} M_\odot$)')

    # plot validation
    ax2.scatter(y_val, pred, s=10, alpha=0.5)
    ax2.plot(y_val, y_val, c='r')

    ax2.set_xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    ax2.set_ylabel(r'Prediction ($\log_{10} M_\odot$)')
    ax2.set_title(f'Random Forest Validation $R^2$ = {r2:.2f}')

    fig.suptitle(f'Random Forest Training and Validation - Patch Data')

    plt.savefig('figures/patch_figures/random_forest_patch_validation.pdf', dpi=300)

    plt.show()

#random_forest_analysis()


def perform_monte_carlo():

    features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'N_inc'] 
    
    features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 
                'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84', 'N_inc'] 

    # take the log of the cut down dataset
    df = log_data(features=features, train=df_clean_cut)

    slices = ((-1, 0), (0, 1), (1, 2))

    # run the resample on the data
    df_mc = monte_carlo_resample(df, features=features, slices=slices, random_state=42)

    features.pop()

    X = df_mc[features].values.reshape(-1, len(features))
    y = df_mc['mf'].values

    model = LassoCV(cv=5).fit(X, y)
    pred = model.predict(X)

    r2 = model.score(X, y)


    plt.figure(figsize=(12, 8))
    plt.scatter(y, pred, s=10, alpha=0.2)
    plt.plot(y, y, c='r')

    plt.title(f'LASSO-CV applied to 0.01 pc patch (monte-carlo), $R^2$ = {r2:.2f}')

    plt.xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    plt.ylabel(r'Prediction ($\log_{10} M_\odot$)')

    plt.savefig('figures/patch_figures/lasso_patch_monte_carlo.pdf', dpi=300)

    plt.show()

#perform_monte_carlo()

def weighted_lasso(weight_method='sqrt'):
    '''
    Runs a lasso cv model that is weighted based on the mass of stars
    '''

    if weight_method == 'sqrt':
        weight = np.sqrt(10**y)

    elif weight_method == 'log':
        weight = y - y.min() + 1

    elif weight_method == 'none':
        weight = 1

    else:
        raise ValueError("Invalid weighting methodology selected.")

    # set up the alpha hyperparameter grid to search over
    param_grid = {'alpha': np.logspace(-4, 1, 50)}

    # initialize the base Lasso model and cross-validation strategy
    lasso = Lasso(max_iter=10000)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # set up the Grid Search
    grid_search = GridSearchCV(
        estimator=lasso, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='neg_mean_squared_error'
    )

    # pass the weights to the fit method via fit_params
    grid_search.fit(X, y, sample_weight=weight)

    # Best alpha and model
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    pred_train = best_model.predict(X)
    r2_train = best_model.score(X, y)

    pred_val = best_model.predict(X_val)
    r2_val = best_model.score(X_val, y_val)

    # plot results
    fig = plt.figure(figsize=(16,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    ax1.scatter(y, pred_train, s=10, alpha=0.7)
    ax1.plot(y, y, c='r')
    ax1.set_title(f'Training Performance - $R^2 =$ {r2_train:.2f}')
    ax1.set_xlabel(r'Ground Truth Mass $M_\odot$')
    ax1.set_ylabel(r'Predicted Mass $M_\odot$')

    ax2.scatter(y_val, pred_val, s=10, alpha=0.7)
    ax2.plot(y_val, y_val, c='r')
    ax2.set_title(f'Validation Peformance - $R^2 =$ {r2_val:.2f}')
    ax2.set_xlabel(r'Ground Truth Mass $\log_{10}M_\odot$')
    ax2.set_ylabel(r'Predicted Mass $\log_{10}M_\odot$')

    plt.suptitle(f'Weighted LASSO-CV model - 0.01 Patch - weight_method = {weight_method}')
    plt.savefig('figures/patch_figures/weighted_lasso/lasso_cv_0.01_patch_weighted_' + weight_method + '.pdf', dpi=300)

    plt.show()

weighted_lasso(weight_method='sqrt')