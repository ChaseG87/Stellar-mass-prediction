import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from process_data import log_data, normalize_data
from data_analysis_v1 import plot_intrinsic_gas_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from monte_carlo import monte_carlo_resample


features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'PE_global', 'L_mag'] 

# read file
df_cut = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut.pq').reset_index()
df_cut.drop(columns=['M_tot'])
df_cut['PE_global'] *= -1  # make potential energy positive for analysis

df_first_snap = pd.read_parquet('star_data/first_snap_table.pq').reset_index()
df_cut = pd.merge(df_cut, df_first_snap[['pid', 'mf']], left_on='sink_id', right_on='pid', how='outer')


# add specific angular momentum
df_cut['L_mag'] = np.sqrt(df_cut['L_vec_1']**2 + df_cut['L_vec_2']**2 + df_cut['L_vec_3']**2)


df_clean_cut = df_cut[(df_cut['N_inc'] >= 10)  & (df_cut['mf'] > 0.1)]

#print(df_cut['mf'].count() - df_clean_cut['mf'].count())
# log taken within the function
#plot_intrinsic_gas_heatmap(df=df_clean_cut)

# There are 108 stars that have less than 10 particles
# 65 of those have a max_dist_peak less than 0.005 pcs

# take log of both
X = np.log10(df_clean_cut[features].values.reshape(-1, len(features)))
y = np.log10(df_clean_cut['mf'].values)

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
# standardize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)


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


def random_forest_analysis():

    best_params = {'max_depth': None, 'max_features': 0.5, 'min_samples_leaf': 2, 'min_samples_split': 2}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    r2 = model.score(X_test, y_test)

    print(f'Random Forest Validation R^2: {r2:.4f}')

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # plot prediction
    ax1.scatter(y_test, pred, s=10, alpha=0.5)
    ax1.plot(y_test, y_test, c='r')

    fig.suptitle(f'Random Forest applied to 0.01 pc cut')

    ax1.set_xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    ax1.set_ylabel(r'Prediction ($\log_{10} M_\odot$)')
    ax1.set_title(f'Random Forest Validation $R^2$ = {r2:.2f}')

    ax2.scatter(y_train, model.predict(X_train), s=10, alpha=0.5)
    ax2.plot(y_train, y_train, c='r')

    ax2.set_title(f'Random Forest Train $R^2$ = {model.score(X_train, y_train):.2f}')
    ax2.set_xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
    ax2.set_ylabel(r'Prediction ($\log_{10} M_\odot$)')

    #plt.savefig('figures/random_forest_0.01_patch.pdf', dpi=300)
    plt.show()

#random_forest_analysis()

def perform_monte_carlo():

    features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'N_inc'] 
    
    features = ['T', 'T_med', 'T_16', 'T_84', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 
                'B', 'B_med', 'B_16', 'B_84', 'rho', 'rho_med', 'rho_16', 'rho_84', 'N_inc'] 

    # take the log of the cut down dataset
    df = log_data(features=features, train=df_clean_cut, normalize=True)

    slices = ((-1, 0), (0, 1), (1, 2))

    # run the resample on the data
    df_mc = monte_carlo_resample(df, features=features, slices=slices, random_state=42)

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

    plt.savefig('figures/lasso_patch_monte_carlo.pdf', dpi=300)

    plt.show()

perform_monte_carlo()