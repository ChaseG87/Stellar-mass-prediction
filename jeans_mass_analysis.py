import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from seaborn import heatmap
from process_data import log_data, n_snaps_extract
import pickle
import argparse
import sys


# Load pre-stellar gas data
df = log_data(features=['rho_med', 'T_med', 'snap', 'N_fb'], primaries=True)
df = log_data(train=n_snaps_extract(n_snaps=2), features=['rho_med', 'T_med', 'snap', 'N_fb'])
#df = df.dropna()


#df = df[10**df['mf'] > 0.1] # get rid of small stars 
#df = df[df['T_med'] < 1.3] # remove really hot outliers
#df = df[df['primary'] == 1] # look only at primaries in accretion


# define theory jeans mass constant
theory_jeans_cf = 0.5917


def emperical_jeans_intercept():
    '''
    Find the emperical y-intercept of best fit for the jeans model and R^2 of the regression
    '''

    # define variables
    rho = df['rho_med'].values
    T = df['T_med'].values
    mf = df['mf'].values

    # the mean of residuals is 0 in a linear regression model
    intercept = np.mean(mf - (1.5*T - 0.5*rho))

    r2 = r2_score(mf, 1.5*T - 0.5*rho + intercept)

    print("R^2 of jeans model: ", r2)
    return intercept




def rho_T_3d(emp_plane=False, ana_plane=False, lr_plane=False):
    '''
    generates a 3d plot of rho_med vs T_med vs mf. 
    '''
    
    x = df['rho_med'].values
    y = df['T_med'].values
    z = df['mf'].values
    #z = np.log10(df['mf'].values + 10**(df['N_fb']*10e3))

    X = np.linspace(min(x), max(x), 20)
    Y = np.linspace(min(y), max(y), 20)
    X, Y = np.meshgrid(X, Y)

    emperical_jean_cf = emperical_jeans_intercept()

    df_under_jeans = df[-0.5 * df['rho_med'] + 1.5 * df['T_med'] + theory_jeans_cf > z]
    print('under jeans theory', len(df_under_jeans))

    # analytical plane
    if ana_plane:
        a_Z = -0.5 * X + 1.5 * Y + theory_jeans_cf

    # emperical plane
    if emp_plane:
        e_Z = -0.5 * X + 1.5 * Y + emperical_jean_cf

    # fit linear regression model
    model = LinearRegression().fit(df[['rho_med', 'T_med']].values.reshape(-1, 2), z)

    # calculate Z plane values from linear regression
    Z_pred = model.coef_[0] * X + model.coef_[1] * Y + model.intercept_
    # print coefficients and scores of linear regression
    print('rho_med coef: ', model.coef_[0],  '\nT_med coef:', model.coef_[1])
    print('R^2 of blind fit: ', model.score(df[['rho_med', 'T_med']].values.reshape(-1, 2), z))

    # calculate cosine similarity between theoretical and regression planes
    dot_product = np.dot(model.coef_, np.array([-0.5, 1.5]))
    norm_v1 = np.linalg.norm(model.coef_)
    norm_v2 = np.linalg.norm(np.array([-0.5, 1.5]))

    similarity = dot_product / (norm_v1 * norm_v2)
    print(f'cosine similarity between jeans and prediction planes: {similarity}')

    # 3D plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    if ana_plane:
        a_plane = ax.plot_surface(X, Y, a_Z, alpha=0.2, label='Jeans Mass Plane (Analytical)')

    if emp_plane:
        e_plane = ax.plot_surface(X, Y, e_Z, alpha=0.2, label='Jeans Mass Plane (Emperical)')

    if lr_plane:
        pred_plane = ax.plot_surface(X, Y, Z_pred, alpha=0.2, label='Linear Regression Plane')

    sc = ax.scatter(x, y, z, s=3, c=df['mf'], cmap='rainbow', alpha=0.5)
    plt.colorbar(sc, label=r'Total Mass ($M_\odot$) [log scale]')
    ax.set_xlabel(r'$\rho_{med}$')
    ax.set_ylabel(r'$T_{med}$')
    ax.set_zlabel(r'$M_f$')
    plt.title(r'$\rho_{med}$ and $T_{med}$ Scatter vs $M_f$')
    plt.show()


def coefficient_bootstrap(df):
    '''
    Runs a bootstrap of the given DataFrame's T_med and rho_med linear regression coefficients
    to generate a confidence interval.
    '''

    n_iterations = 2000  # Number of bootstrap samples
    bootstrapped_coefs = []
    

    for i in range(n_iterations):
        # Resample rows with replacement to create a pseudo-dataset of the same size
        X_resample, y_resample = resample(df[['rho_med', 'T_med']].values.reshape(-1, 2), df['mf'].values, replace=True, random_state=i)
        
        # Fit the linear model to the bootstrapped data
        model = LinearRegression()
        model.fit(X_resample, y_resample)
        
        # Store the coefficients for this iteration
        bootstrapped_coefs.append(model.coef_)
    
    # Convert to a clean numpy array (shape: n_iterations, n_features)
    bootstrapped_coefs = np.array(bootstrapped_coefs)
    
    # Calculate Empirical Percentiles (95% Error Bars)
    # We look at the 2.5th and 97.5th percentiles to get a 95% confidence interval
    lower_bounds = np.percentile(bootstrapped_coefs, 2.5, axis=0)
    upper_bounds = np.percentile(bootstrapped_coefs, 97.5, axis=0)
    
    # Calculate the mean or standard error of the coefficients
    mean_coefs = np.mean(bootstrapped_coefs, axis=0)
    std_errors = np.std(bootstrapped_coefs, axis=0)
    
    # Print the Results
    feature_names = ["log(density)", "log(temperature)"]
    for idx, name in enumerate(feature_names):
        print(f"{name.capitalize()}:")
        print(f"  Mean Coefficient: {mean_coefs[idx]:.3f}")
        print(f"  Standard Error:   {std_errors[idx]:.3f}")
        print(f"  95% CI Error Bar: [{lower_bounds[idx]:.3f}, {upper_bounds[idx]:.3f}]")
        print("-" * 30)


def mass_vs_residual(df):
    '''
    Plots the final mass vs the residual from the jeans mass prediction.
    '''

    rho = df['rho_med'].values
    T = df['T_med'].values
    mf = df['mf'].values

    # define the theoretical jeans mass using the pre-stellar snapshot
    pred_floor = 1.5 * T - 0.5 * rho + theory_jeans_cf

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(mf, mf - pred_floor, s=10, alpha=0.5, c=df['T_med'], cmap='viridis')
    plt.colorbar(sc, label=r'$T_{med}$ (K) [log scale]')
    plt.xlabel(r'Final Mass ($\log_{10}M_\odot$)')
    plt.ylabel(r'Excess Mass above $M_J$ ($M_f - M_J$)')
    plt.plot(np.linspace(mf.min(), mf.max(), 20), np.zeros(20), linestyle=':', c='r', alpha=0.5)
    plt.title('Mass versus Mass in Excess of Analytical Jeans Mass Prediction')
    #plt.savefig('figures/mass_vs_jeans_mass_residual.png', dpi=300)
    plt.show()

def mass_time_series():

    with open("star_data/path_lookup.p", "rb") as ff:

        path_lookup = pickle.load(ff)

        plt.plot(path_lookup["29233"][:,0], path_lookup["29233"][:,9])

    plt.show()

def temp_time_series():

    df = log_data(train=n_snaps_extract(n_snaps=0), features=['rho_med', 'T_med', 'snap'])
    ids = df[(df['T_med'] > 1.3) & (10**df['mf'] > 0.1)].reset_index()['sink_id']

    df_idx = set(n_snaps_extract(n_snaps=-1).set_index('sink_id').index)

    ids = [id for id in ids if id in df_idx]
    id_dict = {id:[] for id in ids}


    plt.figure(figsize=(10, 8))


    for snap in range (-1, 5):
        # going 2 snapshots forward in time stars start to have 0 accreting particles
        snap_df = log_data(train=n_snaps_extract(n_snaps=snap), features=['rho_med', 'T_med', 'snap'])

        for id in ids:
            id_dict[id].append(snap_df.loc[id]['T_med'])

    vmin, vmax = -1, 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    
    for id in ids:
        plt.plot(range(-4 , 2), id_dict[id][::-1], alpha=0.2, c=cmap(norm(df.loc[id]['mf'])))
        plt.scatter(range(-4, 2), id_dict[id][::-1], alpha=0.2, c=cmap(norm(df.loc[id]['mf'])))

    plt.title(r'Outlier Hot Stars $T_{med}$ vs. Snapshots')
    plt.xlabel('Snapshots Relative to Pre-sink Snapshot')
    plt.ylabel(r'$T_{med}$')
    plt.savefig('figures/outliers_temp_time_series.png', dpi=300)
    plt.show()

    #print(len(snap_df))

temp_time_series()
#mass_time_series()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_3d', '-p3d', action='store_true', help='displays a 3d plot of rho vs T vs mass')
    parser.add_argument('--linear', '-l', action='store_true', help='displays the linear model plane in the 3D plot')
    parser.add_argument('--emperical_jeans', '-ej', action='store_true', help='displays the emperical linear regression jeans plane')
    parser.add_argument('--analytical_jeans', '-aj', action='store_true', help='displays the analytical linear regression jeans plane')
    parser.add_argument('--resample', '-rs', action='store_true', help='runs a bootstrap resample of the linear regression coefficients')
    parser.add_argument('--mass_vs_residuals', '-mr', action='store_true', help='plots the masses vs the residual mass compared to the analytical jeans mass')
    args = parser.parse_args()

    if args.plot_3d:
        rho_T_3d(emp_plane=args.emperical_jeans, ana_plane=args.analytical_jeans, lr_plane=args.linear)

    if args.mass_vs_residuals:
        mass_vs_residual(df=df)

    if args.resample:
        coefficient_bootstrap(df=df)

    if len(sys.argv) == 1:
        raise ValueError('No Flag Called')


if __name__ == '__main__':
    main()