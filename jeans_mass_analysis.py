import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from seaborn import heatmap
from process_data import log_data, n_snaps_extract


# Load pre-stellar gas data
df = log_data(features=['rho_med', 'T_med', 'snap'])
#df = log_data(train=n_snaps_extract(n_snaps=1), features=['rho_med', 'T_med', 'snap'])

#df = df[10**df['mf'] > 0.1] # get rid of small stars 
#df = df[df['T_med'] < 1.3] # remove really hot outliers

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




def rho_T_3d():
    
    x = df['rho_med'].values
    y = df['T_med'].values
    z = df['mf'].values

    X = np.linspace(min(x), max(x), 20)
    Y = np.linspace(min(y), max(y), 20)
    X, Y = np.meshgrid(X, Y)

    emperical_jean_cf = emperical_jeans_intercept()

    df_under_jeans = df[-0.5 * df['rho_med'] + 1.5 * df['T_med'] + theory_jeans_cf > df['mf']]
    print('under jeans theory', len(df_under_jeans))

    # analytical plane
    Z = -0.5 * X + 1.5 * Y + theory_jeans_cf

    # emperical plane
    #Z = -0.5 * X + 1.5 * Y + emperical_jean_cf

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

    plane = ax.plot_surface(X, Y, Z, alpha=0.2, label='Jeans Mass Plane')
    pred_plane = ax.plot_surface(X, Y, Z_pred, alpha=0.2, label='Linear Regression Plane')

    sc = ax.scatter(x, y, z, s=3, c=df['mf'], cmap='rainbow', alpha=0.5)
    plt.colorbar(sc, label=r'Total Mass ($M_\odot$) [log scale]')
    ax.set_xlabel(r'$\rho_{med}$')
    ax.set_ylabel(r'$T_{med}$')
    ax.set_zlabel(r'$M_f$')
    plt.title(r'$\rho_{med}$ and $T_{med}$ Scatter vs $M_f$')
    plt.show()


def coefficient_bootstrap(df):

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

    pred_floor = 1.5 * T - 0.5 * rho + theory_jeans_cf

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(mf, mf - pred_floor, s=10, alpha=0.5, c=df['T_med'], cmap='viridis')
    plt.colorbar(sc, label=r'$T_{med}$ (K) [log scale]')
    plt.xlabel(r'Final Mass ($\log_{10}M_\odot$)')
    plt.ylabel(r'Excess Mass above $M_J$ ($M_f - M_J$)')
    plt.plot(np.linspace(mf.min(), mf.max(), 20), np.zeros(20), linestyle=':', c='r', alpha=0.5)
    plt.title('Mass versus Mass in Excess of Analytical Jeans Mass Prediction')
    plt.savefig('figures/mass_vs_jeans_mass_residual.png', dpi=300)
    plt.show()


mass_vs_residual(df)

def temp_time_series():



    pass


#coefficient_bootstrap(df=df)

#rho_T_3d()