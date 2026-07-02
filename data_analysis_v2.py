import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pwlf
import numpy as np
import sys
import argparse
import scipy.stats as stats
from scipy.optimize import minimize
from process_data import merge_snap_table

'''
Namespace Dependencies:
- pandas
- matplotlib
- scikit-learn
- numpy
- pwlf
- argparse
- pyarrow/fastparquet (for reading parquet files)
'''


# Read the pre stellar parquet file into a DataFrame
df = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq').reset_index()
df = merge_snap_table(df)

# add distance from center metric to the DataFrame
df['sp_kin'] = df['E_kin']/df['mf']
df['sp_mag'] = df['E_mag']/df['mf']
df['sp_int'] = df['E_int']/df['mf'] # poorly correlated
# There are NaNs in gravitation so it is not being investigated


class LogNormalPowerLawFitter:
    def __init__(self):
        self.X0 = None  # Log-space peak location
        self.C = None   # Log-space peak height
        self.Xb = None  # Log-space transition boundary
        self.A = None   # Parabola curvature parameter
        self.m = None   # Resulting high-end power-law slope
        
    def piecewise_log_model(self, X, X0, C, Xb, A):
        '''
        Calculates Y given X and the independent shape parameters.
        '''
        # Enforce smoothness
        m = -2 * A * (Xb - X0)
        
        # calculate log-normal component (parabola)
        y_ln = -A * (X - X0)**2 + C
        
        # Calculate power-law component (linear line intersecting at Xb)
        Yb = -A * (Xb - X0)**2 + C  # Value at the boundary
        y_pl = m * (X - Xb) + Yb
        
        # Merge piece-wise arrays based on the boundary condition
        return np.where(X <= Xb, y_ln, y_pl)

    def fit(self, x_data, y_data):
        X_log = np.log10(x_data)
        Y_log = np.log10(y_data)
        
        # Loss function
        def loss_function(params):
            X0, C, Xb, A = params
            Y_pred = self.piecewise_log_model(X_log, X0, C, Xb, A)
            return np.sum((Y_log - Y_pred)**2)
        
        # Initial guesses: [Peak center, Peak height, Breakpoint, Curvature]
        initial_guess = [X_log.mean(), Y_log.max(), X_log.mean() + 0.5, 0.5]
        
        # Set realistic bounds to prevent the optimizer from drifting to infinity
        bounds = [
            (X_log.min(), X_log.max()),      # X0 must stay within data range
            (Y_log.min(), Y_log.max() + 1),  # C height limits
            (X_log.min(), X_log.max()),      # Xb breakpoint
            (1e-3, 10.0)                     # A must be positive for downward parabola
        ]
        
        # Run constraints optimization
        res = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Extract and save optimal parameters
        self.X0, self.C, self.Xb, self.A = res.x
        self.m = -2 * self.A * (self.Xb - self.X0)
        return self

    def predict(self, x_eval):
        X_eval_log = np.log10(x_eval)
        Y_fit_log = self.piecewise_log_model(X_eval_log, self.X0, self.C, self.Xb, self.A)
        return 10**Y_fit_log


def linear_fit(df, feature, axis):
    '''
    Fits a linear model of a given feature to the total mass mf
    '''

    # fit model
    model = LinearRegression()
    X = np.log10(df[[feature]]).values.reshape(-1, 1)
    y = np.log10(df['mf']).values
    model.fit(X, y)

    # score model
    r2 = model.score(X, y)

    #plot scatter
    axis.scatter(X, y, s=10, alpha=0.7)

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = model.predict(xs.reshape(-1, 1))
    axis.plot(xs, ys, c='r', linestyle='--', label=f'$r^2$ = {r2:.2f}')

    # labels and titles
    axis.legend()
    axis.set_title(f'Linear fit of {feature} vs mf')
    axis.set_xlabel(r'$\log_{10}$' + feature)
    axis.set_ylabel(r'$\log_{10}$ mf ($M_\odot$)')


    residuals = y - model.predict(X)

    return X, residuals


def piecewise_fit(df, feature, axis, n=2):
    '''
    Fits a piecewise power-law of a given feature to mf with n breakpoints
    '''

    X = np.log10(df[feature]).values
    y = np.log10(df['mf']).values

    # initialize the model object
    model = pwlf.PiecewiseLinFit(X, y)

    # define the number of breakpoints 
    model.fit(n)

    #plot scatter
    axis.scatter(X, y, s=10, alpha=0.7)
    

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = model.predict(xs)
    axis.plot(xs, ys, c='r', linestyle='--')

    #labels and titles
    axis.set_title(f'Piecewise fit, n={n}, {feature} vs mf')
    axis.set_xlabel(r'$\log_{10}$' + feature)
    axis.set_ylabel(r'$\log_{10}$ mf ($M_\odot$)')


    residuals = y - model.predict(X)

    return X, residuals


def polynomial_fit(df, feature, axis, n=2):
    '''
    Fits a polynomial model of degree n for given feature to the total mass mf
    '''

    # fit model
    poly_model = make_pipeline(PolynomialFeatures(degree=n), LinearRegression())
    X = np.log10(df[[feature]]).values.reshape(-1, 1)
    y = np.log10(df['mf']).values
    poly_model.fit(X, y)

    # score model
    r2 = poly_model.score(X, y)

    #plot scatter
    axis.scatter(X, y, s=10, alpha=0.7)

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = poly_model.predict(xs.reshape(-1, 1))
    axis.plot(xs, ys, c='r', linestyle='--', label=f'$r^2$ = {r2:.2f}')

    # labels and titles
    axis.legend()
    axis.set_title(f'Polynomial fit of {feature} vs mf')
    axis.set_xlabel(r'$\log_{10}$' + feature)
    axis.set_ylabel(r'$\log_{10}$ mf ($M_\odot$)')

    residuals = y - poly_model.predict(X)

    return X, residuals


def log_normal_power_law(df, feature, axis):
    # raw values
    x_raw = df[feature].values
    y_raw = df['mf'].values

    # Initialize and fit the mixture profile
    fitter = LogNormalPowerLawFitter()
    fitter.fit(x_raw, y_raw)

    # generate Evaluation line for plotting
    x_plot = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 200)
    y_plot = fitter.predict(x_plot)

    # render the Figure in Log-Log Space
    axis.scatter(np.log10(x_raw), np.log10(y_raw), s=10, alpha=0.7)
    axis.plot(np.log10(x_plot), np.log10(y_plot), color='r', linestyle='--', 
             label=f"Spliced Model (Slope = {fitter.m:.2f})")

    # plot vertical visual marker indicating exactly where the smooth splice happened
    axis.axvline(fitter.Xb, color="black", linestyle=":", alpha=0.7, 
                label=f"Transition Window (10^{fitter.Xb:.1f})")

    # labels and title
    axis.set_xlabel(r"$\log_{10}$ " + feature)
    axis.set_ylabel(r"$\log_{10}$ mf ($M_\odot$)")
    axis.set_title("Log-Normal Power-Law Mixture Fit of " + feature + " vs mf")
    axis.legend()

    # define residuals
    residuals = np.log10(y_raw) - np.log10(fitter.predict(x_raw))

    return np.log10(x_raw), residuals


def chi_squared_comparison(df, feature):
    # can only compare linear models with linear parameters (no log_normal_power_law or piecewise)


    pass


def gaussian_residuals_check(x, residuals, title, axis):

    # plot histogram of residuals
    axis.hist(residuals, bins=30, density=True)

    # calculate ideal gaussian fit
    mu = np.mean(residuals)
    sigma = np.std(residuals)
    x_axis = np.linspace(min(residuals), max(residuals), 100)
    gaussian_pdf = stats.norm.pdf(x_axis, loc=mu, scale=sigma)

    # plot gaussian pdf
    axis.plot(x_axis, gaussian_pdf, alpha=0.5, c='r', label="Gaussian fit")
    axis.set_title(title)
    axis.legend()


def residuals_scatter(x, residuals, title, axis):

    axis.set_title(f'{title} against X space')
    axis.scatter(x, residuals, s=10, alpha=0.7)
    

def residual_comparision(energy):


    # plot how fits of each method
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    x, linear_res  = linear_fit(df, energy, ax1)
    x, pw_res = piecewise_fit(df, energy, ax2, n=2)
    x, poly_res = polynomial_fit(df, energy, ax3, n=5)
    x, log_normal_res = log_normal_power_law(df, energy, ax4)


    plt.tight_layout()
    #plt.savefig("figures/" + energy + "/" + energy + "_regression_plots.pdf", dpi=300)
    plt.show()


    # plot how gaussian residuals are
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    gaussian_residuals_check(x, linear_res, "Linear Regression", ax1)
    gaussian_residuals_check(x, pw_res, "Piecewise Regression", ax2)
    gaussian_residuals_check(x, poly_res, "Polynomial Regression", ax3)
    gaussian_residuals_check(x, log_normal_res, "Log Normal Power Law Regression", ax4)

    plt.suptitle("Distribution of Residuals for different fits, " + energy)

    plt.tight_layout()
    #plt.savefig("figures/" + energy + "/" + energy + "_residuals_distribution.pdf", dpi=300)
    plt.show()


    # plot scatter plots of residuals for each fit
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    residuals_scatter(x, linear_res, "Linear Regression", ax1)
    residuals_scatter(x, pw_res, "Piecewise Regression", ax2)
    residuals_scatter(x, poly_res, "Polynomial Regression", ax3)
    residuals_scatter(x, log_normal_res, "Log Normal Power Law Regression", ax4)

    plt.suptitle("Scatterplots of Residuals vs input, " + energy)

    plt.tight_layout()
    #plt.savefig("figures/" + energy + "/" + energy + "_residuals_vs_X.pdf", dpi=300)
    plt.show()


def mass_sliding_window(df, feature, window_num=10):

    # define DataFrame of interest
    df_log = np.log10(df[[feature, 'mf']])
    # define windows from window size in feature space
    window_grid = np.linspace(df_log[feature].min(), df_log[feature].max(), window_num)

    means = []
    stds = []

    # loop through windows
    for i in range(window_num - 1):

        # mask out values in the current window
        mask = (df_log[feature] >= window_grid[i]) & (df_log[feature] <= window_grid[i+1])

        sample = df_log['mf'][mask]

        # Calculate the means and standard deviations of the sample
        mean = np.mean(sample)
        std = np.std(sample)

        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    xs = window_grid[:-1] + 0.5 * (window_grid[1] - window_grid[0])

    ax1.set_title("Mean mass vs " + feature)
    ax1.set_xlabel(feature)
    ax1.set_ylabel(r'Mean $\log_{10}$ Mass ($M_\odot$)')
    ax1.scatter(df_log[feature], df_log['mf'], alpha=0.3, c='gray')
    ax1.plot(xs, means+stds, c='crimson', linestyle='--')
    ax1.plot(xs, means-stds, c='crimson', linestyle='--')
    ax1.plot(xs, means, c='r')
    

    ax2.set_title('Standard Deviation of mass vs ' + feature)
    ax2.set_xlabel(feature)
    ax2.set_ylabel(r'Std $\log_{10}$ Mass ($M_\odot$)')
    ax2.plot(xs, stds, c='r')

    plt.suptitle("Sliding Window of Means and Std of Mass vs " + feature)
    plt.tight_layout()
    #plt.savefig('figures/' + feature + '/' + feature + '_sliding_window.pdf', dpi=300)
    plt.show()



if __name__ == "__main__":

    residual_comparision('sp_kin')

    residual_comparision('sp_mag')

    mass_sliding_window(df, 'sp_kin')

    mass_sliding_window(df, 'sp_mag')


