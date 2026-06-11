import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pwlf
import numpy as np
import sys
import argparse
from scipy.optimize import minimize

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
df = pd.read_parquet('accreted_gas_properties_pre_stellar.pq')

# add distance from center metric to the DataFrame
df['sp_kin'] = df['E_kin']/df['M_tot']
df['sp_mag'] = df['E_mag']/df['M_tot']
df['sp_int'] = df['E_int']/df['M_tot'] # poorly correlated
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


def linear_fit(df, feature):
    '''
    Fits a linear model of a given feature to the total mass M_tot
    '''

    # fit model
    model = LinearRegression()
    X = np.log10(df[[feature]]).values.reshape(-1, 1)
    y = np.log10(df['M_tot']).values
    model.fit(X, y)

    # score model
    r2 = model.score(X, y)

    #plot scatter
    plt.scatter(X, y, s=10, alpha=0.7)

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = model.predict(xs.reshape(-1, 1))
    plt.plot(xs, ys, c='r', linestyle='--', label=f'$r^2$ = {r2:.2f}')

    # labels and titles
    plt.legend()
    plt.title(f'Linear fit of {feature} vs M_tot')
    plt.xlabel(f'{feature}')
    plt.ylabel(r'M_tot ($M_\odot$)')

    plt.show()

    residuals = model.predict(X) - y

    return X, residuals


def piecewise_fit(df, feature, n=2):
    '''
    Fits a piecewise power-law of a given feature to M_tot with n breakpoints
    '''

    X = np.log10(df[feature]).values
    y = np.log10(df['M_tot']).values

    # initialize the model object
    model = pwlf.PiecewiseLinFit(X, y)

    # define the number of breakpoints 
    model.fit(n)

    #plot scatter
    plt.scatter(X, y, s=10, alpha=0.7)
    

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = model.predict(xs)
    plt.plot(xs, ys, c='r', linestyle='--')

    #labels and titles
    plt.title(f'Piecewise fit, n={n}, {feature} vs M_tot')
    plt.xlabel(f'{feature}')
    plt.ylabel(r'M_tot ($M_\odot$)')

    plt.show()

    residuals = model.predict(X) - y

    return X, residuals


def polynomial_fit(df, feature, n=2):
    '''
    Fits a polynomial model of degree n for given feature to the total mass M_tot
    '''

    # fit model
    poly_model = make_pipeline(PolynomialFeatures(degree=n), LinearRegression())
    X = np.log10(df[[feature]]).values.reshape(-1, 1)
    y = np.log10(df['M_tot']).values
    poly_model.fit(X, y)

    # score model
    r2 = poly_model.score(X, y)

    #plot scatter
    plt.scatter(X, y, s=10, alpha=0.7)

    # plot model fit
    xs = np.linspace(min(np.log10(df[feature])), max(np.log10(df[feature])), 50)
    ys = poly_model.predict(xs.reshape(-1, 1))
    plt.plot(xs, ys, c='r', linestyle='--', label=f'$r^2$ = {r2:.2f}')

    # labels and titles
    plt.legend()
    plt.title(f'Linear fit of {feature} vs M_tot')
    plt.xlabel(f'{feature}')
    plt.ylabel(r'M_tot ($M_\odot$)')

    plt.show()

    residuals = poly_model.predict(X) - y

    return X, residuals


def log_normal_power_law(df, feature):
    # raw values
    x_raw = df[feature].values
    y_raw = df['M_tot'].values

    # Initialize and fit the mixture profile
    fitter = LogNormalPowerLawFitter()
    fitter.fit(x_raw, y_raw)

    # generate Evaluation line for plotting
    x_plot = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 200)
    y_plot = fitter.predict(x_plot)

    # render the Figure in Log-Log Space
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log10(x_raw), np.log10(y_raw), s=10, alpha=0.7)
    plt.plot(np.log10(x_plot), np.log10(y_plot), color='r', linestyle='--', 
             label=f"Spliced Model (Slope = {fitter.m:.2f})")

    # plot vertical visual marker indicating exactly where the smooth splice happened
    plt.axvline(fitter.Xb, color="black", linestyle=":", alpha=0.7, 
                label=f"Transition Window (10^{fitter.Xb:.1f})")

    # labels and title
    plt.xlabel(r"$\log_{10}$ " + feature)
    plt.ylabel(r"$\log_{10}$ M_tot ($M_\odot$)")
    plt.title("Power-Law Mixture Fit")
    plt.legend()
    plt.savefig(f'figures/log_normal_power_law_{feature}_fit.pdf', dpi=300)
    plt.show()

    residuals = np.log10(fitter.predict(x_raw)) - np.log10(y_raw)

    return np.log10(x_raw), residuals


def chi_squared_comparison(df, feature):
    # can only compare linear models with linear parameters (no log_normal_power_law)


    pass



linear_fit(df, 'sp_kin')

piecewise_fit(df, 'sp_kin', n=2)

polynomial_fit(df, 'sp_kin', n=2)

log_normal_power_law(df, 'sp_kin')


