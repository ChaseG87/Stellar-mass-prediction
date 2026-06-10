import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from seaborn import heatmap


# Load pre-stellar gas data
df = pd.read_parquet('accreted_gas_properties_pre_stellar.pq')

df['sp_kin'] = df['E_kin'] / df['M_tot']
df['sp_mag'] = df['E_mag'] / df['M_tot']


def get_residuals(df, feature):
    '''
    Computes the residuals of a linear regression model predicting total mass from a specified feature.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - feature: string name of the column in the DataFrame to be used as the predictor variable.
    
    Returns:
    - residuals: numpy array containing the residuals of the linear regression model.
    '''
    
    X = np.log10(df[[feature]])  # Predictor variable (log-transformed)
    y = np.log10(df['M_tot'])     # Target variable (log-transformed)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict values and compute residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    return residuals


def plot_residuals(df, feature):
    '''
    Plots the residuals of a linear regression model predicting total mass from a specified feature.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - feature: string name of the column in the DataFrame to be used as the predictor variable.
    '''
    
    residuals = get_residuals(df, feature)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log10(df[feature]), residuals, alpha=0.5)
    plt.xlabel(f'Log of {feature}')
    plt.ylabel('Residuals (log(M_tot) - predicted log(M_tot))')
    plt.title(f'Residual Plot for Linear Regression of log(M_tot) on log({feature})')
    plt.show()


def plot_residual_correlation(df, res_feature, feature):
    '''
    Plots the correlation between residuals of a linear regression model and other features.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - res_feature: string name of the column in the DataFrame to be used as the predictor variable for computing residuals.
    - feature: string name of the column in the DataFrame to be plotted against the residuals.
    '''
    
    residuals = get_residuals(df, res_feature)

    model = LinearRegression()

    model.fit(np.log10(df[feature].values.reshape(-1, 1)), residuals)
    r_squared = model.score(np.log10(df[feature].values.reshape(-1, 1)), residuals)

    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(df[feature]), residuals, alpha=0.5, label=feature)
    plt.xlabel('Log of Feature Value')
    plt.ylabel('Residuals (log(M_tot) - predicted log(M_tot))')
    plt.title(f'Residual Correlation Plot for Linear Regression of log(M_tot) on log({res_feature}) R²={r_squared}')
    plt.show()


def residual_regression_analysis(df, res_feature, features):
    '''
    Performs regression analysis on the residuals of a linear regression model and another feature.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - res_feature: string name of the column in the DataFrame to be used as the predictor variable for computing residuals.
    - feature: string name of the column in the DataFrame to be used as the predictor variable for regression analysis on residuals.
    
    Returns:
    - model: fitted regression model.
    - predictions: numpy array of shape (n_samples,) containing the predicted values of the residuals based on the specified feature.
    '''
    
    residuals = get_residuals(df, res_feature)
    
    X = np.log10(df[features])  # Predictor
    y = residuals                 # Target variable (residuals)
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # plot scatter of residuals vs predictions
    plt.scatter(y, predictions, s=10)

    # plot perfect prediction line
    vals = np.linspace(min(y), max(y), 100)
    plt.plot(vals, vals, linestyle='--', c='r')

    plt.xlabel(f'Actual Residuals of log(M_tot) on log({res_feature})')
    plt.ylabel('Predicted Residuals')
    R_squared = model.score(X, y)
    plt.title(f'Regression Analysis of Residuals R²: {R_squared:.2f}')
    plt.show()

    return model, predictions


def cross_correlation_scatter(df, features):
    '''
    Creates a pairplot of the specified features.
    '''

    sns.pairplot(np.log10(df[features]), kind='hist')

    plt.show()


def residual_heatmap(df):

    # allow snap to be accessed
    df = df.reset_index()

    df['kin_res'] = get_residuals(df, 'sp_kin')
    df['mag_res'] = get_residuals(df, 'sp_mag')

    chosen_log_labels = ['M_tot', 'T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                     'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std'] 
    

    df_log = df[['kin_res', 'mag_res', 'snap']]
    df_log[chosen_log_labels] = np.log10(df[chosen_log_labels])

    correlation_matrix = df_log.corr()
    plt.figure(figsize=(15, 12))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Residuals, Mass, and Intrinsic Properties')
    plt.savefig('residual_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    



#plot_residual_correlation(df, 'sp_kin', 'B_84')

#plot_residuals(df, 'sp_kin')

# Pairplot of the statistics of each intensic property
#cross_correlation_scatter(df, ['B', 'B_med', 'B_16', 'B_84'])
#cross_correlation_scatter(df, ['T', 'T_med', 'T_16', 'T_84'])
#cross_correlation_scatter(df, ['Ne', 'Ne_med', 'Ne_16', 'Ne_84'])
#cross_correlation_scatter(df, ['rho', 'rho_med', 'rho_16', 'rho_84'])

#cross_correlation_scatter(df, ['M_tot', 'T', 'B', 'Ne', 'rho'])
#cross_correlation_scatter(df, ['M_tot', 'T_med', 'B_med', 'Ne_med', 'rho_med'])
#cross_correlation_scatter(df, ['M_tot', 'T_16', 'B_16', 'Ne_16', 'rho_16'])
#cross_correlation_scatter(df, ['M_tot', 'T_84', 'B_84', 'Ne_84', 'rho_84'])

#cross_correlation_scatter(df, ['M_tot', 'T_std', 'B_std', 'Ne_std', 'rho_std'])

#plot_residual_correlation(df, 'sp_mag', 'M_tot')

#residual_regression_analysis(df, 'sp_kin', ['Ne_med', 'Ne_84', 'B'])

#cross_correlation_scatter(df, ['sp_kin', 'sp_mag', 'T', 'Ne', 'B'])

residual_heatmap(df)