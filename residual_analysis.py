import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from seaborn import heatmap
from data_analysis_v2 import linear_fit, piecewise_fit, polynomial_fit, log_normal_power_law
from process_data import merge_snap_table, add_standard_error


# Load pre-stellar gas data
df = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq').reset_index()
df = df.drop(columns=['M_tot'])
df = merge_snap_table(df)
df = add_standard_error(df)

df['sp_kin'] = df['E_kin'] / df['mf']
df['sp_mag'] = df['E_mag'] / df['mf']
df['sp_int'] = df['E_int'] / df['mf']


def get_residuals(df, feature, model):
    '''
    Computes the residuals of a linear regression model predicting total mass from a specified feature.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - feature: string name of the column in the DataFrame to be used as the predictor variable.
    - model: string name of the model to be evaluated.
    
    Returns:
    - residuals: numpy array containing the residuals of the linear regression model.
    '''
    
    # dummy axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)

    if model == 'linear':
        x, residuals = linear_fit(df, feature, ax)
        plt.close()
        return residuals
    if model == 'piecewise':
        x, residuals = piecewise_fit(df, feature, ax)
        plt.close()
        return residuals
    if model == 'polynomial':
        x, residuals = polynomial_fit(df, feature, ax)
        plt.close()
        return residuals
    if model == 'lognormpower':
        x, residuals = log_normal_power_law(df, feature,ax)
        plt.close()
        return residuals
    
    


def plot_residuals(df, feature, model='linear'):
    '''
    Plots the residuals of a linear regression model predicting total mass from a specified feature.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - feature: string name of the column in the DataFrame to be used as the predictor variable.
    '''
    
    residuals = get_residuals(df, feature, model)
    
    plt.figure(figsize=(8, 5))
    plt.scatter(np.log10(df[feature]), residuals, alpha=0.5)
    plt.xlabel(f'Log of {feature}')
    plt.ylabel('Residuals (log(mf) - predicted log(mf))')
    plt.title(f'Residual Plot for Linear Regression of log(mf) on log({feature}), model = ' + model)
    plt.show()


def plot_residual_correlation(df, res_feature, feature, model='linear'):
    '''
    Plots the correlation between residuals of a regression model and other features.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - res_feature: string name of the column in the DataFrame to be used as the predictor variable for computing residuals.
    - feature: string name of the column in the DataFrame to be plotted against the residuals.
    - model: string defines the model being compared to, linear by default.
    '''
    
    residuals = get_residuals(df, res_feature, model)
    model_str = model

    model = LinearRegression()

    model.fit(np.log10(df[feature].values.reshape(-1, 1)), residuals)
    r_squared = model.score(np.log10(df[feature].values.reshape(-1, 1)), residuals)

    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(df[feature]), residuals, alpha=0.5, label=feature)
    plt.xlabel('Log of Feature Value, ' + feature)
    plt.ylabel('Residuals (log(mf) - predicted log(mf))')
    plt.title(f'Residual Correlation Plot for Regression of log(mf) on log({res_feature}) r²={r_squared:.2f}, model = ' + model_str)
    plt.show()


def residual_regression_analysis(df, res_feature, features, model):
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
    
    residuals = get_residuals(df, res_feature, model='linear')
    
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

    plt.xlabel(f'Actual Residuals of log(mf) on log({res_feature})')
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
    plt.suptitle('Standard Error vs Mass Pairplot')
    plt.tight_layout()
    #plt.savefig('figures/mass_vs_std_er.pdf', dpi=300)

    plt.show()


def residual_heatmap(df, model):

    # allow snap to be accessed
    df = df.reset_index()

    df['kin_res'] = get_residuals(df, 'sp_kin', model)
    df['mag_res'] = get_residuals(df, 'sp_mag', model)

    chosen_log_labels = ['mf', 'T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                     'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std','snap'] 
    

    df_log = df[['kin_res', 'mag_res']]
    df_log[chosen_log_labels] = np.log10(df[chosen_log_labels])

    correlation_matrix = df_log.corr()
    plt.figure(figsize=(15, 12))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Residuals, Mass, and Intrinsic Properties, model = ' + model)
    plt.savefig('figures/residual_correlation_heatmaps/' + model + '_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    


def rho_coefficients():



    model_rho = LinearRegression().fit(np.log10(df['rho'].values.reshape(-1, 1)), np.log10(df['mf'].values))
    model_rho_med = LinearRegression().fit(np.log10(df['rho_med'].values.reshape(-1, 1)), np.log10(df['mf'].values))
    model_rho_16 = LinearRegression().fit(np.log10(df['rho_16'].values.reshape(-1, 1)), np.log10(df['mf'].values))
    model_rho_84 = LinearRegression().fit(np.log10(df['rho_84'].values.reshape(-1, 1)), np.log10(df['mf'].values))
    model_r_eff = LinearRegression().fit(np.log10(df['R_eff'].values.reshape(-1, 1)), np.log10(df['mf'].values))
    rho_geometry = df['mf'] / (df['R_eff']**3)
    model_rho_geo = LinearRegression().fit(np.log10(rho_geometry.values.reshape(-1, 1)), np.log10(df['mf'].values))


    print(f'Rho Coefficient: {model_rho.coef_[0]:.4f}, R²: {model_rho.score(np.log10(df["rho"].values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    print(f'Rho Median Coefficient: {model_rho_med.coef_[0]:.4f}, R²: {model_rho_med.score(np.log10(df["rho_med"].values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    print(f'Rho 16th Percentile Coefficient: {model_rho_16.coef_[0]:.4f}, R²: {model_rho_16.score(np.log10(df["rho_16"].values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    print(f'Rho 84th Percentile Coefficient: {model_rho_84.coef_[0]:.4f}, R²: {model_rho_84.score(np.log10(df["rho_84"].values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    print(f'R_eff Coefficient: {model_r_eff.coef_[0]:.4f}, R²: {model_r_eff.score(np.log10(df["R_eff"].values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    print(f'Rho Geometry Coefficient: {model_rho_geo.coef_[0]:.4f}, R²: {model_rho_geo.score(np.log10(rho_geometry.values.reshape(-1, 1)), np.log10(df["mf"].values)):.4f}')
    
#rho_coefficients()



#plot_residual_correlation(df, 'sp_kin', 'B_84')

#plot_residuals(df, 'sp_kin')

# Pairplot of the statistics of each internal property in log space
#cross_correlation_scatter(df, ['B', 'B_med', 'B_16', 'B_84'])
#cross_correlation_scatter(df, ['T', 'T_med', 'T_16', 'T_84'])
#cross_correlation_scatter(df, ['Ne', 'Ne_med', 'Ne_16', 'Ne_84'])
#cross_correlation_scatter(df, ['mf','true_bulk_density', 'rho_med', 'rho_16', 'rho_84', 'R_eff'])
#df = df.reset_index()
#cross_correlation_scatter(df, ['mf', 'sp_kin', 'sig_3D', 'T', 'sp_int'])
#cross_correlation_scatter(df, ['mf', 'T_med', 'B_med', 'Ne_med', 'rho_med'])
#cross_correlation_scatter(df, ['mf', 'T_16', 'B_16', 'Ne_16', 'rho_16'])
#cross_correlation_scatter(df, ['mf', 'T_84', 'B_84', 'Ne_84', 'rho_84'])

#cross_correlation_scatter(df, ['mf', 'T_med', 'Ne_std', 'rho_med'])
#cross_correlation_scatter(df, ['mf', 'T_std', 'rho_std', 'B_std', 'Ne_std'])

cross_correlation_scatter(df, ['mf', 'T_std_er', 'B_std_er', 'Ne_std_er', 'rho_std_er'])


#, 'rho', 'Ne_std', 'rho_std'
#plot_residual_correlation(df, 'sp_mag', 'mf')

#residual_regression_analysis(df, 'sp_kin', ['Ne_med', 'Ne_84', 'B'])

#cross_correlation_scatter(df, ['sp_kin', 'sp_mag', 'T', 'Ne', 'B'])

#residual_heatmap(df, model='linear')
#residual_heatmap(df, model='polynomial')
#residual_heatmap(df, model='piecewise')
#residual_heatmap(df, model='lognormpower')

#plot_residual_correlation(df, 'sp_kin', 'mf', model='piecewise')