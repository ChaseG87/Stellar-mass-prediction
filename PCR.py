import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

df = pd.read_parquet('accreted_gas_properties_pre_stellar.pq')

def perform_pca(df, n_components=3):
    '''
    Performs Principal Component Analysis (PCA) on the given DataFrame and plots the results.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - n_components: number of principal components to compute (default is 3).
    
    Returns:
    - pca: PCA object containing the fitted model.
    - transformed_data: numpy array of shape (n_samples, n_components) containing the PCA-transformed data.
    '''
    
    df['sp_kin'] = df['E_kin'] / df['M_tot']
    df['sp_mag'] = df['E_mag'] / df['M_tot']

    features = [ 'T', 'T_med', 'T_84', 'T_16', 'Ne', 'Ne_med', 'Ne_84', 'Ne_16',
            'B', 'B_med', 'B_84', 'B_16', 'rho', 'rho_med', 'rho_84', 'rho_16']  
    
    # Take the logarithm of the data to handle wide ranges and reduce skewness
    log_data = np.log(df[features])

    # Standardize the data (mean=0, variance=1)
    standardized_data = (log_data - log_data.mean()) / log_data.std()
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(standardized_data)
    
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()

    # Plot the cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    # 2D scatter plot of the first two principal components colored by total mass
    plt.figure(figsize=(8, 5))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=np.log(df['M_tot']), cmap='viridis', alpha=0.7)
    plt.colorbar(label='Total Mass (M_sun) [log scale]')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Pre-Stellar Gas Properties (features: T, Ne, B, rho)')
    plt.savefig('pca_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 3D scatter plot of the first three principal components colored by total mass
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    sc = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=np.log(df['M_tot']), cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Total Mass (M_sun) [log scale]')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title('3D PCA Visualization of Pre-Stellar Gas Properties (features: T, Ne, B, rho)')
    plt.show()
    
    return pca, transformed_data


def regression_analysis(transformed_data):
    '''
    Performs regression analysis using the PCA-transformed data to predict a target variable.
    Parameters:
    - transformed_data: numpy array of shape (n_samples, n_components) containing the PCA-transformed data.
    - target: numpy array of shape (n_samples,) containing the target variable to predict.
    Returns:
    - model: fitted regression model.
    - predictions: numpy array of shape (n_samples,) containing the predicted values.
    '''

    target = np.log(df['M_tot'])  # target variable (total mass of accreted gas)
    model = LinearRegression()
    model.fit(transformed_data, target)
    predictions = model.predict(transformed_data)
    plt.plot(target, predictions, 'o')
    plt.xlabel('Actual Total Mass (M_sun)')
    plt.ylabel('Predicted Total Mass (M_sun)')
    R_squared = model.score(transformed_data, target)
    plt.title(f'Regression Analysis using PCA-Transformed Data R²: {R_squared:.2f}')
    plt.show()


def blind_regression_analysis(df):

    features = [ 'Ne', 'Ne_med', 'Ne_84', 'Ne_16',
                 'B', 'B_med', 'B_84', 'B_16', 'rho', 'rho_med', 'rho_84', 'rho_16']
    
    # Take the logarithm of the data to handle wide ranges and reduce skewness
    log_data = np.log(df[features])

    target = np.log(df['M_tot'])  # target variable (total mass of accreted gas)
    model = LinearRegression()

    model.fit(log_data, target)
    predictions = model.predict(log_data)
    plt.plot(target, predictions, 'o')
    plt.xlabel('Actual Total Mass (M_sun)')
    plt.ylabel('Predicted Total Mass (M_sun)')
    R_squared = model.score(log_data, target)
    plt.title(f'Blind Regression Analysis R²: {R_squared:.2f}')
    plt.show()

def main():
    # Perform PCA on the DataFrame and plot the results
    pca, transformed_data = perform_pca(df, n_components=3)
    # Perform regression analysis using the PCA-transformed data
    regression_analysis(transformed_data)
    # Perform blind regression analysis using the original data
    blind_regression_analysis(df)


if __name__ == "__main__":
    main()