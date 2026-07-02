import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from internal_gas_model import LassoAidedModelConstructor
from internal_gas_model import GreedyModelConstructor
import matplotlib.pyplot as plt
import argparse
import sys
from process_data import log_data

features = [ 'T', 'T_med', 'T_84', 'T_16', 'T_std', 'Ne', 'Ne_med', 'Ne_84', 'Ne_16', 'Ne_std',
            'B', 'B_med', 'B_84', 'B_16', 'B_std', 'rho', 'rho_med', 'rho_84', 'rho_16', 'rho_std', 'snap']  

df, df_test = log_data(features=features, train_val = True)

def perform_pca(df, three_d=False, n_components=21):
    '''
    Performs Principal Component Analysis (PCA) on the given DataFrame and plots the results.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - three_d: boolean indicating whether to perform 3D analysis. If False, performs 2D analysis (default is False).
    - n_components: number of principal components to compute (default is 3).
    
    Returns:
    - pca: PCA object containing the fitted model.
    - transformed_data: numpy array of shape (n_samples, n_components) containing the PCA-transformed data.
    '''
    
    # Take the logarithm of the data to handle wide ranges and reduce skewness
    log_data = df[features]

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
    plt.xticks(range(1, n_components + 1))
    plt.savefig('figures/PCA/components_ratios.pdf', dpi=300)
    plt.show()

    # Plot the cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, n_components + 1))
    plt.savefig('figures/PCA/cumulative_variance.pdf', dpi=300)
    plt.show()

    # 2D scatter plot of the first two principal components colored by total mass
    plt.figure(figsize=(8, 5))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=7, c=df['mf'], cmap='viridis', alpha=0.7)
    plt.colorbar(label=r'Total Mass ($M_\odot$) [log scale]')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(r'PCA of Pre-Stellar Gas Property Distributions (features: T, $n_e$, B, $\rho$)')
    plt.savefig('figures/PCA/pca_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    if three_d:
        # 3D scatter plot of the first three principal components colored by total mass
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        sc = ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], s=10, c=df['mf'], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label=r'Total Mass ($M_\odot$) [log scale]')
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

    target = df['mf']  # target variable (total mass of accreted gas)
    model = LinearRegression()
    model.fit(transformed_data, target)
    predictions = model.predict(transformed_data)
    plt.scatter(target, predictions, s = 10, alpha = 0.5)
    plt.xlabel(r'Actual Total Mass ($M_\odot$)')
    plt.ylabel(r'Predicted Total Mass ($M_\odot$)')
    R_squared = model.score(transformed_data, target)
    plt.title(f'PCA Regression Data R²: {R_squared:.2f}, Components = {len(transformed_data[0])}')
    plt.plot(target, target, c='r')
    #plt.savefig('figures/PCA/pca_8_components.pdf', dpi=300)
    plt.show()


def lasso_pca_regression(transformed_data, alpha, pca=None, test=False):
    # define components and df_pca

    components = ['comp' + str(i+1) for i in range(len(transformed_data[0]))]
    df_pca = pd.DataFrame(transformed_data, columns=components)
    clean_df = df.reset_index()
    df_pca['mf'] = clean_df['mf']


    if test:

        # Make sure PCA object is provided in test mode
        if pca == None:
            raise ValueError("Calling test requires pca object of train data.")

        test_transformed_data = standardize_fit_pca(pca, df_train=df[features], df_test=df_test[features])

        # define test dataframe
        df_test_pca = pd.DataFrame(test_transformed_data, columns=components)
        df_test_pca['mf'] = df_test['mf']

    # initialize model and fit
    model = LassoAidedModelConstructor(df_pca, alpha=alpha)
    model.train(components)

    if test:
        # visualize model results
        model.visualize(df=df_test_pca)
    else:
        model.visualize(df=df_pca)


def greedy_pca_regression(transformed_data, pca=None, test=False, n_preds=8):

    # define components and df_pca

    components = ['comp' + str(i+1) for i in range(len(transformed_data[0]))]
    df_pca = pd.DataFrame(transformed_data, columns=components)
    clean_df = df.reset_index()
    df_pca['mf'] = clean_df['mf']


    if test:

        # Make sure PCA object is provided in test mode
        if pca == None:
            raise ValueError("Calling test requires pca object of train data.")

        test_transformed_data = standardize_fit_pca(pca, df_train=df[features], df_test=df_test[features])

        # define test dataframe
        df_test_pca = pd.DataFrame(test_transformed_data, columns=components)
        df_test_pca['mf'] = df_test['mf']


    # initialize model and fit
    model = GreedyModelConstructor(df_pca)
    model.train(components, num_preds=n_preds, heatmap=True)

    if test:
        # visualize test model results
        model.visualize(df=df_test_pca)
    else:
        # visualize train model results
        model.visualize(df=df_pca)


def standardize_fit_pca(pca_model, df_train, df_test):
    

    log_train = df_train
    log_test = df_test

    mean = log_train.mean()
    std = log_train.std()

    # Standardize the train data wrt the train data
    standardized_test = (log_test - mean) / std

    transformed_test = pca_model.transform(standardized_test)

    return transformed_test



def blind_regression_analysis(df):

    
    # Take the logarithm of the data to handle wide ranges and reduce skewness
    log_data = df[features]

    target = df['mf']  # target variable (total mass of accreted gas)
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
    argparser = argparse.ArgumentParser(description='Perform PCA analysis and correlation heatmap on pre-stellar gas properties.')
    argparser.add_argument('--pca_3d', '-3d', action='store_true', help='Perform 3D PCA analysis and visualization.')
    argparser.add_argument('--blind_regression', '-br', action='store_true', help='Perform blind regression analysis.')
    argparser.add_argument('--n_components', '-n_comp', type=int, default=21)
    argparser.add_argument('--greedy', '-gr', action='store_true', help='Run greedy PCA model')
    argparser.add_argument('--lasso', '-l',  action='store_true', help='Run LASSO PCA model')
    args = argparser.parse_args()


    if args.pca_3d:
        pca, transformed_data = perform_pca(df, three_d=True, n_components=args.n_components)
    else:
        pca, transformed_data = perform_pca(df, three_d=False, n_components=args.n_components)

    if args.greedy:
        greedy_pca_regression(transformed_data, pca=pca, test=False)
    
    if args.lasso:
        lasso_pca_regression(transformed_data, pca=pca, alpha=0.0095, test=False)

    if args.blind_regression:
        blind_regression_analysis(df)

    #regression_analysis(transformed_data)


if __name__ == "__main__":
    main()