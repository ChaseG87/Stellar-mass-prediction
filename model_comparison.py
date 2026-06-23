import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import sys
import argparse
from process_data import n_snaps_extract, log_train_val_data
from internal_gas_model import GreedyModelConstructor, LassoAidedModelConstructor
from PCR import standardize_fit_pca

# tasks:
# 1. compare RMSE performance for training and test data as a function of the number of predictors for each model
# 2. compare the coefficients of the features in the models vs their coefficient on their own

features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
            'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'snap'] 

# load clean train and validation data
df_train, df_val = log_train_val_data(features=features)

known_alphas = {}
known_alphas_pca = {}


def alpha_to_num_preds(df, n_preds, features, pca=False):
    '''
    Finds the value of alpha in lasso model which delivers a chosen n_preds for comparision to greedy model
    using an adjustment for alpha when under/over shooting.
    inputs:
        -df: dataframe with training data for the LASSO model
        -n_preds: desired number of predictions
    returns: LassoAidedModelConstructor object
    '''

    # define known alpha space
    prior_alphas = known_alphas_pca if pca else known_alphas

    # If we have seen an alpha value producing n_preds before, use it instead of searching
    if n_preds in prior_alphas:

        model = LassoAidedModelConstructor(df=df, alpha=prior_alphas[n_preds])
        model.train(features=features)

        return model
    c_alpha = 0.01

    # initialize model
    model = LassoAidedModelConstructor(df=df, alpha=c_alpha)

    # start while loop that iterates through alphas to find matching n_preds
    while len(model.key_features) != n_preds:

        # train model
        model.train(features=features)

        if len(model.key_features) == n_preds:

            # return the model if the apt value of alpha has been chosen
            return model

        elif len(model.key_features) > n_preds:
            prior_alphas[len(model.key_features)] = c_alpha
            c_alpha *= 1.15
        
        elif len(model.key_features) < n_preds:
            prior_alphas[len(model.key_features)] = c_alpha
            c_alpha *= 0.8
            
        
        model = LassoAidedModelConstructor(df=df, alpha=c_alpha)


def rmse_comparison(train, val, preds):
    '''
    Visualizes the train and validation RMSE losses for each of the four models: Greedy, LASSO, PCA Greedy, and PCA LASSO.
    inputs:
        -train: DataFrame containing the stellar gas data to be trained on
        -val: DataFrame containing the stellar gas data to be used as the validation set

    returns:
        None
    '''
    
    train_rmse_greedy, val_rmse_greedy, train_rmse_lasso, val_rmse_lasso = [], [], [], []
    train_rmse_pca_greedy, val_rmse_pca_greedy, train_rmse_pca_lasso, val_rmse_pca_lasso = [], [], [], []    

    # Perform PCA
    standardized_data = (train[features] - train[features].mean()) / train[features].std()

    pca = PCA(n_components=21)
    transformed_train_data = pca.fit_transform(standardized_data)

    # get PCA validation data in the normalized space defined by the training data
    transformed_val_data = standardize_fit_pca(pca_model=pca, df_train=train[features], df_test=val[features])

    # generate component names
    components = ['comp' + str(i+1) for i in range(len(transformed_train_data[0]))]

    # PCA train DF creation
    train_df_pca = pd.DataFrame(transformed_train_data, columns=components)
    clean_df = train.reset_index()
    train_df_pca['M_tot'] = clean_df['M_tot']

    # PCA validation DF creation
    val_df_pca = pd.DataFrame(transformed_val_data, columns=components)
    clean_val_df = val.reset_index()
    val_df_pca['M_tot'] = clean_val_df['M_tot']

    # iterate through num_preds
    for i in range(preds):

        # BASE MODELS

        # base greedy model
        b_greedy = GreedyModelConstructor(df=df_train) 
        b_greedy.train(features=features, num_preds=i+1)

        # get base greedy train rmse
        train_b_greedy_preds = b_greedy.predict(train[b_greedy.ordered_predictors].values.reshape(-1, i+1))
        train_b_greedy_rmse = np.sqrt(np.mean((train['M_tot'].values - train_b_greedy_preds)**2))

        # append rmse base greedy train
        train_rmse_greedy.append(train_b_greedy_rmse)


        # get base greedy validation rmse
        val_b_greedy_preds = b_greedy.predict(val[b_greedy.ordered_predictors].values.reshape(-1, i+1))
        val_b_greedy_rmse = np.sqrt(np.mean((val['M_tot'].values - val_b_greedy_preds)**2))

        # append rmse base greedy val
        val_rmse_greedy.append(val_b_greedy_rmse)


        # base lasso model
        b_lasso = alpha_to_num_preds(df=df_train, features=features, n_preds=i+1)

        # get base lasso train rmse
        train_b_lasso_preds = b_lasso.model.predict(train[b_lasso.key_features].values.reshape(-1, i+1))
        train_b_lasso_rmse = np.sqrt(np.mean((train['M_tot'].values - train_b_lasso_preds)**2))

        # append rmse base lasso train
        train_rmse_lasso.append(train_b_lasso_rmse)


        # get base lasso validation rmse
        val_b_lasso_preds = b_lasso.model.predict(val[b_lasso.key_features].values.reshape(-1, i+1))
        val_b_lasso_rmse = np.sqrt(np.mean((val['M_tot'].values - val_b_lasso_preds)**2))

        # append rmse base lasso val
        val_rmse_lasso.append(val_b_lasso_rmse)

        # PCA MODELS

        # PCA greedy model
        pca_greedy = GreedyModelConstructor(df=train_df_pca) 
        pca_greedy.train(features=components, num_preds=i+1)

        # get pca greedy train rmse
        train_pca_greedy_preds = pca_greedy.predict(train_df_pca[pca_greedy.ordered_predictors].values.reshape(-1, i+1))
        train_pca_greedy_rmse = np.sqrt(np.mean((train_df_pca['M_tot'].values - train_pca_greedy_preds)**2))

        # append rmse pca greedy train
        train_rmse_pca_greedy.append(train_pca_greedy_rmse)

        # get pca greedy val rmse
        val_pca_greedy_preds = pca_greedy.predict(val_df_pca[pca_greedy.ordered_predictors].values.reshape(-1, i+1))
        val_pca_greedy_rmse = np.sqrt(np.mean((val_df_pca['M_tot'].values - val_pca_greedy_preds)**2))

        # append rmse pca greedy train
        val_rmse_pca_greedy.append(val_pca_greedy_rmse)

        # pca lasso model
        pca_lasso = alpha_to_num_preds(df=train_df_pca, features=components, n_preds=i+1, pca=True)

        # get pca lasso train rmse
        train_pca_lasso_preds = pca_lasso.model.predict(train_df_pca[pca_lasso.key_features].values.reshape(-1, i+1))
        train_pca_lasso_rmse = np.sqrt(np.mean((train_df_pca['M_tot'].values - train_pca_lasso_preds)**2))

        # append rmse pca lasso train
        train_rmse_pca_lasso.append(train_pca_lasso_rmse)

        # get pca lasso val rmse
        val_pca_lasso_preds = pca_lasso.model.predict(val_df_pca[pca_lasso.key_features].values.reshape(-1, i+1))
        val_pca_lasso_rmse = np.sqrt(np.mean((val_df_pca['M_tot'].values - val_pca_lasso_preds)**2))

        # append rmse pca lasso val
        val_rmse_pca_lasso.append(val_pca_lasso_rmse)


    # plotting

    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # define the x space
    xs = range(1, preds + 1)

    # plot the base greedy model loss
    ax1.set_title('Train and Validation RMSE of Base Greedy Model')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('RMSE Loss')
    ax1.plot(xs, train_rmse_greedy, 'o', label='Train Loss', c='r', alpha=0.7)
    ax1.plot(xs, val_rmse_greedy, 'o', label='Validation Loss', c='b', alpha=0.7)
    ax1.plot(xs, train_rmse_greedy, c='r', alpha=0.7)
    ax1.plot(xs, val_rmse_greedy, c='b', alpha=0.7)
    ax1.legend()
    ax1.set_xticks(xs)

    # plot the base lasso model loss
    ax2.set_title('Train and Validation RMSE of Base LASSO-Aided Model')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('RMSE Loss')
    ax2.plot(xs, train_rmse_lasso, 'o', label='Train Loss', c='r', alpha=0.7)
    ax2.plot(xs, val_rmse_lasso, 'o',label='Validation Loss', c='b', alpha=0.7)
    ax2.plot(xs, train_rmse_lasso, c='r', alpha=0.7)
    ax2.plot(xs, val_rmse_lasso, c='b', alpha=0.7)
    ax2.legend()
    ax2.set_xticks(xs)

    # plot the pca greedy model loss
    ax3.set_title('Train and Validation RMSE of PCA LASSO-Aided Model')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('RMSE Loss')
    ax3.plot(xs, train_rmse_pca_lasso, 'o', label='Train Loss', c='r', alpha=0.7)
    ax3.plot(xs, val_rmse_pca_lasso, 'o',label='Validation Loss', c='b', alpha=0.7)
    ax3.plot(xs, train_rmse_pca_lasso, c='r', alpha=0.7)
    ax3.plot(xs, val_rmse_pca_lasso, c='b', alpha=0.7)
    ax3.legend()
    ax3.set_xticks(xs)

    ax4.set_title('Train and Validation RMSE of PCA LASSO-Aided Model')
    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('RMSE Loss')
    ax4.plot(xs, train_rmse_pca_lasso, 'o', label='Train Loss', c='r', alpha=0.7)
    ax4.plot(xs, val_rmse_pca_lasso, 'o',label='Validation Loss', c='b', alpha=0.7)
    ax4.plot(xs, train_rmse_pca_lasso, c='r', alpha=0.7)
    ax4.plot(xs, val_rmse_pca_lasso, c='b', alpha=0.7)
    ax4.legend()
    ax4.set_xticks(xs)

    plt.tight_layout()
    plt.savefig('figures/model_RMSE_comparision.pdf', dpi=300)
    plt.show()



rmse_comparison(train=df_train, val=df_val, preds=21)


    