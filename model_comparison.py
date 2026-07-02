import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import sys
import argparse
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from process_data import n_snaps_extract, log_data
from internal_gas_model import GreedyModelConstructor, LassoAidedModelConstructor

# tasks:
# 1. compare RMSE performance for training and test data as a function of the number of predictors for each model
# 2. compare the coefficients of the features in the models vs their coefficient on their own

features = ['T', 'T_med', 'T_16', 'T_84', 'T_std_er', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std_er',
            'B', 'B_med', 'B_16', 'B_84', 'B_std_er', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std_er', 'snap'] 

# load clean train and validation data
df_train, df_val = log_data(features=features, train_val=True, normalize=True)

known_alphas = {0:0.01}
known_alphas_pca = {0:0.01}


def alpha_to_num_preds(df, n_preds, features, pca=False, pure=False):
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

        model = LassoAidedModelConstructor(df=df, alpha=prior_alphas[n_preds], pure=pure)
        model.train(features=features)

        return model
    

    if max(prior_alphas.keys()) < n_preds:
        c_alpha = prior_alphas[max(prior_alphas.keys())]
    else:
        c_alpha = 0.01

    # initialize model
    model = LassoAidedModelConstructor(df=df, alpha=c_alpha, pure=pure)

    # start while loop that iterates through alphas to find matching n_preds
    while len(model.key_features) != n_preds:

        # train model
        model.train(features=features)

        if len(model.key_features) == n_preds:

            # return the model if the apt value of alpha has been chosen
            return model

        elif len(model.key_features) > n_preds:
            prior_alphas[len(model.key_features)] = c_alpha
            c_alpha *= 1.05
        
        elif len(model.key_features) < n_preds:
            prior_alphas[len(model.key_features)] = c_alpha
            c_alpha *= 0.9
            
        
        model = LassoAidedModelConstructor(df=df, alpha=c_alpha, pure=pure)


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
    pca = PCA(n_components=21)
    transformed_train_data = pca.fit_transform(train[features])

    # get PCA validation data in the normalized space defined by the training data
    print(val.isna().sum())
    transformed_val_data = pca.transform(val[features])

    # generate component names
    components = ['comp' + str(i+1) for i in range(len(transformed_train_data[0]))]

    # PCA train DF creation
    train_df_pca = pd.DataFrame(transformed_train_data, columns=components)
    clean_df = train.reset_index()
    train_df_pca['mf'] = clean_df['mf']

    # PCA validation DF creation
    val_df_pca = pd.DataFrame(transformed_val_data, columns=components)
    clean_val_df = val.reset_index()
    val_df_pca['mf'] = clean_val_df['mf']

    # iterate through num_preds
    for i in range(preds):

        # BASE MODELS

        # base greedy model
        b_greedy = GreedyModelConstructor(df=df_train) 
        b_greedy.train(features=features, num_preds=i+1)

        # get base greedy train rmse
        train_b_greedy_preds = b_greedy.predict(train[b_greedy.ordered_predictors].values.reshape(-1, i+1))
        train_b_greedy_rmse = np.sqrt(np.mean((train['mf'].values - train_b_greedy_preds)**2))

        # append rmse base greedy train
        train_rmse_greedy.append(train_b_greedy_rmse)


        # get base greedy validation rmse
        val_b_greedy_preds = b_greedy.predict(val[b_greedy.ordered_predictors].values.reshape(-1, i+1))
        val_b_greedy_rmse = np.sqrt(np.mean((val['mf'].values - val_b_greedy_preds)**2))

        # append rmse base greedy val
        val_rmse_greedy.append(val_b_greedy_rmse)


        # base lasso model
        b_lasso = alpha_to_num_preds(df=df_train, features=features, n_preds=i+1, pure=True)

        # get base lasso train rmse
        train_b_lasso_preds = b_lasso.model.predict(train[features])
        train_b_lasso_rmse = np.sqrt(np.mean((train['mf'].values - train_b_lasso_preds)**2))

        # append rmse base lasso train
        train_rmse_lasso.append(train_b_lasso_rmse)


        # get base lasso validation rmse
        val_b_lasso_preds = b_lasso.model.predict(val[features])
        val_b_lasso_rmse = np.sqrt(np.mean((val['mf'].values - val_b_lasso_preds)**2))

        # append rmse base lasso val
        val_rmse_lasso.append(val_b_lasso_rmse)

        # PCA MODELS

        # PCA greedy model
        pca_greedy = GreedyModelConstructor(df=train_df_pca) 
        pca_greedy.train(features=components, num_preds=i+1)

        # get pca greedy train rmse
        train_pca_greedy_preds = pca_greedy.predict(train_df_pca[pca_greedy.ordered_predictors].values.reshape(-1, i+1))
        train_pca_greedy_rmse = np.sqrt(np.mean((train_df_pca['mf'].values - train_pca_greedy_preds)**2))

        # append rmse pca greedy train
        train_rmse_pca_greedy.append(train_pca_greedy_rmse)

        # get pca greedy val rmse
        val_pca_greedy_preds = pca_greedy.predict(val_df_pca[pca_greedy.ordered_predictors].values.reshape(-1, i+1))
        val_pca_greedy_rmse = np.sqrt(np.mean((val_df_pca['mf'].values - val_pca_greedy_preds)**2))

        # append rmse pca greedy train
        val_rmse_pca_greedy.append(val_pca_greedy_rmse)

        # pca lasso model
        pca_lasso = alpha_to_num_preds(df=train_df_pca, features=components, n_preds=i+1, pca=True, pure=True)

        # get pca lasso train rmse
        train_pca_lasso_preds = pca_lasso.model.predict(train_df_pca[components])
        train_pca_lasso_rmse = np.sqrt(np.mean((train_df_pca['mf'].values - train_pca_lasso_preds)**2))

        # append rmse pca lasso train
        train_rmse_pca_lasso.append(train_pca_lasso_rmse)

        # get pca lasso val rmse
        val_pca_lasso_preds = pca_lasso.model.predict(val_df_pca[components])
        val_pca_lasso_rmse = np.sqrt(np.mean((val_df_pca['mf'].values - val_pca_lasso_preds)**2))

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
    ax2.set_title('Train and Validation RMSE of Base LASSO Model')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('RMSE Loss')
    ax2.plot(xs, train_rmse_lasso, 'o', label='Train Loss', c='r', alpha=0.7)
    ax2.plot(xs, val_rmse_lasso, 'o',label='Validation Loss', c='b', alpha=0.7)
    ax2.plot(xs, train_rmse_lasso, c='r', alpha=0.7)
    ax2.plot(xs, val_rmse_lasso, c='b', alpha=0.7)
    ax2.legend()
    ax2.set_xticks(xs)

    # plot the pca greedy model loss
    ax3.set_title('Train and Validation RMSE of PCA Greedy Model')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('RMSE Loss')
    ax3.plot(xs, train_rmse_pca_greedy, 'o', label='Train Loss', c='r', alpha=0.7)
    ax3.plot(xs, val_rmse_pca_greedy, 'o',label='Validation Loss', c='b', alpha=0.7)
    ax3.plot(xs, train_rmse_pca_greedy, c='r', alpha=0.7)
    ax3.plot(xs, val_rmse_pca_greedy, c='b', alpha=0.7)
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
    plt.savefig('figures/model_RMSE_comparison.pdf', dpi=300)
    plt.show()


def coefficient_comparision(train, preds, las=False, las_cv=False):
    '''
    Generates a dot plot of the coefficients chosen by the model compared to their univariate coefficients.
    inputs:
        -train: DataFrame of training data
        -las: bool - determines whether a greedy or LASSO model will be run
    returns:
        None
    '''

    lone_coefs = []

    # normalize input features data
    z_df = (train[features] - np.mean(train[features])) / np.std(train[features], axis=0)
    z_df['mf'] = train['mf']

    # define figsize for formatting
    plt.figure(figsize=(12, 8))


    # retrieve coefficents in greedy case
    if not las and (not las_cv):
        greedy = GreedyModelConstructor(df=z_df)
        greedy.train(features=features, num_preds=preds)

        coupled_coefs = greedy.model.coef_
        tick_features = greedy.ordered_predictors

        plt.title('Greedy model Coupled vs Univariate Coefficents (standardized)')

        for feature in greedy.ordered_predictors:

            linear = LinearRegression()
            linear.fit(z_df[feature].values.reshape(-1, 1), z_df['mf'].values)
            lone_coefs.append(linear.coef_)

    
    # retrieve coefficients in lasso case
    if las:
        lasso = alpha_to_num_preds(df=z_df, features=features, n_preds=preds)
        lasso_pure = alpha_to_num_preds(df=z_df, features=features, n_preds=preds, pure=True)

        coupled_coefs = lasso.model.coef_
        coupled_pure = lasso_pure.model.sparse_coef_.data
        print(coupled_pure)
        tick_features = lasso.key_features

        plt.title('LASSO models vs Univariate Coefficents (standardized)')     

        for feature in lasso.key_features:

            linear = LinearRegression()
            linear.fit(z_df[feature].values.reshape(-1, 1), z_df['mf'].values)
            lone_coefs.append(linear.coef_)

    elif las_cv:
        # defining and training lasso with cross validation
        lasso_cv = LassoCV(cv=5).fit(z_df[features].values.reshape(-1, len(features)), z_df['mf'])
        #lasso_cv = ElasticNetCV(l1_ratio=0.5, cv=5).fit(z_df[features].values.reshape(-1, len(features)), z_df['mf'])
        #lasso_cv = RidgeCV(alphas=np.logspace(0, 6, 13), cv=5).fit(z_df[features].values.reshape(-1, len(features)), z_df['mf'])

        # pulling coefficients
        coupled_pure = lasso_cv.coef_
        tick_features = [features[i] for i in range(len(features)) if coupled_pure[i]]

        # extracting only non-zero coefficients
        coupled_pure = [coupled_pure[i] for i in range(len(features)) if coupled_pure[i]]

        # initialize and train post lasso
        post_lasso = LinearRegression().fit(z_df[tick_features].values.reshape(-1, len(tick_features)), z_df['mf'])
        coupled_coefs = post_lasso.coef_


        for feature in tick_features:

            linear = LinearRegression()
            linear.fit(z_df[feature].values.reshape(-1, 1), z_df['mf'].values)
            lone_coefs.append(linear.coef_)

        plt.title('LASSO-CV vs Post-LASSO vs Univariate Coefficients (standardized)')

        # define preds by the number of features lasso cv selected
        preds = len(tick_features)

    # define x space
    xs = range(1, preds+1)


    # Loop through the range and fill every alternating block
    for i in range(0, preds):
        if i % 2 == 0:
            # facecolor handles the gray tone, alpha controls the transparency
            plt.axvspan(i + 0.5, i + 1.5, facecolor='lightgrey', alpha=0.4, zorder=1)

    # plot scatter of coefficients
    plt.scatter(xs, lone_coefs, label='Univariate Coefficient', alpha=0.5, c='b')
    plt.scatter(xs, coupled_coefs, label='Post-LASSO Coefficient', alpha=0.5, c='r')
    plt.scatter(xs, coupled_pure, label='LASSO Coefficient', alpha=0.5, c='g')
    plt.legend()
    plt.xticks(xs, tick_features)
    plt.tight_layout()
    #plt.savefig('figures/lasso_coefficient_comp.pdf', dpi=300)
    plt.show()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmse', action='store_true', help='Generates a visualization of the RMSE error of each model vs number of predictors')
    parser.add_argument('--greedy_coefs', '-g_cfs', default=0, type=int, help='Generates a visualization comparing the univariate and coupled coefficients of the greedy model\'s features.')
    parser.add_argument('--lasso_coefs', '-l_cfs', default=0, type=int, help='Generates a visualization comparing the univariate and coupled coefficients of the LASSO model\'s features.')
    parser.add_argument('--lasso_cv_coefs', '-lcv_cfs', action='store_true', help='Generates a visualization comparing the univariate and coupled coefficients of the LASSO-CV model\'s features.')
    args = parser.parse_args()
    
    if args.rmse:
        rmse_comparison(train=df_train, val=df_val, preds=21)

    if args.greedy_coefs:
        coefficient_comparision(train=df_train, preds=args.greedy_coefs, las=False)
    
    if args.lasso_coefs:
        coefficient_comparision(train=df_train, preds=args.lasso_coefs, las=True)

    if args.lasso_cv_coefs:
        coefficient_comparision(train=df_train, preds=1, las_cv=True)
    
    if len(sys.argv) < 2:
        raise ValueError('Must call a flag - run --help to see options')


if __name__ == '__main__':

    main()


    