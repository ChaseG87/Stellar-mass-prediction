import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import heatmap
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import argparse
from process_data import log_train_val_data, artificial_data



# define features to be investigated
features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
            'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'snap'] 

# pre-process data
df, df_val = log_train_val_data(features=features)


class GreedyModelConstructor:

    def __init__(self, df):
        '''
        df: DataFrame with stellar data
        input_features: list of features that can be selected to predict the mass

        initializes the GreedyModelContructor
        '''
        self.df_input = df
        self.target = df['M_tot'].values
        self.ordered_predictors = []


    def create_heatmap(self, df, features):
        '''
        df: DataFrame containing data of features to be turned into a heatmap
        features: list of strings of features to be visualized in the heatmap


        produces a heatmap of chosen features from a DataFrame df.
        '''
    
        # create correlation matrix
        correlation_matrix = df[features].corr().iloc[-2:, :]
        plt.figure(figsize=(15, 3))
        heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()


    def choose_first_input(self, features):
        '''
        features: list of strings of features to be evaluated


        returns: tuple of type (str, str, float) with the first string being the type of fit and the second being the feature
        and the final being the r2 score of the fit
        '''

        X = self.df_input
        y = self.target

        # initialize highest r2
        best_r2 = 0


        # loop through lin_features to find best option
        for feature in features:

            # define input
            input = X[feature].values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2



        return (best_feature, best_r2)
        
        

    def get_residuals(self):
        '''
        Calculates the residuals of the predictor with respect to the target (mass)
        '''

        # clean X and y data to fit it
        X = self.df_input[self.ordered_predictors].values
        y = self.target

        # fit current linear model 
        model = LinearRegression()
        model.fit(X, y)

        return y - model.predict(X)

    
    def additional_predictor(self, features, heatmap=True):
        '''
        Chooses the next parameter to add to the predictor using the residuals
        of the current prediction model.

        features: list of strings of features to be evaluated
        heatmap: bool if set to true plots a heatmap of the correlations

        returns: tuple of (str, str, float), (type of fit, chosen feature, r2 value)
        '''

        # extract residuals and define
        residuals = self.get_residuals()
        y = residuals

        # exclude current predictors from being added as the new predictor
        features = [i for i in features if i not in self.ordered_predictors]


        # define X space
        X = self.df_input[features].copy()
        X['residuals'] = residuals
        X['M_tot'] = self.target


        # initialize r2
        best_r2 = 0

        # create heatmap of options for new perdictor if heatmap is set to True
        if heatmap:
            features = features.copy()
            features.append('residuals')
            features.append('M_tot')

            self.create_heatmap(X, features=features)

            # remove residuals and mass from features
            features.pop()
            features.pop()


        # loop through features to find best option
        for feature in features:

            # define input
            input = self.df_input[feature].values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2
            

        return (best_feature, best_r2)
        


    def train(self, features, heatmap=False, num_preds = 8):
        '''
        features: list of features on which the model will be trained on.

        num_preds: integer that defines how many operations of the greedy algorithm will be run.
        '''

        # initialize list to track r2s of each additional 
        r2s = []

        # make the first choice of the features
        feature_tuple = self.choose_first_input(features)
            
        self.ordered_predictors.append(feature_tuple[0])

        r2s.append(feature_tuple[1])



        # loop through to add additional features based on residuals of the current model
        for _ in range(num_preds - 1):
            feature_tuple = self.additional_predictor(features, heatmap=heatmap)

            self.ordered_predictors.append(feature_tuple[0])

            r2s.append(feature_tuple[1])
        
        print(f'Greedy features {self.ordered_predictors}')


        X = self.df_input[self.ordered_predictors].values
        y = self.target

        # fit current linear model 
        model = LinearRegression()
        model.fit(X, y)

        self.model = model
        
    
    def predict(self, input, return_model= False):
        '''
        input: array with dimensions of the model used to predict masses
        return_model: bool that determines if the model will be returned

        returns: an array of predictions and the model if return_model -
        just array of predictions if return_model = False
        '''

        # create model with GreedyModelConstructor object's properties
        X = self.df_input[self.ordered_predictors].values
        y = self.target

        # fit current linear model 
        model = LinearRegression()
        model.fit(X, y)

        if return_model:
            return model.predict(input), model

        return model.predict(input)
    

    def visualize(self, preds, df=None):
        '''
        Visualizes the prediction model
        '''

        # get X and y for scatter and prediction
        if type(df) != type(pd.DataFrame()):
            X = self.df_input[self.ordered_predictors].values
            y = self.target
        else:
            X = df[self.ordered_predictors].values
            y = df['M_tot']

        pred, model = self.predict(X, return_model='True')

        r2 = model.score(X, y)

        # plot scatter of actual vs predicted
        plt.scatter(y, pred, alpha=0.5, s = 10)

        # plot perfect prediction line
        plt.plot(y, y, c='r')
        plt.title(f"Predicted vs Actual of Greedy Model $R^2 =$ {r2:.2f}, num_preds = {preds}")
        plt.xlabel('Ground Truth Mass')
        plt.ylabel('Predicted Mass')
        if __name__ == "__main__" and type(df) == type(None):
            plt.savefig(f'figures/greedy_figures/greedy_pred{preds}.pdf', dpi=300)
        
        plt.show()


class LassoAidedModelConstructor:
    '''
    Constructs a LASSO model to choose the reduced features for predicting the mass stars given input features
    and generates a linear model of the selected features.
    '''

    def __init__(self, df, alpha=0.01):
        '''
        initializes the LassoModelConstuctor

        df: DataFrame with feature and mass data
        alpha: float - L1 normalization coefficient
        '''
        # initialize Lasso model
        self.model = Lasso(alpha=alpha)

        # save df and target
        self.df_input = df
        self.target = df['M_tot'].values
        self.key_features = []
    
    def train(self, features):
        '''
        Trains a linear regression with reduced features chosen by LASSO L1 regularization

        features: features modeled in log log space wrt mass
        '''

        # fit the Lasso model
        X = self.df_input[features].values
        self.model.fit(X, self.target)

        # select only features that survived the pruning
        self.key_features = [features[i] for i in range(len(self.model.coef_)) if abs(self.model.coef_[i]) > 0.00001]
        print(f'Lasso features {self.key_features}')

        # fit a linear regression to the pruned features
        X_fin = self.df_input[self.key_features].values
        self.model = LinearRegression()
        self.model.fit(X_fin, self.target)
    

    def visualize(self, df=None, title=None):
        '''
        Visualizes the prediction vs ground truth of the LASSO aided reduced linear model
        '''

        # score fitted model
        if type(df) == type(None):
            X = self.df_input[self.key_features].values
            y = self.target
        else:
            X = df[self.key_features].values
            y = df['M_tot']

        pred = self.model.predict(X)
        r2 = self.model.score(X, y)

        # plot fitted model against scatter
        plt.scatter(y, pred, alpha=0.5, s= 10)
        if title == None:
            plt.title(f'LASSO Aided Model Ground Truth vs Pred, $R^2 = {r2:.2f}$, features = {len(X[0])}')
        else:
            plt.title(title)
        plt.xlabel('Ground Truth (mass)')
        plt.ylabel('Predicted Mass')
        plt.plot(y, y, c='r')
        if __name__ == "__main__" and type(df) == None:
            plt.savefig('figures/lasso_tuned_8.pdf', dpi=300)
        plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform Greedy Linear Model based on Covariance with Residuals.')
    parser.add_argument('--greedy', "-gr", help='Runs Greedy Model with determined provided amount of features')
    parser.add_argument('--heatmaps', '-hm', action='store_true', help='Shows intermittent heatmaps if called')
    parser.add_argument('--lasso', '-l', action='store_true', help='Run the LASSO model')
    parser.add_argument('--alpha', '-a', default=0.01, type=float, help='Coefficient for normalization in the LASSO model')
    parser.add_argument('--unit_test', '-ut', action='store_true', help='Runs Greedy model and LASSO on artificial dataset')
    args = parser.parse_args()

    # initialize predictor
    model = GreedyModelConstructor(df)

    if args.greedy:
        
        num_pred = int(args.greedy)

        # select features

        model.train(features=features, heatmap=args.heatmaps, num_preds=num_pred)

        # visualize model
        model.visualize(preds=num_pred, df=df)


        print(f'Greedy predictors in order of strength: {model.ordered_predictors}')


    if args.lasso:
        # evaluate lasso model
        lasso = LassoAidedModelConstructor(df=df, alpha=args.alpha)

        lasso.train(features=features)

        # visualize model
        lasso.visualize(df=df_val)


    if args.unit_test:
        df = artificial_data()

        features = ['x' + str(i+1) for i in range(10)]

        num_pred = 8

        model = GreedyModelConstructor(df)


        model.train(features=features, heatmap=args.heatmaps, num_preds=num_pred)

        # visualize model
        model.visualize(preds=num_pred, df=df)

        # evaluate lasso model
        lasso = LassoAidedModelConstructor(df=df, alpha=0.55)

        lasso.train(features=features)

        # visualize model
        lasso.visualize()






    
