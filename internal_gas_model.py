import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import heatmap
from sklearn.linear_model import LinearRegression


# initialize dataframe
df = pd.read_parquet("accreted_gas_properties_pre_stellar.pq")
df = df.reset_index()


# define features to be investigated
log_features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
            'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std'] 

lin_features = ['snap']



class GreedyModelConstructor:

    def __init__(self, df):
        '''
        df: DataFrame with stellar data
        input_features: list of features that can be selected to predict the mass

        initializes the GreedyModelContructor
        '''
        self.df_input = df
        self.target = df['M_tot'].values
        self.predictors_log = []
        self.predictors_lin = []
        self.ordered_predictors = []


    def create_heatmap(self, df, log_features, lin_features=None):
        '''
        df: DataFrame containing data of features to be turned into a heatmap
        log_features: list of strings of features to be visualized in the heatmap in log space
        lin_features: list of strings of features to be visualized in the heatmap in linear space

        produces a heatmap of chosen features from a DataFrame df.
        '''
    

        df_log = np.log10(df[log_features])

        if lin_features != None:
            df_log[lin_features] = df[lin_features]

        correlation_matrix = df_log.corr()
        plt.figure(figsize=(15, 12))
        heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()


    def choose_first_input(self, log_features, lin_features=None):
        '''
        log_features: list of strings of features to be evaluated in log space to be evaluated
        lin_features: list of strings of features to be evaluated in lin space to be evalutated

        returns: tuple of type (str, str, float) with the first string being the type of fit and the second being the feature
        and the final being the r2 score of the fit
        '''

        X = self.df_input
        y = np.log10(self.target)

        # initialize highest r2
        best_r2 = 0

        # loop through log_features to find best option
        for feature in log_features:

            # define input
            input = np.log10(X[feature]).values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2


        # loop through lin_features to find best option
        for feature in lin_features:

            # define input
            input = X[feature].values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2


        # return a tuple with the information of the best feature 
        self.ordered_predictors.append(best_feature)   


        if best_feature in log_features:
            return ('log', best_feature, best_r2)
        
        if best_feature in lin_features:
            return ('lin', best_feature, best_r2)
        

    def clean_X_helper(self):
        feature_arrays = []

        X_log = np.log10(self.df_input[self.predictors_log]).values
        feature_arrays.append(X_log)

        X_lin = self.df_input[self.predictors_lin].values
        feature_arrays.append(X_lin)

        return np.hstack(feature_arrays)
        

    def get_residuals(self):
        '''
        Calculates the residuals of the predictor with respect to the target (mass)
        '''

        # clean X and y data to fit it
        X = self.clean_X_helper()
        y = self.target

        # fit current linear model 
        model = LinearRegression()
        model.fit(X, y)

        return y - model.predict(X)

    
    def additional_predictor(self, log_features, lin_features=None, heatmap=True):
        '''
        Chooses the next parameter to add to the predictor using the residuals
        of the current prediction model.

        log_features: list of strings of features to be evaluated in log space
        lin_features: list of strings of features to be evaluaed in lin space
        heatmap: bool if set to true plots a heatmap of the correlations

        returns: tuple of (str, str, float), (type of fit, chosen feature, r2 value)
        '''

        # extract residuals and define
        residuals = self.get_residuals()
        y = residuals

        # exclude current predictors from being added as the new predictor
        log_features = [i for i in log_features if i not in self.predictors_log]
        lin_features = [i for i in lin_features if i not in self.predictors_lin]


        # define X space
        X = self.df_input[log_features].copy()
        X[lin_features] = self.df_input[lin_features]
        X['residuals'] = residuals


        # initialize r2
        best_r2 = 0

        # create heatmap of options for new perdictor if heatmap is set to True
        if heatmap:
            features = log_features.copy()
            features.append('residuals')

            self.create_heatmap(X, log_features=features, lin_features=lin_features)


        # loop through log_features to find best option
        for feature in log_features:

            # define input
            input = np.log10(self.df_input[feature]).values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2
        

        # loop through lin_features to find best option
        for feature in lin_features:

            # define input
            input = X[feature].values.reshape(-1, 1)

            # fit linear regression model
            model = LinearRegression()
            model.fit(input, y)
            r2 = model.score(input, y)

            if r2 > best_r2:
                best_feature = feature
                best_r2 = r2

        self.ordered_predictors.append(best_feature)
            
        if best_feature in log_features:
            return ('log', best_feature, best_r2)
        
        if best_feature in lin_features:
            return ('lin', best_feature, best_r2)


    def construct_predictor(self, log_features, lin_features=None, method='defined', num_preds = 3, threshold=0.02):
        '''
        method: string defining how the predictor is build with two options;
        'defined' continues for a set number of predictors,
        'threshold' continues until the model stops improving r2 by a threshold.

        num_preds: integer that defines how many operations if defined is the method.

        threshold: float that determines the r2 increase requirement until the model stops.
        '''

        # initialize list to track r2s of each additional 
        r2s = []

        # make the first choice of the features
        feature_tuple = self.choose_first_input(log_features, lin_features=lin_features)
            
        if feature_tuple[0] == 'log':
            self.predictors_log.append(feature_tuple[1])

        if feature_tuple[0] =='lin':
            self.predictors_lin.append(feature_tuple[1])

        r2s.append(feature_tuple[2])


        # if the method is defined the addition to the model aquired num_preds predictors
        if method == 'defined':
            # loop through to add additional features based on residuals of the current model
            for _ in range(num_preds - 1):
                feature_tuple = self.additional_predictor(log_features, lin_features=lin_features)

                if feature_tuple[0] == 'log':
                    self.predictors_log.append(feature_tuple[1])

                if feature_tuple[0] =='lin':
                    self.predictors_lin.append(feature_tuple[1])

                r2s.append(feature_tuple[2])

        # if the method is threshold the addition to the model stops when the threshold is greater to the improvement to the model
        if method == 'threshold':
            r2_increase = 1

            while r2_increase > threshold:
                pass
        
    
    def predict(self, input, return_model= False):
        '''
        input: array with dimensions of the model used to predict masses
        return_model: bool that determines if the model will be returned

        returns: an array of predictions and the model if return_model -
        just array of predictions if return_model = False
        '''

        # create model with GreedyModelConstructor object's properties
        X = self.clean_X_helper()
        y = np.log10(self.target)

        # fit current linear model 
        model = LinearRegression()
        model.fit(X, y)

        if return_model:
            return model.predict(input), model

        return model.predict(input)
    

    def visualize(self, preds):
        '''
        Visualizes the prediction model
        '''

        # get X and y for scatter and prediction
        X = self.clean_X_helper()
        y = np.log10(self.target)


        pred, model = self.predict(X, return_model='True')

        r2 = model.score(X, y)


        # plot scatter of actual vs predicted
        plt.scatter(y, pred, alpha=0.5, s = 10)

        # plot perfect prediction line
        plt.plot(y, y, c='r')
        plt.title(f"Predicted vs Actual of Greedy Model $R^2 =$ {r2:.2f}, num_preds = {preds}")
        plt.xlabel('Ground Truth Mass')
        plt.ylabel('Predicted Mass')
        plt.savefig(f'figures/greedy_figures/greedy_pred{preds}.pdf', dpi=300)
        
        plt.show()


if __name__ == '__main__':

    # initialize predictor
    model = GreedyModelConstructor(df)

    num_pred = 10

    # select features
    model.construct_predictor(log_features=log_features, lin_features=lin_features, method='defined', num_preds=num_pred)

    # visualize model
    model.visualize(preds=num_pred)

    print(f'Model Trained logorithmically on: {model.predictors_log}\n Model Trained Linearly on: {model.predictors_lin}')

    print(f'\nPredictors in order of strength: {model.ordered_predictors}')

# need to code the threshold method
# need to find and resolve runtime error in log10



    
