# Stellar-mass-prediction
This is a working repository analyzing relationship between pre-stellar gas properties and the final mass of stars using STARFORGE simulations of GMCs.

## Files and Usage

#### data_analysis_v1.py
Python file containing initial investigations of the correlations in the dataset primarily using heatmaps and scatter plots with power-law fits in log-log space.

#### data_analysis_v2.py
Python file containing code used to generate the different fitting methods (Linear, Piecewise-Linear, Polynomial, Log-Normal/Power-law) of the specific kinetic and specific
magnetic energy. Created the plots in **figures/sp_kin** and **figures/sp_mag**.

#### data_analysis_v3.py
Python file containing code used to perform data analysis on time series data at different snapshots. Used to produce the figures in **figures/three_snap_vs_pre_stellar**.

Usage:
  1. --plot_3_snap_comp or -plot_3sc to run the code to save the comparision of snapshots right before sink point creation and three before that snapshot.

#### residual_analysis.py
Python file containing code used to generate the heatmaps in **figures/residual_correlation_heatmaps** along with functions that generate pairplots of a variety of comparisons
between variables.

#### internal_gas_model.py
Python file containing code for both the Greedy and LASSO-tuned methods of feature selection. The internal gas properties and final snap time before a sink point is generated are the input
features being selected from. 

Usage: 
  1. --greedy or -gr flag followed by an integer evaluates the greedy model on the training dataset
  2. --heatmaps or -hm flag shows the intermediate heatmaps of features vs residuals and mass for the greedy algorithm 
  3. --lasso or -l flag calls the LASSO-aided model to be (the default alpha is set to 0.01 which produced 8 predictors)
  4. --alpha or -a flag followed by a float sets the weight of the $L_1$ regularization for the LASSO model
  5. --unit-test or -ut flag runs the unit-test for both the greedy and LASSO-aided models

The Greedy model works by iteratively adding features to a linear regression model. The first feature is selected by choosing the highest correlation with 
the mass. Each subsequent feature is chosen based on the highest covariance with the residuals of the current linear model. This iteration is carried out for a determined finite number of steps. 
The LASSO-tuned model uses a LASSO fit from the entire input space to the final mass. All features that are not tuned to zero by the $L_1$ loss are then fit to a linear regession model to the mass.

#### PCR.py
Python file containing code used to generate the **figures/PCA** figures and PCR regression. The lasso-tuned and greedy models are also implemented on the principal components in this file (shows the 2D PCA by default with mass colored). Note: Both the greedy and LASSO models with 8 components choose the same components (1, 2, 3, 4, 5, 6, 9, 13). Because of multicollinearity the PCA is reduced rank which can been seen in **figures/PCA**.

Usage:
  1. --pca_3d or -3d flag visualizes the first 3 principal components in 3d space 
  2. --blind_regression or -br flag visualizes a blind regression of all of the features
  3. --n_components or -n_comp flag followed by a positive int determines the number of principal compoenents with the default being 21 (the maximum)
  4. --greedy or -gr flag runs the greedy PCA model with 8 components
  5. --lasso or -l flag runs the LASSO-aided PCA model with 8 components



#### t-SNE.py
Python file containing code used to generate **figures/tsne_visualization**. t-SNE is applied to the internal pre-stellar gas properties and colored by final mass. 

Usage:
  1. --tsne_2d or -2d flag visualizes the 2d t-SNE
  2. --tsne_3d or -3d flag visualizes the 3d t-SNE

#### model_comparision.py
Python file that creates visual comparisions of the train and validation RMSE of different complexities of models on pre-stellar gas and the consistency of their coefficients. 

Usage:
  1. --rmse flag runs a function that compares the performance of the greedy, greedy PCA, LASSO, and LASSO PCA models as more features are included (produces **figures/model_RMSE_comparision**).
  2. --greedy_coefs or -g_cfs visualizes the difference in coefficients between the greedy model and their univariate linear regression coefficient for a selected number of features
  3. --lasso_coefs or -l_cfs visualizes the difference in coefficients between the LASSO model and their univariate linear regression coefficient for a selected number of features

#### process_data.py
Python file which preprocesses train and validation datasets for analysis. 

## star_data

#### star_data/data_properties.pdf
Contains a breakdown of the specific contents of stellar data provided in different parquet files.

#### star_data/accreted_gas_properties_pre_stellar.pq
Parquet file containing pre-stellar gas properties and statistics associated with each star the snapshot before a sink point is produced by that star in the STARFORGE simulation.
This simulation is used as the training data for predictive models and data analysis.

#### star_data/accreted_gas_properties_pre_stellar_test.pq
Parquet file containing pre-stellar gas properties and statisics associated with each star the snapshot before a sink point is produced by that star in a STARFORGE simulation with 
the same hyperparmeters but different initial conditions. This dataset is used as the validation set for predictive models.

## figures

#### figures/PCA
Folder containing figures analyzing the magnitude of principal components from a PCA performed on internal gas statisics (of $T$, $n_e$, $B$, $\rho$) from accreted_gas_properties_pre_stellar.pq. All figures
in the folder were produced from **PCR.py**.
* **components_ratios.pdf** visualizes the proportion of variance described by each principal compoenent
* **cumulative_variance.pdf** visualizes the contribution to the total variance using a cumulative distribution
* **pca_8_components.pdf** visualizes a PCR (Principal Component Regression) using the first 8 components in the space to predict the final stellar mass.

#### figures/greedy_figures
Folder containing the results on the training data of the greedy linear regression algorithm from **internal_gas_model.py** using from 1 to 10 features.

#### figures/residual_correlation_heatmaps
Folder containing correlation heatmaps comparing residuals of different models, shown in **figures/sp_kin** and **figures/sp_mag** for the specific kinetic and specific magnetic energies respectively,
and internal properties of the pre-stellar gas of each star. Generated by **data_analysis_v2.py**.

#### figures/sp_kin
Folder containing plots analyzing the fits of four different regression fits in log-log space(Linear, Piecewise-Linear, Polynomial, and Log-Normal/Power-Law) to the relation between specific kinetic energy the snap before sink point
formation and the final mass of stars. 
* **sp_kin_regression_plots.pdf** visualizes the scatter plot vs the fits
* **sp_kin_residuals_distribution.pdf** visualizes the distribution of the residuals with a gaussian fit for reference
* **sp_kin_residuals_vs_X.pdf** visualizes the residuals as a function of the specific kinetic energy
* **sp_kin_sliding_window.pdf** visualizes the mean and standard deviation of the final star mass using a sliding window

#### figures/sp_mag
Folder containing plots analyzing the fits of four different regression fits in log-log space(Linear, Piecewise-Linear, Polynomial, and Log-Normal/Power-Law) to the relation between specific magnetic energy the snap before sink point
formation and the final mass of stars. 
* **sp_mag_regression_plots.pdf** visualizes the scatter plot vs the fits
* **sp_mag_residuals_distribution.pdf** visualizes the distribution of the residuals with a gaussian fit for reference
* **sp_mag_residuals_vs_X.pdf** visualizes the residuals as a function of the specific magnetic energy
* **sp_mag_sliding_window.pdf** visualizes the mean and standard deviation of the final star mass using a sliding window

#### figures/three_snap_vs_pre_stellar
Folder containing figures inspecting how the stellar gas data is evolving through time. Specifically investigating 3 snaps backwards in time for each data point.

Notes on figures:
* All the pre-stellar gas's magnetic fields appear to be increasing as they approach the pre-sink snap time. The standard deviation in the distribution also appears to increase generally.
* $\rho$ statistics appears to follow a similar trend as magnetic field but with a much tighter distribution. In the rho_16 and rho_med figures it is clear that already more dense gas in the older data yields a greater rate of increase in the stats. Additionally, the standard deviation of the change in these statistics when comparing different stars seems to be increasing with higher prior density.
* For $n_e$, the standard deviation and mean don't appear to be very informative. However, the median, 16th percentile, and 84th percentile appear to show a large change from being relatively uniformly distributed to being more varied like a sort of "decay" of the statistic as time approaches the sink creation time.
* The temperature statistics appears to stay relatively consistent with a general decrease in the standard deviation and marginal increase in all the other statistics on average.
* The mass shows some variance which I beleive can just be attributed to floating point error of dividing values close to zero (just there to validate alignment of value comparison).


# Week 4 Notes

### Time Series Investigation
**6/22** - I am starting to investigate the time series data. The first step I am planning on taking is creating a dataset of the gas formation 3 snaps before the pre-stellar snap (the snap before a sink point is formed). I am interested in investigating the difference in predictive capacity of the older gas data and stellar mass. Additionally, I am interested in quantifying how fast the internal gas properties evolve with respect to the time from the pre-sink snap. To investigate the relationship between the pre-sink snap data and the data from 3 snaps prior, I am generating histograms of the percent changes in the features by star and ploting the pre-sink and -3 snap data in a log-log scatter plot.

##### Next steps:
* [x] 1. Create a pre-stellar snap minus n snaps function to investigate the different datasets
* [x] 2. Analyze the connection between the pre-stellar snap features and the features at minus 3 snaps

##

### Pre-Stellar Continuation
**6/22** - B, $n_e$, T, $\rho$ statistics and the pre-stellar snap have weak correlation independently with mass but collectively have predictive power of the mass in linear regression. The two models being employed to select the most important features in predicting final mass are the LASSO-aided model and a greedy residual model both in feature coordinates and PCA coordinates. The validation set of data from a simulation run with different initial conditions is fit well by the training dataset further supporting a meaningful connection between the variables. 

###### Side Note: The specific internal energy is almost perfectly correlated with mean temperature while sig_3D is perfectly correlated with specific kinetic energy. This can be visualized by running **residual_analysis.py**.

**6/23** - After comparing the RMSE performance of the four different models it appears that the greedy model in the normal coordinate space is the most efficient at choosing loss reducing predictors. The base greedy model being a better preditor than the PCA models is a positive outcome as a model with meaningful components provides more explanatory value in understanding the characteristics of pre-stellar gas's connection to final stellar mass. However, I still need to investiate if the coefficients in multicompoent models differ meaningfully from independent regression on a feature (i.e. next step two of this section). 

##### Next steps:
* [x] 1. Create a visualization of the RMSE of each model as the number of features increases (LASSO, PCA-LASSO, Greedy, PCA-Greedy) (see **figures/model_RMSE_comparision.pdf**)
* [x] 2. Create a visualization of the exponent of the power-law with respect to mass to see if it is conserved/consistent as more (potentially colinear) variables are added (generated from **model_comparision.py**)
* [x] 3. Create a method which visualizes the residuals vs next chosen predictor in the greedy algorithm
* [x] 4. Implement pure LASSO and LASSO-CV

**6/24** - The first seven features selected by the greedy model in order are T_std, rho_med, T_med, B_std, rho, Ne_std, rho_std. The standard deviation of temperature I think can be connected to the gas being more disperate in larger stars while smaller stars only have local gas (Kaalva et al.). The 6th choice, Ne_std, is the second most correlated as a single variable. There are two clusters that appear to form in the $n_e$ statistics. This should be investigated.

**6/25** - The "LASSO-aided" model where I have been running LASSO to choose features then running a linear regression is known as Post-LASSO. It aims to create sparse models while getting rid of regularization biased placed on features you are interested in using as predictors. There appear to be some statistical problems when performing hypothesis testing on Post-LASSO. Debiased-LASSO is another methd taht behaves in a similar way but hypothesis tests and error estimation methods remain valid. The LASSO CV model with 5-folds for cross validation resulted in the selection of 12 features.

##

### Unit Testing of Model
**6/22** - I created artificial linearly correlated data in **process_data.py** with noise to test the LASSO and greedy models in **internal_gas_model.py**. The models appear to be able to use the correlations to parse good fits even with low signal to noise ratios. I am unsure how to evaluate the theoretical $R^2$ values of the models. I am planning on creating different highly collinear dimensions and see how the values are selected (desiring the models to select un-correlated predictors more often).

##### Next steps:
* 1. [ ] Test **internal_gas_model.py** with collinear data.


## Week 5 Notes

### Time Series Investigation

**6/29** - As we move back in time I want to know: What is the nature of the rate of information loss for predicting mass? In the case that it is exponential, what is the half-life of information loss in the simulation? I plan on applying a model to each snapshot before sink point creation and see how the $R^2$ evolves as a metric of this information loss. The type of model needed for such a task could need to be robust to non-linear relations so I will implement a random forest in addition to a simple linear regression model. The relability of prediction in both the train and validation sets should be investigated.

##### Next Steps:
* [ ] 1. Produce heatmaps of internal gas properties of different pre-sink snapshots
* [ ] 2. Analyze $R^2$ decay as time evolves using linear regression (train and validation)
* [ ] 3. Analyze $R^2$ decay as time evolves using random forest (train and validation)

##

### Pre-Stellar Continuation 

**6/29** - The LASSO-CV model has been successful in predicting the star mass with an $R^2$ of 0.87. However, it selects 16 features when normalized. When the data is not normalized it has the same predictive power of $R^2$ = 0.87 but is more sparse with only 12 features being selected. A deeper investigation of the loss as a function of alpha should be investigated to determine the contour of the loss space. Additionally, a the $n_e$ appears to be a good predictor of low mass stars that turns into scatter for higher mass stars. A linear piecewise function could be useful in determining how informative $n_e$ values are in predicting the final mass.

###### Side Inquiry: Should I refactor **process_data.py** to pre-standardize all input features in the training and validation sets with respect to the mean and standard deviation of the training set? Some reading has lead me to believe that validation data should be normalized with respect to training statistics and not its own ([source](https://medium.com/@spinjosovsky/normalize-data-before-or-after-split-of-training-and-testing-data-7b8005f81e26)).  


##### Next Steps:
* [ ] 1. Compare to LASSO-aided linear regression to LASSO-CV
* [ ] 2. Investigate LASSO loss as a function of alpha for standardized data
* [ ] 3. Investigate Ridge Loss instead of $L_1$ LASSO loss
* [ ] 4. Investigate a piecewise model of $n_e$ properties (train and validation)

### Sampling influencing Standard Deviation

The potential sampling problem with the standard deviation is being approached by replacing it with a different statistic that investigates how the star looking 0.01 parsecs around the center of mass of the star. To make sure the same sampling error isn't repeated a subset of stars may be excluded that are too small and be unreliable due to sampling issues.

##### Next Steps:
* [x] 1. Look at distribution of "safe" stars to select (108 are elminated by demanding more than 10 particles, 65 are then eliminated demanding greater max distance than 0.005 pcs)
* [ ] 2. Investigate correlation strength with final mass on "safe" subset of stars for the 0.01 pc cut
* [ ] 3. Investigate resulting changes in model performance
