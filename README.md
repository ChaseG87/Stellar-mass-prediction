# Stellar-mass-prediction
This is a working repository analyzing relationship between pre-stellar gas properties and the final mass of stars using STARFORGE simulations of GMCs.

## Files and Usage

#### data_analysis_v1.py
Python file containing initial investigations of the correlations in the dataset primarily using heatmaps and scatter plots with power-law fits in log-log space.

#### data_analysis_v2.py
Python file containing code used to generate the different fitting methods (Linear, Piecewise-Linear, Polynomial, Log-Normal/Power-law) of the specific kinetic and specific
magnetic energy. Created the plots in **figures/sp_kin** and **figures/sp_mag**.

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
Python file containing code used to generate the **figures/PCA** figures and PCR regression. The lasso-tuned and greedy models are also implemented on the principal components in this file (shows the 2D PCA by default with mass colored).

Usage:
  1. --pca_3d or -3d flag visualizes the first 3 principal components in 3d space 
  2. --blind_regression or -br flag visualizes a blind regression of all of the features
  3. --n_components or -n_comp flag followed by a positive int determines the number of principal compoenents with the default being 21 (the maximum)



#### t-SNE.py
Python file containing code used to generate **figures/tsne_visualization**. t-SNE is applied to the internal pre-stellar gas properties and colored by final mass. 

Usage:
  1. --tsne_2d or -2d flag visualizes the 2d t-SNE
  2. --tsne_3d or -3d flag visualizes the 3d t-SNE

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


# Week 4 Notes

### Time Series Investigation
6/22 - I am starting to investigate the time series data. The first step I am planning on taking is creating a dataset of the gas formation 3 snaps before the pre-stellar snap (the snap before a sink point is formed). I am interested in investigating the difference in predictive capacity of the older gas data and stellar mass. Additionally, I am interested in quantifying how fast the observed quantities I am using as predictors evolve with respect to the time from the pre-stellar snap. To investigate the relationship between the pre-stellar snap data and the data from 3 snaps prior I am generating histograms of the percent changes in the features by star and ploting the pre-stellar and older data in a log-log scatter plot.

##### Next steps:
* [x] Create a pre-stellar snap minus n snaps function to investigate the different datasets
* [ ] Analyze the connection between the pre-stellar snap features and the features of the minus n snaps features
* [ ] Analyze the predictive capactity of the minus n snaps features onto mass

### Pre-Stellar Continuation
6/22 - B, $n_e$, T, $\rho$ statistics and the pre-stellar snap have weak correlation independently with mass but collectively have predictive power of the mass in linear regression. The two models being employed to select the most important features in predicting final mass are the LASSO-aided model and a greedy residual model both in feature coordinates and PCA coordinates. The validation set of data from a simulation run with different initial conditions is fit well by the training dataset further supporting a meaningful connection between the variables. 

##### Next steps:
* [ ] Create a visualization of the RMSE of each model as the number of features increases (LASSO, PCA-LASSO, Greedy, PCA-Greedy)
* [ ] Create a visualization of the exponent of the power-law with respect to mass to see if it is conserved/consistent as more (potentially colinear) variables are added

### Unit Testing of Model
6/22 - I created artificial linearly correlated data in **process_data.py** with noise to test the LASSO and greedy models in **internal_gas_model.py**. The models appear to be able to use the correlations to parse good fits even with low signal to noise ratios. I am unsure how to evaluate the theoretical $R^2$ values of the models. I am planning on creating different highly collinear dimensions and see how the values are selected (desiring the models to select un-correlated predictors more often).

##### Next steps:
* [ ] Test **internal_gas_model.py** with collinear data.


