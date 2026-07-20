# Stellar-mass-prediction
This is a working repository analyzing relationship between pre-stellar gas properties and the final mass of stars using STARFORGE simulations of GMCs.

## Files and Usage

#### data_analysis_v1.py
Python file containing initial investigations of the correlations in the dataset primarily using heatmaps and scatter plots with power-law fits in log-log space.

#### data_analysis_v2.py
Python file containing code used to generate the different fitting methods (Linear, Piecewise-Linear, Polynomial, Log-Normal/Power-law) of the specific kinetic and specific
magnetic energy. Created the plots in [figures/sp_kin](figures/sp_kin) and [figures/sp_kin](figures/sp_mag).

#### data_analysis_v3.py
Python file containing code used to perform data analysis on time series data at different snapshots. Used to produce the scatter plots in [figures/three_snap_vs_pre_stellar](figures/three_snap_vs_pre_stellar) and [figures/time_series_heatmaps](figures/time_series_heatmaps).

Usage:
  1. --plot_3_snap_comp or -plot_3sc flag to run the code to save the comparision of snapshots right before sink point creation and three before that snapshot.
  2. --plot_heatmap or -hm flag to generate a heatmap n snapshots before the pre-sink snapshot
  3. --save or -s flag that saves generated figures from other flags

#### residual_analysis.py
Python file containing code used to generate the heatmaps in [residual_correlation_heatmaps](figures_bad_target/residual_correlation_heatmaps) along with functions that generate pairplots of a variety of comparisons
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
Python file containing code used to generate the [figures/PCA](figures/PCA) figures and PCR regression. The lasso-tuned and greedy models are also implemented on the principal components in this file (shows the 2D PCA by default with mass colored). Note: Both the greedy and LASSO models with 8 components choose the same components (1, 2, 3, 4, 5, 6, 9, 13). Because of multicollinearity the PCA is reduced rank which can been seen in [figures/PCA](figures/PCA).

Usage:
  1. --pca_3d or -3d flag visualizes the first 3 principal components in 3d space 
  2. --blind_regression or -br flag visualizes a blind regression of all of the features
  3. --n_components or -n_comp flag followed by a positive int determines the number of principal compoenents with the default being 21 (the maximum)
  4. --greedy or -gr flag runs the greedy PCA model with 8 components
  5. --lasso or -l flag runs the LASSO-aided PCA model with 8 components

#### information_decay.py
Python file containing code used to visualize the rate of information loss with respect to the number of snapshots back from the pre-sink snapshot. 

Usage:
  1. --linear or -l flag visualizes the $R^2$ values of training and validation sets as you move back in time for a linear regression model
  2. --random_forest or -rf flag visualizes the $R^2$ values of training and validation sets as you move back in time for a random forest model
  3. --rf_grid_search or -rf_gs flag runs a grid search of random forest hyperparameters and determines the best parameters using cross validation on the pre-sink snapshot

#### t-SNE.py
Python file containing code used to generate [figures/tsne_visualization.pdf](figures/tsne_visualization.pdf). t-SNE is applied to the internal pre-stellar gas properties and colored by final mass. 

Usage:
  1. --tsne_2d or -2d flag visualizes the 2d t-SNE
  2. --tsne_3d or -3d flag visualizes the 3d t-SNE
  3. --patch or -p flag called in tandem with --tsne_2d visualizes a 2d t-SNE using the 0.01a parsec patch data

#### model_comparision.py
Python file that creates visual comparisions of the train and validation RMSE of different complexities of models on pre-stellar gas and the consistency of their coefficients. 

Usage:
  1. --rmse flag runs a function that compares the performance of the greedy, greedy PCA, LASSO, and LASSO PCA models as more features are included (produces [figures/model_RMSE_comparison.pdf](figures/model_RMSE_comparison.pdf)).
  2. --greedy_coefs or -g_cfs visualizes the difference in coefficients between the greedy model and their univariate linear regression coefficient for a selected number of features
  3. --lasso_coefs or -l_cfs visualizes the difference in coefficients between the LASSO model and their univariate linear regression coefficient for a selected number of features

#### process_data.py
Python file which preprocesses train and validation datasets for analysis. 

#### monte_carlo.py
Python file which computes a Monte Carlo resampling of a provided data set using the standard errors of a particular star to inform the resampling.

#### large_patch_analysis.py
Python file producing figures analyzing the 0.5 parsec pre-stellar snapshot patch dataset. 

Usage:
  1. --heatmap or -hm flag visualizes a heatmap of the 0.5 parsec patch data
  2. --tsne flag visualizes a tsne using the provided features colored by mass
  3. --lasso_cv or -l flag plots a lasso-cv model on the 0.5 parsec patch data predicting for final mass
  4. --single or -s flag uses only the singular data to plot figures for the selected visualization 

## star_data

#### [star_data/data_properties.pdf](star_data/data_properties.pdf)
Contains a breakdown of the specific contents of stellar data provided in different parquet files.

#### star_data/accreted_gas_properties_pre_stellar.pq
Parquet file containing pre-stellar gas properties and statistics associated with each star the snapshot before a sink point is produced by that star in the STARFORGE simulation.
This simulation is used as the training data for predictive models and data analysis.

#### star_data/accreted_gas_properties_pre_stellar_test.pq
Parquet file containing pre-stellar gas properties and statisics associated with each star the snapshot before a sink point is produced by that star in a STARFORGE simulation with 
the same hyperparmeters but different initial conditions. This dataset is used as the validation set for predictive models.

## figures

#### [figures/PCA](figures/PCA)
Folder containing figures analyzing the magnitude of principal components from a PCA performed on internal gas statisics (of $T$, $n_e$, $B$, $\rho$) from accreted_gas_properties_pre_stellar.pq. All figures
in the folder were produced from **PCR.py**.
* **components_ratios.pdf** visualizes the proportion of variance described by each principal compoenent
* **cumulative_variance.pdf** visualizes the contribution to the total variance using a cumulative distribution
* **pca_8_components.pdf** visualizes a PCR (Principal Component Regression) using the first 8 components in the space to predict the final stellar mass.

#### [figures/greedy_figures](figures/greedy_figures)
Folder containing the results on the training data of the greedy linear regression algorithm from **internal_gas_model.py** using from 1 to 10 features.

#### [figures/residual_correlation_heatmaps](figures/residual_correlation_heatmaps)
Folder containing correlation heatmaps comparing residuals of different models, shown in **figures/sp_kin** and **figures/sp_mag** for the specific kinetic and specific magnetic energies respectively,
and internal properties of the pre-stellar gas of each star. Generated by **data_analysis_v2.py**.

#### [figures/sp_kin](figures/sp_kin)
Folder containing plots analyzing the fits of four different regression fits in log-log space(Linear, Piecewise-Linear, Polynomial, and Log-Normal/Power-Law) to the relation between specific kinetic energy the snap before sink point
formation and the final mass of stars. 
* **sp_kin_regression_plots.pdf** visualizes the scatter plot vs the fits
* **sp_kin_residuals_distribution.pdf** visualizes the distribution of the residuals with a gaussian fit for reference
* **sp_kin_residuals_vs_X.pdf** visualizes the residuals as a function of the specific kinetic energy
* **sp_kin_sliding_window.pdf** visualizes the mean and standard deviation of the final star mass using a sliding window

#### [figures/sp_mag](figures/sp_mag)
Folder containing plots analyzing the fits of four different regression fits in log-log space(Linear, Piecewise-Linear, Polynomial, and Log-Normal/Power-Law) to the relation between specific magnetic energy the snap before sink point
formation and the final mass of stars. 
* **sp_mag_regression_plots.pdf** visualizes the scatter plot vs the fits
* **sp_mag_residuals_distribution.pdf** visualizes the distribution of the residuals with a gaussian fit for reference
* **sp_mag_residuals_vs_X.pdf** visualizes the residuals as a function of the specific magnetic energy
* **sp_mag_sliding_window.pdf** visualizes the mean and standard deviation of the final star mass using a sliding window

#### [figures/three_snap_vs_pre_stellar](figures/three_snap_vs_pre_stellar)
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


### Pre-Stellar Continuation
**6/22** - B, $n_e$, T, $\rho$ statistics and the pre-stellar snap have weak correlation independently with mass but collectively have predictive power of the mass in linear regression. The two models being employed to select the most important features in predicting final mass are the LASSO-aided model and a greedy residual model both in feature coordinates and PCA coordinates. The validation set of data from a simulation run with different initial conditions is fit well by the training dataset further supporting a meaningful connection between the variables. 

###### Side Note: The specific internal energy is almost perfectly correlated with mean temperature while sig_3D is perfectly correlated with specific kinetic energy. This can be visualized by running **residual_analysis.py**.

**6/23** - After comparing the RMSE performance of the four different models it appears that the greedy model in the normal coordinate space is the most efficient at choosing loss reducing predictors. The base greedy model being a better preditor than the PCA models is a positive outcome as a model with meaningful components provides more explanatory value in understanding the characteristics of pre-stellar gas's connection to final stellar mass. However, I still need to investiate if the coefficients in multicompoent models differ meaningfully from independent regression on a feature (i.e. next step two of this section). 

##### Next steps:
* [x] 1. Create a visualization of the RMSE of each model as the number of features increases (LASSO, PCA-LASSO, Greedy, PCA-Greedy) (see [figures/model_RMSE_comparision.pdf](figures/model_RMSE_comparision.pdf))
* [x] 2. Create a visualization of the exponent of the power-law with respect to mass to see if it is conserved/consistent as more (potentially colinear) variables are added (generated from **model_comparision.py**, example: [figures/lasso_coefficient_comp.pdf](figures/lasso_coefficient_comp.pdf))
* [x] 3. Create a method which visualizes the residuals vs next chosen predictor in the greedy algorithm
* [x] 4. Implement pure LASSO and LASSO-CV

**6/24** - The first seven features selected by the greedy model in order are T_std, rho_med, T_med, B_std, rho, Ne_std, rho_std. The standard deviation of temperature I think can be connected to the gas being more disperate in larger stars while smaller stars only have local gas (Kaalva et al.). The 6th choice, Ne_std, is the second most correlated as a single variable. There are two clusters that appear to form in the $n_e$ statistics. This should be investigated.

**6/25** - The "LASSO-aided" model where I have been running LASSO to choose features then running a linear regression is known as Post-LASSO. It aims to create sparse models while getting rid of regularization biased placed on features you are interested in using as predictors. There appear to be some statistical problems when performing hypothesis testing on Post-LASSO. Debiased-LASSO is another methd taht behaves in a similar way but hypothesis tests and error estimation methods remain valid. The LASSO CV model with 5-folds for cross validation resulted in the selection of 12 features.

### Unit Testing of Model
**6/22** - I created artificial linearly correlated data in **process_data.py** with noise to test the LASSO and greedy models in **internal_gas_model.py**. The models appear to be able to use the correlations to parse good fits even with low signal to noise ratios. I am unsure how to evaluate the theoretical $R^2$ values of the models. I am planning on creating different highly collinear dimensions and see how the values are selected (desiring the models to select un-correlated predictors more often).


# Week 5 Notes

### Time Series Investigation

**6/29** - As we move back in time I want to know: What is the nature of the rate of information loss for predicting mass? In the case that it is exponential, what is the half-life of information loss in the simulation? I plan on applying a model to each snapshot before sink point creation and see how the $R^2$ evolves as a metric of this information loss. The type of model needed for such a task could need to be robust to non-linear relations so I will implement a random forest in addition to a simple linear regression model. The relability of prediction in both the train and validation sets should be investigated. 

**7/1** - The decay of the $R^2$ value of the full linear regression model when applied to the internal gas propererties excluding both the standard deviation and standard error yields a characteristic exponential shape. Additionally, the distance between the training and validation data increases implying past a certain point the model is fitting more to noise as the signal is reduced. This drop of in time series data is what I expect of a highly turbulent system like the STARFORGE simulations.

**7/2** - The validation performance of both the linear regression model and random forest's $R^2$ values decay exponentially as they move back in time. This can be seen in the information decay plots with the linear decay of $R^2$ on their linear-log axes. The correlation coefficient between the validation $R^2$ exponentiated and snapshot number is -0.89 for the linear model and -0.90 for the random forest model (these change when the base is change but remain high). I think the exponential rate of losing information in time, which is what you expect turbulent system, strengthens the variables I am investigating are not subject to trivial data leaking. When the standard deviations of the features are included in the analysis, the shape of the decay changes from a clean exponential to initially losing data exponentially then diverges to be better at predicting than the models without standard deviation statistics. This leads me to believe there is data leakage in the standard deviation as the models retain greater predictive performance than you would expect in a turbulent system.

For the random forest model I tuned the hyperparameters of max_features, max_depth, min_samples_split, and min_samples_leaf. I am concerned that the training performance way overperforms the validation data in old snapshots but this appears to be a feature of random forests with many trees.

**7/10** - The half-life of information loss (the rate of $R^2$ decay in predicting the final mass) is about 14 snapshots which equates to about 345,800 years. I am curious if this aligns with some other sense of information loss rate due to turbulence in the simulation.

##### Next Steps:
* [x] 1. Produce heatmaps of internal gas properties of different pre-sink snapshots ([time series heatmaps](figures/time_series_heatmaps))
* [x] 2. Analyze $R^2$ decay as time evolves using linear regression ([figures/information_decay_linear_regression.pdf](figures/information_decay_linear_regression.pdf))
* [x] 3. Tune hyperparameters for the random forest model.
* [x] 4. Analyze $R^2$ decay as time evolves using random forest ([figures/information_decay_random_forest.pdf](figures/information_decay_random_forest.pdf))
* [x] 5. Investigate results of decay vs. time plots to discover potentially exponential or power-law relationships (appears to be exponential in both linear and random forest)


### Pre-Stellar Continuation 

**6/29** - The LASSO-CV model has been successful in predicting the star mass with an $R^2$ of 0.87. However, it selects 16 features when normalized. When the data is not normalized it has the same predictive power of $R^2$ = 0.87 but is more sparse with only 12 features being selected. A deeper investigation of the loss as a function of alpha should be investigated to determine the contour of the loss space. Additionally, a the $n_e$ appears to be a good predictor of low mass stars that turns into scatter for higher mass stars. A linear piecewise function could be useful in determining how informative $n_e$ values are in predicting the final mass.

###### Side Inquiry: Should I refactor **process_data.py** to pre-standardize all input features in the training and validation sets with respect to the mean and standard deviation of the training set? Some reading has lead me to believe that validation data should be normalized with respect to training statistics and not its own ([source](https://medium.com/@spinjosovsky/normalize-data-before-or-after-split-of-training-and-testing-data-7b8005f81e26)).  

**6/30** - Including the standard error instead of the standard deviation in the model resulted in greater performance for the LASSO-CV model from $R^2 = 0.87$ to $R^2 = 0.92$. There is a risk of data leakage in the opposite direction of standard deviation as smaller samples with naturally have a larger standard error from the division by $\sqrt{n}$. 

**7/1** - Experimenting with Ridge loss and a mixed Ridge Loss and LASSO model known as Elastic Net, I have found that the coefficients remain similarly erradic when compared to the univariate coefficients. 

##### Next Steps:
* [x] 1. Compare to post-LASSO linear regression to coefficients to LASSO-CV (with the same features)
* [x] 2. Investigate Ridge Loss instead of $L_1$ LASSO loss
* [x] 3. Investigate Elastic Net (Ridge and LASSO loss)
* [x] 4. Investigate the affect of changing from standard deviation to standard error

### Sampling influencing Standard Deviation

**6/29** - The potential sampling problem with the standard deviation is being approached by replacing it with a different statistic that investigates how the star looking 0.01 parsecs around the center of mass of the star (0.01a). To make sure the same sampling error isn't repeated a subset of stars may be excluded that are too small and be unreliable due to sampling issues. The 0.01a pc patch data is a lot less correlated than the data of the full pre-stellar cloud. When modelling a LASSO-CV model onto the patch data an $R^2$ of 0.38 is achieved with especially poor predictive capabilities for smaller stars. When including the magnitude of angular velocity the performance does not improve. 

**6/30** - The time series heatmaps show a consistent connection between the standard deviation of features and the total mass. This maintained correlation leads me to believe there is a sampling bias or some other bias connecting the standard deviations to a trival correlation to mass. Looking the standard error ($\frac{std}{\sqrt{n}}$) as you go back in time, the statistic becomes less correlated to the total mass. This is an expected behavior in the turbulent system of the STARFORGE simulation and further supports the view that standard deviation is subject to contamination of some kind.

Increasing the minimum 'max_dist_peak' to 0.009 and requiring a minimum of 10 particles reduces the total number of stars by 296. This reduction did not appear to improve the performance of the LASSO-CV model even with the inclusion of angular velocities and the magnitude of mass weighted angular velocity. 

**7/1** - Removing low mass stars (less than 0.1 ($M_\odot$) increases the performance of the 0.01a pc patch model to an $R^2$ of 0.59. This operation eliminates 738 total stars which is about 1/3rd of the full dataset. However, the mass of these stars are in the brown dwarf zone which could explain their divergent behavior.


###### **7/2** - The ground truth for the prior analysis has been incorrectly using M_tot, the total mass instead of mf, the final mass of the star. After fixing the preprocessing to shift the target over the results appear to be qualitatively the same in the prior analysis. The old plots will be stored in [figures_bad_target](figures_bad_target) and all future figures and regenerated figures with the correct figure will be in [figures](figures).


**7/3** - Including the global potential energy to the model and comparing it to final mass yielded a small positive correlation of 0.16. When including it in both the LASSO and random forest models it had no additional benefit to the predictive performance. 


##### Next Steps:
* [x] 1. Look at distribution of "safe" stars to select (168 are elminated by demanding at least 10 particles and a greater max distance than 0.005 pcs of the 2249 total)
* [x] 2. Investigate correlation strength with final mass on "safe" subset of stars for the 0.01a pc patch ([figures/patch_figures_0.01a/heatmap_0.01_patch.pdf](figures/patch_figures_0.01a/heatmap_0.01_patch.pdf))
* [x] 3. Investigate resulting changes in model performance ([figures/patch_figures_0.01a/lasso_cv_0.01_patch.pdf](figures/patch_figures_0.01a/lasso_cv_0.01_patch.pdf))
* [x] 4. Investigate the performace of random forest on the 0.01 pc patch data
* [x] 5. Investigate the performance of angular velocity on 0.01 pc patch data ([figures/patch_figures_0.01a/internal_gas_0.01_patch.pdf](figures/patch_figures_0.01a/internal_gas_0.01_patch.pdf)* updated)
* [x] 6. Investigate the connection between the global potential energy at the pre-sink snapshot and final mass
* [x] 7. Investigate the performace of validation partitions of 0.01 pc patch data ([figures/patch_figures_0.01a/lasso_cv_patch_validation.pdf](figures/patch_figures_0.01a/lasso_cv_patch_validation.pdf))


# Week 6 Notes

### Monte Carlo Resampling

**7/6** - There is a wide range of the number of particles being used to derive the statisics used in the model. To understand the impact of sampling error on our predictive models I am using Monte Carlo Resampling. The standard error for each of the values will be used to generate the artificial data points. Monte Carlo Resampling requires a "kernel" underlying distribution (usually guassian) but I am unsure of the shape of the distributions of the underlying distributions (B, T, $\rho$, $n_e$). The resampling I carried out breaks the stars into five mass categories and computes a covariance matrix for each. Then, for each star the covariance matrix is divided by the number of particles used to generate the statistics. Finally the features are sampled from the multivariate gaussian space from the covariance matrix and mass is kept the same.

##### Next Steps:
* [x] 1. Generate a data set from Monte Carlo Resampling in the training set for full data
* [x] 2. Generate a data set from Monte Carlo Resampling in the training set for the patch data
* [x] 3. Fit LASSO-CV model to Monte Carlo resample of full data
* [x] 4. Fit LASSO-CV model to Monte Carlo resample of patch data
* [x] 5. Transform covariance matrix informing multivariate distribution by standard error

**7/7** - For each data point, five samples of input features are taken from a multivariate distribution using the covariance of similarly sized stars regularized by the number of particles that make up a given star (mass is untouched). When fitting a LASSO-CV to the resulting the samples from the full data and patch data, a difference in the prediction variance is seen ([figures/lasso_cv_5_fold_monte_carlo.pdf](figures/lasso_cv_5_fold_monte_carlo.pdf) and [figures/lasso_patch_monte_carlo.pdf](figures/lasso_patch_monte_carlo.pdf)) to exhibit different behavior. In the full data case the variance in the prediction from the model for each star reduces for larger stars. I believe this is a reflection of the lower overall variance in the data due to the regularization based on sample size. In the patch data it appears that even for large stars the LASSO model's predictions have a high amount of variance for stars sampled from the same mass. This divergent behavior is either a result of a difference in overall variance for similarly sized stars in the patch data or a result of the patch model fitting to more noise. I believe it is likely the latter as the current Monte Carlo sampling machinery is not weighting variances by the standard error.

###### A big caveat of using Monte Carlo Resampling in the way is that the assumed underlying distribution of these properties is log-normal (because the log is taken of the statisics, without it would be normal). If this assumption does not hold the method does not useful in informing the true behavior of the system. This property may need to be tested directly from the STARFORGE data.

**7/7** - I have formulated a method that uses the standard errors of the individual properties to calculate a covariance matrix which is then feed into a multivariate gaussian distribution for the creation of resampled properties. The process starts with approximating the standard error of the mean of each of the features in log space using the delta method. Then, using a correlation to convariance scaling with a diagonal matrix of the standard errors in log space and the correlation matrix of a specific star, the covariance matrix used for the basis of the multivariate gaussian distribution is created. I am finding that this method maintains high variance for both low and high mass stars when compared to just using the covariance of a group of stars regularized by $N_{\text{inc}}$ of a specific star which produces very tight distributions of features for large stars. I believe the standard error informed model could have this property due to the positive correlation between the mass and the standard deviation of different features which is scaling at a similar rate to $\sqrt{N_{\text{inc}}}$. Further investigation between the behavior of these two properties is needed. [Standard Deviation Informed LASSO-CV Results](figures/lasso_std_error_monte_carlo.pdf) 

**7/8** - Looking at the plot of [mass vs the standard errors](figures/mass_vs_std_er.pdf) of the different properties for the full cloud data, $T$ and $n_e$ appear to have positive correlations with the mass while $B$ and $\rho$ exhibit a negative correlation. These combinations of conflicting error relations are reflected in the behavior of the variance seen in Monte Carlo resampled stars which have a similar spread for both small and large stars. There is a concern that using the correlation matrix between the group of stars as the basis for covariance matrix along with the standard errors is not reflective of the correlation of the intrastellar gas properties. That is to say that the correlation of properties between stars could be different than the correlation of properties of the particles making up an individual star.

##### Next Steps:
* [x] 1. Investigate the relation $\dfrac{\sigma}{\sqrt{N_{\text{inc}}}}$ (SE) as a function of mass

### Error Weighting 
**7/6** - Due to the mass distribution in the data, the importance of predicting high mass stars is discounted due to the rarity of large stars. This results in the model spending its bandwidth on fitting to the plentiful low and mid mass stars due to their volume. To eliminate this bias, I propose a couple methods to improve the robustity of the analysis:

* [x] 1. Direct weighting in the training set (RMSE loss is weighted linearly based on mass directly)
* [x] 2. Monte Carlo Resampling (Use Monte Carlo resampling to create a dataset with similar weights at different mass ranges)


For the second approch, I am considering sampling a set number of stars from each of the five compartments used in the generation of the Monte Carlo resampling and running a model on that subset as a type of weighting.

**7/8** - The direct weighting approach weighting by the squareroot of the true mass of the stars and log of the mass of the stars (shifted for numerical reasons) both yielded similarly poor results for predicting the stellar mass using the properties in the 0.01 patch data. ([figures/patch_figures_0.01a/weighted_lasso](figures/patch_figures_0.01a/weighted_lasso))


### $R_{\text{eff}}$, $M_{\text{f}}$, and $\rho$ relationship

**7/6** - The power-law relationship found between $M_{\text{f}}$ and different $\rho$ statisics is approximately $\rho \propto M_{\text{f}}^{-2.3}$. The power-law relationship between $R_{\text{eff}}$ and $M_{\text{f}}$ is $R_{\text{eff}} \propto M_{\text{f}}^{1.06}$. Comparing the two measures using dimensional analysis we find agreement in the values even though $R_{\text{eff}}$ is a global view of the volume and the $\rho$ statistics are samples from the distribution of particles. 

Using $\rho \propto M_{\text{f}}^{-2.3}$:

$$
R_{\text{eff}} \propto \left( \frac{M}{\rho} \right)^{1/3} \rightarrow
R_{\text{eff}} \propto \left( \frac{M}{M^{-2.3}} \right)^{1/3} \rightarrow
R_{\text{eff}} \propto \left( M^{3.3} \right)^{1/3} \rightarrow
R_{\text{eff}} \propto M^{1.1} \approx M^{1.06}
$$

These power-law relationships represent a massive drop off in density and an effective radius scaling approximately proportionally to $M_\text{f}$ in agreement with Kaalva et al. The more massive a star is the more diffuse its mass is in the pre-stellar snapshot.

**7/9** - Investigating the free electron abundance ($n_e$) vs mass the skew in the underlying distribution results in very different power law relations for the mean and median values ($n_e$ mean $\propto M^{3.84}$ and $n_e$ median $\propto M^{1.29}$ with $R^2$ values of 0.49 and 0.51 respectively) in the full data. This discrepency could be an indicator of volume sampling bias though I am not sure how to interpret it. 

### Validation Set of Patch

**7/7** - Applying the same data reduction in the 0.01 parsec patch data of requiring the mass to be greater than 0.1 $M_\odot$ and the count of particles to be greater than 10 reduces the validation set from 1966 stars to 1508. Looking at both the results of the [LASSO-CV](figures/patch_figures_0.01a/lasso_cv_patch_validation.pdf) and [random forest](figures/patch_figures_0.01a/random_forest_patch_validation.pdf) model results, the signal separating large stars from smaller stars using all the gas properties appears to be poor with validation $R^2$ values of 0.29 and 0.27 respectively.

### Mean and Median Full Data

**7/8** - Looking at only the mean and median values for each of the four predictors (T, B, $n_e$, and $\rho$) the model performs still incredibly well ($R^2$ = 0.81) and all eight features are selected in the LASSO-CV process (can be seen in [figures/lasso_cv_5_mean_and_median.pdf](figures/lasso_cv_5_mean_and_median.pdf)). This reduction of features I believe eliminates the concern of predicting the mass through data leakage from the sample size of the standard deviations, 84th and 16th percentiles. 

##### Next Steps:
* [x] 1. Examine validation set performance of mean and median only model.
* [x] 2. Examine more reduced combinations for variables.
* [x] 3. Investigate the coefficients of the reduced model with just the medians for the full data.
* [x] 4. Investigate the residual correlations with $\rho_{\text{med}}$ as the first variable of interest.

**7/9** - Using just the four median values from each of the distributions yields a training $R^2$ of 0.79 and a validation $R^2$ of 0.78. Removing the median $n_e$ actually increases the validation performce very marginally. Removing $T_{\text{med}}$ or $B_{\text{med}}$ reduces the training and validation $R^2$ by about 0.04. Eliminating both of them plummets the validation $R^2$ to 0.47. This makes me think they are describing about the same variance in mass which is further supported by their cross correlation of 0.65. Dropping $\rho_{\text{med}}$ reduces the validation $R^2$ to 0.66. The plot of the validation performance is in [figures/lasso_validation_small_model.pdf](figures/lasso_validation_small_model.pdf)

**7/10** - The high predictive ability of the full data and poor predictive ability of the 0.01 patch data leads me to believe that the sampling of far out gas is important to the success of the full data models. Investigating at the specific coefficients and $R^2$ values of $T_{\text{med}}$ and $B_{\text{med}}$, the coefficient of $T_{\text{med}} = 0.48$ and $B_{\text{med}} = -0.40$. They both independently have weak correlations with mass with $R^2$ values of 0.08 and 0.10 respectively. Note that the coefficient of $B_{\text{med}}$ is very similar to that of $\rho_{\text{med}} = -0.41$ and their covariance is 0.85. It is interesting then with such minor direct predictive abilities of mass that the when removing both from the reduced model the validation $R^2$ drops from 0.78 to 0.47.

Another interesting feature of the data is the discrepency in the power-law from the fit to $T_{\text{mean}}$ versus $T_{\text{med}}$ as $T_{\text{mean}} \propto M^{1.4}$ and $T_{\text{med}} \propto M^{2.1}$. This could be a result of small variations in data having high impacts on the coefficients due to the low $R^2$ values but it could also be a characteristic separation of the scaling of the mean and median implying a non-log-normal distribution in temperature.

Investigating the coefficients of the LASSO-CV model and Post-LASSO models with only $T_{\text{med}}$, $B_{\text{med}}$, and $\rho{\text{med}}$ as selectable features, the coefficient of $T_{\text{med}}$ is very similar to it's univariate coefficient while the sign of $B_{\text{med}}$ flips and $\rho{\text{med}}$ is magnified to approximately twice the size. [figures/lasso_reduced_coefficient_comp.pdf](figures/lasso_reduced_coefficient_comp.pdf)

**7/14** - The correlation of the residuals of the univariate $\rho_{\text{med}}$ to the second chosen feature of $T_{\text{med}}$ is 0.69 which is much higher than the direct correlation between $T_{\text{med}}$ and the final mass of 0.28. The correlation of residuals of the two-variable model to the next chosen feature of $B_{\text{med}}$ is just 0.12. This results in a very minor bump in performace of the overall model.


### Multiplicity Investigation

**7/9** - In the [visualization of multiplicity](figures/lasso_cv_multiplicity.pdf) on the LASSO-CV model, for the full data there does not appear to be a clear separation of stars based on multiplicity besides the tendency of the largest stars to have a high multiplicity (4+). The model appears to be biased in underestimating the mass of these stars which could also be a product of the low count of these stars. In the patch data colored by multiplicity ([figures/patch_figures_0.01a/weighted_lasso](figures/patch_figures_0.01a/weighted_lasso)) the trend appears to be an overestimating of low multiplicity and an underestimation of high multiplicity stars.

###### I added a t-sne with the masses colored for the patch data in [figures/patch_figures/patch_tsne_visualization.pdf](figures/patch_data_patch_tsne_visualization.pdf)

# Week 7 Notes

### Weighting 0.01a Parsec Patch Data

**7/13** - The performance of weighted models is similarly poor for both the LASSO and random forest (seen in [figures/patch_figures_0.01a/weighted_lasso](figures/patch_figures_0.01a/weighted_lasso) and [figures/patch_figures_0.01a/weighted_random_forest](figures/patch_figures_0.01a/weighted_random_forest) respectively). The weighted and standard $R^2$ values track much more closely in the random forest model for both the training and validation sets when compared to the LASSO results. Through each application of weights the case for the features in the 0.01a parsec patch around the peak density in the pre-stellar snap being poor predictors of the final mass is further supported.

##### Next Steps:
* [x] 1. Investigate weighted $R^2$ to weighted LASSO-CV patch for training and validation
* [x] 2. Investigate weighted random forest performance on patch data (train and val with both $R^2$ values)

### Multiplicity Continued

**7/13** - The multiplicity appears to be highly correlated with mass for larger stars and a pure scatter for smaller stars. Investigating how it is predicted by the gas properties could yield physical insight into the kinds of stars that end up with higher multiplicity.

##### Next Steps:
* [x] 1. Examine results of different multiplicities separately for the reduced model on the full data
* [x] 2. Examine results of different multiplicities for the 0.01a patch data
* [x] 3. Investigate the use of a vector target using both mass and multiplicity
* [x] 4. Investigate t-SNE coloring by multiplicity

**7/14** - Coloring t-SNE on gas properties by multiplity for the full data reveals separaration in proportion of higher multiplicity stars in different regions but there is not a clear separation dividing low and high multiplicity stars ([figures/multiplicity_figures/multiplicity_tsne_visualization.pdf](figures/multiplicity_figures/multiplicity_tsne_visualization.pdf)). Investigating the full data linear regression model with the features of $\rho_{\text{med}}$, $T_{\text{med}}$, and $B_{\text{med}}$, stars with different multiplicities don't appear to perform differently from each other in either the training or validation set (seen in [figures/multiplicity_figures](figures/multiplicity_figures)). 

**7/15** - Examining the 0.01a pc patch broken up by multiplicity for the training and validation datasets, the $R^2$ values of stars with multiplicities of 1 and 2 are 0.12 and 0.13, respectively. The multiplicity 3 and 4 stars have $R^2$ values of 0.26 and 0.28, respectively. The validation set shows similar performace for each multiplicity besides the multiplicity 1 set which has a negative $R^2$ value ([figures/multiplicity_figures](figures/multiplicity_figures)). 

Looking at the reduced linear model performance broken up by multiplicity, there is a clear underprediction of the highest mass stars with a multiplicity of four ([figures/multiplicity_figures/lin_reg_train_multiplicity.pdf](figures/multiplicity_figures/lin_reg_train_multiplicity.pdf)). I believe this could either be due to a non-linear (in log-log space so power-law) behavior, A underweighting of more massive stars due to their rarity, or little distinguishing the large mass stars are massive stars in the gas properties.

**7/17** - Implementing a multi-target random forest does not appear to be advantageous in predicting mass in the reduced model. Additionally, the multi-target performs very poorly in predicting the max multiplicity in the validation set ($R^2 = 0.09$). This poor performance in the multiplicity could be a model selection issue as the max multiplicity behave as classes of integers not a continuous domain ([training](figures/multiplicity_figures/multi_target_rf_train.pdf) - [validation](figures/multiplicity_figures/multi_target_rf_validation.pdf)).

##### Next Steps:
* [x] 1. Investigate the performance of a random forest on full data broken up by multiplicity (to check if the kink is non-linear)
* [x] 2. Weight masses directly on full data broken up by multiplicity

**7/15** - The linear regression of the reduced model weighted by the squareroot of the mass and separated by multiplicity can be seen in [figures/multiplicity_figures](figures/multiplicity_figures). Even with aggressive weighting applied the largest stars still appear to be underestimated by the model. This leads me to believe that the that the poor fitting is either due to a poor ability to fit to non-linear data or a insignificant difference between the gas properties of large and massive stars. 

**7/16** - The random forest model on the full data with reduced features, on both weighted and unweighted runs broken up by multiplicity, presented the same characteristic kink at the highest mass stars in which the models underestimate the mass of the highest mass stars. This along with the linear weighting not working leads me to believe that there is little differentiating the large stars from the massive stars in the pre-stellar gas properties ([figures/multiplicity_figures/rand_forest_validation_multiplicity.pdf](figures/multiplicity_figures/rand_forest-validation_multiplicity.pdf)).

I also looked at the random forest with all of the gas property statistics and the same kink is exhibited with and without sample weigthing. From a physical perspective this could be due to the model being able to parse out the minimal Jeans mass that a star needs given the properties of gas and higher mass stars far exceed this minimum. It could also be a result of how the largest stars from according to the competitive accretion model.

### 0.5 Parsec Patch Investigation

**7/13** - I am running the 0.5 parsec pre-stellar sink snapshot patch data through the pipeline that I have from analyzing the other two data sets to see how it performs at predicting the mass. Looking at the correlation heatmap, the features appear to have even worse correlation to the final mass then the 0.01a patch data. This includes the direct correlation between final mass and num_stars which is the number of stars within the 0.5 parsec sphere at the time the snapshot was taken. This weak correlation could mean that the size of a forming star depends very weakly on the number of stars that are in its neighborhood at the time of formation or that the 0.5 parsec sphere encapsulates a definition of neighborhood too large or too small to detect the connection.

The t-SNE plot displayed the same story of uncorrelated data when coloring by the mass of the stars on the 0.5 pre-stellar snapshot patch data.

In a scenario where you have a large star forming and a smaller star forming a bit later in the large star's proto stellar disc, the gas that is around them is almost entirely the same. In a model trying to predict the final mass of a star, the model will look at two instances of roughly the exact same gas and two different masses. The minor residual difference between the two defined spheres is all the model would have to go off of in differentiating the stars which would also be near the furtherest extreme away from the stars. This problem is futher exasperated by the structure of GMCs in which stars form in filaments of higher density material, packing potential star formation locations closer together. (Use distance weighted properties to create more separation?)

The LASSO-CV model was trained on all of the internal gas feature statistics including the standard deviations as well as the snap, PE_global, and num_stars. The $R^2 = 0.12$ displaying an incredibly poor ability to predict the final mass using a linear model (in log-log space) for the features chosen. Looking at just the single multiplicity stars the $R^2$ improves to 0.23 though this is still a very poor fit and the individual correlations of features is still poor. There is an interesting separation in the [t-SNE of the singles](figures/patch_figures_0.5/tsne_visualization_0.5_patch_singles.pdf) 

##### Next Steps:
* [x] 1. Create a heatmap of the 0.5 patch data ([figures/patch_figures_0.5/correlation_heatmap_0.5_patch.pdf](figures/patch_figures/correlation_heatmap_0.5_patch.pdf))
* [x] 2. Create a t-SNE map of the 0.5 patch features with mass colored ([figures/patch_figures_0.5/tsne_visualization_0.5_patch.pdf](figures/patch_figures_0.5/tsne_visualization_0.5_patch.pdf))
* [x] 3. Create a LASSO-CV model to investigate the linear (in log-log space) perdictability of the 0.5 parsec dataset ([figures/patch_figures_0.5/lasso_cv_0.5_patch.pdf](figures/patch_figures_0.5/lasso_cv_0.5_patch.pdf))
* [x] 4. Investigate singular stars only to control for the "partnered star effect" (in [figures/patch_figures_0.5](figures/patch_figures_0.5))

### 0.1 Parsec Patch Investigation

**7/15** - Upon initial inspection, of the t-SNE, LASSO-CV, and heatmap of the 0.1 parsec patch for both single stars, the predictive power appears to be similar to the 0.5 with a slight increase the LASSO-CV of the single multiplicity only subset. All of the figures can be seen in the folder ([figures/patch_figures_0.1](figures/patch_figures_0.1)).

### Justification of the Use of Medians

**7/15** - If the assumption of a underlying log-normal distribution is true for $B$, $T$, and $\rho$, then the physical meaning of the use of the median is the tracking the mean of the logarithm of the distribution. Due to the extreme skew of a log-normal distribution, the mean is a combination of the underlying mean of the normal distribution in log space and the variance. I believe the uncorrupted nature of the median by the variance of the underlying log distribution yields a more "pure" understanding of the average value of the distribution to the skew strongly influencing the arithmetic mean. However, I am unsure if the assumption of a underlying log-normal holds for these properties.

### Temperature Mass Connection

**7/15** - The Jeans Mass of a star being proportional to temperature to the 3/2 and rho to the -1/2, $M_J \propto T^{3/2} \rho^{-1/2}$, may be the reason we see correlations between the temperature and density of the pre-stellar gas clouds of stars. If a clould of gas is hot and going to turn into a star, it must be massive enough to still collapse under its own gravity. Additionally, if a gas cloud has a low density and is going to turn into a star, the mass making up the cloud must be high. The univariate fit of $M_f \propto \rho_{\text{med}}^{-0.41}$ ($R^2 = 0.48$) is somewhat in line with this analysis. However, the data-driven power-law of $M_f \propto T_{\text{med}}^{0.48}$ ($R^2 = 0.08$) is not well aligned to this conception.  

**7/16** - I looked at the 3D plot of $\rho_{\text{med}}$ vs. $T_{\text{med}}$ vs. $M_f$ in log-space for all components and plotted the plane $M_f =1.5 T_{\text{med}} - 0.5 \rho_{\text{med}}$ (the Jeans mass relation $M_J \propto T^{3/2} \rho^{-1/2}$ in log-space) and the plane visually looks like a floor function for the mass. An intercept in log-space needs to be added to account for the scaling in linear space but I am not sure what that value should be. Regardless, this may be a physical explaination of the success of the reduced model and also explain the kink in the higher mass star prediction if the model is being given information about the floor of the mass from the gas properties but not the ceiling or true value (code is in **residual_analysis.py**).

The nature of the data describing the floor of the mass could also explain why the power-laws do not appear to fall in line with the Jeans Mass assumption directly. The time dependency of this relationship could be investigated to see if the trend survives time steps that are older than the pre-stellar snapshot.

###### Another idea is that if the temperature and density are showing us the floor of the mass, if we subtract this value out what explain the rest of the variance in how much more massive a star will be than its fundamental mass?

##### Next Steps:
* [x] 1. Plot linear regression fit plane on $\rho_{\text{med}}$ and $T_{\text{med}}$
* [x] 2. Investigate coefficents of the two variable linear regression

**7/17** - Comparing the linear regression coefficients of $\rho_{\text{med}}$ and $T_{\text{med}}$ on to $M_f$ in log-space to the theoretical values predicted by the Jeans Mass equation, $\rho_{\text{med}}$ almost precisely aligns with the theoretical value of -0.5 with a value of -0.51. However, $T_{\text{med}}$ does not produce the same clean results, compared to the theoretical 1.5, the coefficient for $T_{\text{med}}$ is 0.95. Though this is closer to the theoretical value than the univariate $T_{\text{med}}$ of 0.48. Regardless, the cosine similarity between the two planes is 0.98 meaning that their normal directions are highly aligned ([image of the 3D plot](figures/jeans_mass_3d.png) with prediction plane in orange and jeans mass plane in blue).

A theoretical reason that could explain the difference in temperature power-law besides the lower bound not being a good predictor of the mean, is a difference in mean molecular weight, $\mu$, of the stars as $M_J \propto \left(\dfrac{T}{\mu}\right)^{1.5}$. So if there is a correlation between $\mu$ and $M_f$ the data-driven power-law for temperature would be affected. 


### 0.01 Parsec Patch All Gas Investigation

**7/16** - Investigating the 0.01 parsec patch around the peak density of the pre-stellar snapshot (0.01b patch) with the same sampling limitations as applied to the 0.01a parsec patch ($M_f > 0.1 M_\odot$ and $N_{\text{inc}} \geq 10$) the $R^2$ value of the LASSO-CV model 0.25 with a similar shape to the original 0.01a parsec patch cut (which had a greater $R^2$ of 0.35). When comparing the 0.01b parsec patch to the 0.1 and 0.5 parsec patchs, the 0.01b (without sampling limitations) LASSO-CV model yields an $R^2$ of 0.27 compared to the 0.1 and 0.5's, 0.18 and 0.12, respectively. There appears to be a trend of increasing performance as the radius is shrunk. It is possible that the optimal radius has been overshot as from the 0.1 parsec patch to the 0.01 parsec patch the sampled volume has been reduced by a factor of 1,000 (or the sampled volume may still larger than the optimal). (figures in [figures/patch_figures_0.01b](figures/patch_figures_0.01b))

##### Next Steps:
* [x] 1. Investigate the performance of the 0.01b patch on singles alone [figures/patch_figures_0.01b/lasso_cv_0.01b_patch_singles.pdf](figures/patch_figures_0.01b/lasso_cv_0.01b_patch_singles.pdf)

**7/16** - The 0.01b patch singles had a LASSO-CV $R^2$ score of 0.36 and unlike other patch models, the CV $\alpha$ weighing the $L_1$ regularization selected only 12 of the 23 features. This performance is very similar to the 0.01a performance with an $R^2$ of 0.35 potentially supporting the "partnered star effect" blurrying the data in patches.

### 7/17 Meeting
1. 0.01b results (improvement from 0.1 and 0.5, still worse than 0.01a, selecting just singles boosts performance - "partner star effect") [figures/patch_figures_0.01b](figures/patch_figures_0.01b) 
2. Kink investigation

   (i) [Random Forest Validation](figures/multiplicity_figures/rand_forest-validation_multiplicity.pdf) (non-linear check)
   
   (ii) [Random Forest Validation Weighted](figures/multiplicity_figures/rand_forest-validation_multiplicity_weighted.pdf) (weighting check)
   
   (iii) If it's not a non-linear feature and not fixed by weighting, it could be that large and massive stars have little distinguishing them.
  
3. Median Value success rationale in log-normal space (median = $\log(\mu)$ in the log-space $\mu$ when $\log(X)$ ~ $N$($\mu$, $\sigma^2$))
4. Jeans Mass explaination of reduced model success ($M_J \propto T^{3/2} \rho^{-1/2}$) - what should the coefficient/intercept be? (**residual_analysis.py**)
5. Multi-target random forest [training](figures/multiplicity_figures/multi_target_rf_train.pdf) - [validation](figures/multiplicity_figures/multi_target_rf_validation.pdf)


# Week 8 Notes

### Jean Mass Analysis Continued

**7/20** - Bootstrapping the full data set to get sample error data on coefficients of $\rho_{\text{med}}$ and $T_{\text{med}}$, $\rho_{\text{med}}$ yielded a 95% confidence interval of (-0.541, -0.496) and a mean of -0.513 falling closely in line with the theoretical -0.5 from the Jean's Mass equation. $T_{\text{med}}$ yielded a 95% confidence interval of (0.905, 0.991) and a mean value of 0.947, leaving the theoretical value of 1.5 far outside of the confidence interval. 

The possible explanations as to why the theoretical power-law and data-driven power-law of $T_{\text{med}}$ are in disagreement; the median could be a poor metric of determining the bulk behavior of the gas and due to the Jeans Mass being a model of the smallest mass a star needs to form, the mass greater than this floor could be dependent on mass. 

Forcing the coefficients of $\rho_{text{med}}$ and $T_{\text{med}}$ to be -0.5 and 1.5, respectively, and fitting an intercept to the data yields and $R^2$ value of 0.65. This means that 65% of the variance in stellar mass can be explained by the relation.

##### Next Steps (Misc):
* [x] 1. Investigate the snap vs. relationship in primary stars
* [x] 2. Investigate patch data on primary stars only  
* [ ] 3. Investigate multi-target model for patch datasets on primaries
* [ ] 4. Investigate at sample weighted version of the 0.01b patch
* [ ] 5. Look at older snapshots for the Jeans Mass 3D analysis
* [ ] 6. Calculate Jeans Mass intercept from astropy

**7/20** - Looking at the connection between snapshot number of sink point formation and final mass in primary stars, there appears to be a very minor bump in the mean mass of stars formed in the middle third of th 500 snapshot period. The spread away from this mean is very large. The code visualizing the relation is in **residual_analysis.py**.

### Primary Stars Analysis

**7/20** - Looking at the Performance of all stars versus only primary stars in the 0.01b, 0.1, and 0.5 parsec patches, there is little difference between the performance with effectively identical $R^2$ values and individual correlation coefficients. In all cases the more restrictive singles only cut of the dataset performs better though this cut removes the majority of stars and all of the largest stars which the primaries cut preserves. These figures can be found in their respective patch folders with directory names starting with 'figures/patch_figures_xxxx'.



