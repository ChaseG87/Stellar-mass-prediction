import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from process_data import log_train_val_data, normalize_data
from data_analysis_v1 import plot_intrinsic_gas_heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'snap'] 

# read file
df_cut = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut.pq').reset_index()

print(df_cut['max_dist_peak'][df_cut['max_dist_peak'] < 0.005].count())

df_clean_cut = df_cut[(df_cut['N_inc'] >= 10) & (df_cut['max_dist_peak'] > 0.005)]

# log taken within the function
plot_intrinsic_gas_heatmap(df=df_clean_cut)

# There are 108 stars that have less than 10 particles
# 65 of those have a max_dist_peak less than 0.005 pcs

# take log of both
X = np.log10(df_clean_cut[features].values.reshape(-1, 21))
y = np.log10(df_clean_cut['M_tot'].values)


# standardize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LassoCV(cv=5, max_iter=10000).fit(X, y)

pred = model.predict(X)

r2 = model.score(X, y)

used_features = [features[i] for i in range(21) if model.coef_[i]]


# plot prediction
plt.figure(figsize=(10, 8))
plt.scatter(y, pred, s=10, alpha=0.5)
plt.plot(y, y, c='r')

plt.title(f'LASSO-CV applied to 0.01 pc cut, $R^2$ = {r2:.2f}, num_features = {len(used_features)}')

plt.xlabel(r'Ground Truth Mass ($\log_{10} M_\odot$)')
plt.ylabel(r'Prediction ($\log_{10} M_\odot$)')

plt.savefig('lasso_cv_0.01_cut.pdf', dpi=300)
plt.show()




