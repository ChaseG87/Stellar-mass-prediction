import numpy as np
import pandas as pd
from process_data import log_data
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.manifold import TSNE

# define features
features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
        'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'PE_global', 'snap', 'num_stars'] 

df_patch = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_0.5_patch.pq').reset_index()

# flip sign of potential energy to be strictly positive
df_patch ['PE_global'] *= -1

# add 1 so 0 star regions don't throw log errors (count the forming star)
df_patch['num_stars'] += 1 

# log and normalize data
df_patch_log = log_data(features=features, train=df_patch, train_val=False, normalize=True)

df_patch = df_patch.set_index('sink_id')

# reshape data into numpy arrays for regression
X = df_patch_log[features].values.reshape(-1, len(features))
y = df_patch_log['mf'].values

def correlation_heatmap(df):

    correlation_matrix = df_patch_log.corr()
    plt.figure(figsize=(15, 12))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Accreted Gas Properties (all log)')

    plt.tight_layout()
    plt.savefig('figures/patch_figures_0.5/correlation_heatmap_0.5_patch.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#correlation_heatmap(df=df_patch_log)

def plot_tsne(df):

    X = df[features]


    # Apply t-SNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    # Create a plotting DataFrame
    df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['t-SNE 1'], df_tsne['t-SNE 2'], s=10, c=df['mf'], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label=r'Final Mass ($M_\odot$) [log scale]')
    plt.title(r't-SNE of 0.5 Pcs Patch Properties (features: $n_e$, T, B, $\rho$, $PE_{global}$, snap, num_stars)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('figures/patch_figures_0.5/tsne_visualization_0.5_patch.pdf', dpi=300, bbox_inches='tight')
    plt.show()

#plot_tsne(df=df_patch_log)

def plot_lasso_cv(X_train, y_train):

    model = LassoCV(cv=5, max_iter=10000).fit(X_train, y_train)
    coefs = model.coef_
    chosen_features = [features[i] for i in range(len(features)) if coefs[i]]

    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)

    print(chosen_features)

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred, s=10, alpha=0.5)
    plt.plot(y_train, y_train, c='r')
    plt.xlabel(r'Ground Truth Mass [$\log_{10}M_\odot$]')
    plt.ylabel(r'Predicted Mass [$\log_{10}M_\odot$]')
    plt.title(f'LASSO-CV model of 0.5 parsec patch - $R^2 =$ {r2:.2f} - features = {len(chosen_features)}')
    plt.savefig('figures/patch_figures_0.5/lasso_cv_0.5_patch.pdf', dpi=300)

    plt.show()

plot_lasso_cv(X_train=X, y_train=y)

    