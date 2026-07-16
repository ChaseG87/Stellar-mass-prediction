import pandas as pd
import numpy as np
from process_data import log_data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


features = ['T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std'] 

features_reduced = ['rho_med', 'T_med', 'B_med']

# extract full data
df, df_val = log_data(features=features_reduced, train_val=True, multiplicity=True)

# extract and clean patch data
df_cut = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut.pq')
df_cut_val =  pd.read_parquet('star_data/accreted_gas_properties_pre_stellar_cut_test.pq')

df_cut, df_cut_val = log_data(train=df_cut, val=df_cut_val, features=features + ['N_inc'], train_val=True, multiplicity=True)

print(len(df_cut))

# cut out stars that have too low of a mass or too low sample size
df_cut = df_cut[(df_cut['N_inc'] >= 10)  & (10**df_cut['mf'] > 0.1)]
df_cut_val = df_cut_val[(df_cut_val['N_inc'] >= 10)  & (10**df_cut_val['mf'] > 0.1)]

print(len(df_cut))


def multiplicity_tsne(features):
    
    X = df[features]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)

    # Create a plotting DataFrame
    df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['t-SNE 1'], df_tsne['t-SNE 2'], s=10, c=df['max_mult'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=r'Max Multiplicity')
    plt.title(r't-SNE of Pre-Stellar Gas Property Distributions (features: $n_e$, T, B, $\rho$)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('figures/multiplicity_tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()


#multiplicity_tsne(features=features)


def separate_model(
    df=df,
    df_val=df_val,
    features=features_reduced,
    val=False,
    model_type='linear',
    weighting=False
        ):
    
    # format data for sklearn models
    X_train = df[features].values.reshape(-1, len(features))
    y_train = df['mf'].values

    if weighting:
        sample_weight = np.sqrt(10**(y_train))
        weight_str = r'Weighted by $\sqrt{M_f}$ '
    else:
        sample_weight = None
        weight_str = ''


    if model_type == 'linear':
        model = LinearRegression().fit(X=X_train, y=y_train, sample_weight=sample_weight)
    
    elif model_type == 'random forest':
        model = RandomForestRegressor().fit(X=X_train, y=y_train, sample_weight=sample_weight)
    
    plt.figure(figsize=(15, 15))

    c_dict = {1:'b', 2:'g', 3:'c', 4:'y'}

    for mult in range(1, 5):

        if not val:
            X_mult = df[df['max_mult'] == mult][features].values.reshape(-1, len(features))
            y_mult = df[df['max_mult'] == mult]['mf'].values
        else:
            X_mult = df_val[df_val['max_mult'] == mult][features].values.reshape(-1, len(features))
            y_mult = df_val[df_val['max_mult'] == mult]['mf'].values

        pred = model.predict(X=X_mult)
        r2 = r2_score(y_mult, pred)

        ax = plt.subplot(2, 2, mult)

        ax.scatter(y_mult, pred, s=10, alpha=0.5, c=c_dict[mult])
        ax.plot(y_train, y_train, c='r')
        ax.set_title(f'Multiplicity = {mult}, $R^2 =$ {r2:.2f}')
        ax.set_xlabel(r'Ground Truth Mass ($\log_{10}M_\odot$)')
        ax.set_ylabel(r'Predicted Mass ($\log_{10}M_\odot$)')

    if not val:
        plt.suptitle(model_type + r' on $\rho_{med}$, $T_{med}$, and $B_{med}$ Separated by Multiplicity ' + weight_str + '(Training Data)')
        #plt.suptitle(r'Linear Regression on 0.01 Parsec Patch Separated by Multiplicity (Training Data)')
        plt.tight_layout(pad=1.5)
        #plt.savefig('figures/multiplicity_figures/rand_forest_train_multiplicity_weighted.pdf', dpi=300)
    else:
        plt.suptitle(model_type + r' on $\rho_{med}$, $T_{med}$, and $B_{med}$ Separated by Multiplicity ' + weight_str + '(Validation Data)')
        #plt.suptitle(r'Linear Regression on 0.01 Parsec Patch Separated by Multiplicity (Validation Data)')
        plt.tight_layout(pad=1.5)
        #plt.savefig('figures/multiplicity_figures/rand_forest_validation_multiplicity_weighted.pdf', dpi=300)

    plt.show()


# plots training and validation for full data separated by multiplicity linear
#separate_model(df=df, features=features_reduced, model_type='linear')
#separate_model(df=df, val=True, features=features_reduced, model_type='linear')

# plots training and validation for patch data separated by multiplicity
#separate_linear_model(df=df_cut, features=features)
#separate_linear_model(df=df_cut, df_val=df_cut_val, val=True, features=features)

# plots training and validation for full data separated by multiplicity random forest
separate_model(df=df, features=features_reduced, model_type='random forest', weighting=True)
separate_model(df=df, val=True, features=features_reduced, model_type='random forest', weighting=True)
