import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


# Load your pre-stellar gas data
df = pd.read_parquet('accreted_gas_properties_pre_stellar.pq')

# Select input features 


features = ['Ne', 'T', 'T_med', 'T_84', 'T_16', 'Ne_med', 'Ne_84', 'Ne_16'
            , 'B', 'B_med', 'B_84', 'B_16', 'rho', 'rho_med', 'rho_84', 'rho_16']  

def perform_2D_tsne(df, features):
    '''
    Performs t-SNE on the given DataFrame and plots the results.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - features: list of column names in the DataFrame to be used as features for t-SNE.
    
    Returns:
    - tsne: t-SNE object containing the fitted model.
    - X_tsne: numpy array of shape (n_samples, 2) containing the t-SNE-transformed data.
    '''

    X = np.log(df[features])

    # Standardize features (Crucial: t-SNE relies on Euclidean distances)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create a plotting DataFrame
    df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2'])

    # Plot t-SNE results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_tsne['t-SNE 1'], df_tsne['t-SNE 2'], c=np.log(df['M_tot']), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Total Mass (M_sun) [log scale]')
    plt.title('t-SNE Visualization of Pre-Stellar Gas Properties (features: Ne, T, B, rho)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig('tsne_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return tsne, X_tsne


def perform_3D_tsne(df, features):
    '''
    Performs t-SNE on the given DataFrame and plots the results in 3D.
    
    Parameters:
    - df: pandas DataFrame containing the data to be analyzed.
    - features: list of column names in the DataFrame to be used as features for t-SNE.
    
    Returns:
    - tsne: t-SNE object containing the fitted model.
    - X_tsne: numpy array of shape (n_samples, 3) containing the t-SNE-transformed data.
    '''

    X = np.log(df[features])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create a plotting DataFrame
    df_tsne = pd.DataFrame(X_tsne, columns=['t-SNE 1', 't-SNE 2', 't-SNE 3'])

    # Plot t-SNE results in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_tsne['t-SNE 1'], df_tsne['t-SNE 2'], df_tsne['t-SNE 3'], c=np.log(df['M_tot']), cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Total Mass (M_sun) [log scale]')
    ax.set_title('3D t-SNE Visualization of Pre-Stellar Gas Properties (features: Ne, T, B, rho)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.savefig('tsne_3d_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return tsne, X_tsne


def main():


    perform_2D_tsne(df, features)
    perform_3D_tsne(df, features)


if __name__ == "__main__":
    main()