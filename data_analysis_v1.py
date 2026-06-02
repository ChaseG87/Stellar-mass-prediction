import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from seaborn import heatmap
import numpy as np

"""
Investigation of the properties of the accreted gas at the snapshot before star formation.

Namespace Dependencies:
- pandas
- matplotlib
- scikit-learn
- numpy
- seaborn
- pyarrow/fastparquet (for reading parquet files)
"""

# Read the pre stellar parquet file into a DataFrame
df = pd.read_parquet('accreted_gas_properties_pre_stellar.pq')

# add distance from center metric to the DataFrame
df['distance_from_center'] = np.sqrt((df['x_cm_1']-50)**2 + (df['x_cm_2']-50)**2 + (df['x_cm_3']-50)**2)

# Investigation of data structure and properties
#print(df.head().T)
#print(df.shape)

def plot_correlation_heatmap(df):
    '''
    Plots a heatmap of the correlation matrix for the properties of the accreted gas.
    '''

    chosen_columns = ['M_tot', 'E_mag', 'R_eff', 'sig_3D', 'T', 'T_med', 'T_16', 'T_84',
                       'distance_from_center', 'B', 'Ne', 'rho', 'Ne_med']

    correlation_matrix = np.log(df[chosen_columns]).corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Accreted Gas Properties')
    plt.show()


def plot_magnetic_energy_vs_mass(df):
    '''
    Plots the relationship between magnetic energy and total mass of accreted gas on a log-log scale.
    '''

    # Plot scatter of comparision
    plt.scatter(df['E_mag'], df['M_tot'], alpha=0.5)
    plt.xlabel('Magnetic Energy (erg)')
    plt.ylabel('Total Mass (M_sun)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Magnetic Energy vs Total Mass of Accreted Gas')

    # Fit a linear model to the log-log data to find the power-law relationship
    log_E_mag = np.log(df['E_mag'])
    log_M_tot = np.log(df['M_tot'])

    model = LinearRegression()
    model.fit(log_E_mag.values.reshape(-1, 1), log_M_tot.values)
    x_fit = np.linspace(log_E_mag.min(), log_E_mag.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(np.exp(x_fit), np.exp(y_fit), color='red', label=f'Linear Fit - power law = {model.coef_[0]:.2f}', linestyle='--')
    plt.legend()
    plt.show()


def plot_position_vs_mass(df):
    '''
    Plots the relationship between the position of the accreted gas and its total mass.
    '''
    plt.scatter(df['distance_from_center'], df['M_tot'], alpha=0.5)
    plt.xlabel('Distance from Center (pc)')
    plt.ylabel('Total Mass (M_sun)')
    plt.title('Position vs Total Mass of Accreted Gas')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


def plot_dispersion_vs_mass(df):
    '''
    Plots the relationship between the velocity dispersion of the accreted gas and its total mass.
    '''
    plt.scatter(df['sig_3D'], df['M_tot'], alpha=0.5)
    plt.xlabel('Velocity Dispersion (m/s)')
    plt.ylabel('Total Mass (M_sun)')
    plt.title('Velocity Dispersion vs Total Mass of Accreted Gas')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_Ne_vs_mass(df):
    '''
    Plots the relationship between the electron density of the accreted gas and its total mass.
    '''
    plt.scatter(df['Ne_med'], df['M_tot'], alpha=0.5)
    plt.xlabel('Electron Density (free electrons per baryon)')
    plt.ylabel('Total Mass (M_sun)')
    plt.title('Electron Density vs Total Mass of Accreted Gas')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


#plot_magnetic_energy_vs_mass(df)
#plot_position_vs_mass(df)
#plot_dispersion_vs_mass(df)
plot_Ne_vs_mass(df)
plot_correlation_heatmap(df)
