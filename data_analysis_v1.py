import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from seaborn import heatmap
import numpy as np
import sys
import argparse

"""
Investigation of the properties of the accreted gas at the snapshot before star formation.

Namespace Dependencies:
- pandas
- matplotlib
- scikit-learn
- numpy
- seaborn
- argparse
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

    chosen_columns = ['M_tot', 'E_mag', 'E_kin', 'E_int', 'R_eff', 'sig_3D', 'T',
                         'L_vec_1', 'L_vec_2', 'L_vec_3']

    correlation_matrix = np.log10(df[chosen_columns]).corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Accreted Gas Properties')
    plt.savefig('gas_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_magnetic_energy_vs_mass(df):
    '''
    Plots the relationship between magnetic energy and total mass of accreted gas on a log-log scale.
    '''
    df['sp_mag'] = df['E_mag'] / df['M_tot']


    # Plot scatter of comparision
    plt.scatter(np.log10(df['sp_mag']), np.log10(df['M_tot']), alpha=0.5)
    plt.xlabel('Specific Magnetic Energy')
    plt.ylabel(r'Total Mass ($M_\odot$)')
    plt.title('Specific Magnetic Energy vs Total Mass of Accreted Gas')

    # Fit a linear model to the log-log data to find the power-law relationship
    log_sp_mag = np.log10(df['sp_mag'])
    log_M_tot = np.log10(df['M_tot'])

    model = LinearRegression()
    model.fit(log_sp_mag.values.reshape(-1, 1), log_M_tot.values)
    x_fit = np.linspace(log_sp_mag.min(), log_sp_mag.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(x_fit, y_fit, color='red', label=f'Linear Fit - power law = {model.coef_[0]:.2f}', linestyle='--')
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

        # Fit a linear model to the log-log data to find the power-law relationship
    log_sig_3D = np.log10(df['sig_3D'])
    log_M_tot = np.log10(df['M_tot'])

    model = LinearRegression()
    model.fit(log_sig_3D.values.reshape(-1, 1), log_M_tot.values)
    x_fit = np.linspace(log_sig_3D.min(), log_sig_3D.max(), 100)
    y_fit = model.predict(x_fit.reshape(-1, 1))
    plt.plot(np.exp(x_fit), np.exp(y_fit), color='red', label=f'Linear Fit - power law = {model.coef_[0]:.2f}', linestyle='--')
    plt.legend()
    plt.show()



def plot_Ne_vs_mass(df):
    '''
    Plots the relationship between the electron density of the accreted gas and its total mass.
    '''
    plt.scatter(df['Ne'], df['M_tot'], alpha=0.5)
    plt.xlabel('Electron Density (free electrons per baryon)')
    plt.ylabel('Total Mass (M_sun)')
    plt.title('Electron Density vs Total Mass of Accreted Gas')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_star_locations(df):
    '''
    Plots the 3D positions of the accreted gas particles, colored by their total mass.
    '''

    #df = df[np.log10(df['M_tot']) > 0]  # Filter out small stars to see the distribution of more massive stars more clearly

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['x_cm_1'], df['x_cm_2'], df['x_cm_3'], s=3, c=np.log10(df['M_tot']), cmap='viridis', alpha=0.5)
    plt.colorbar(sc, label='Total Mass (M_sun) [log scale]')
    ax.set_xlabel('X Position (pc)')
    ax.set_ylabel('Y Position (pc)')
    ax.set_zlabel('Z Position (pc)')
    plt.title('3D Positions of Accreted Gas Particles Colored by Total Mass')
    plt.show()


def plot_temperature_vs_ne(df):
    '''
    Plots the relationship between the temperature of the accreted gas and its electron density.
    '''
    plt.scatter(df['T_med'], df['Ne_med'], alpha=0.5)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Electron Density (free electrons per baryon)')
    plt.title('Temperature vs Electron Density of Accreted Gas')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_specific_energies_vs_mass(df):
    '''
    Plots the relationship between the specific energy of the accreted gas and its total mass.
    '''
    specific_kinetic = df['E_kin'] / df['M_tot']
    specific_gravitational = df['E_grav'] / df['M_tot']
    specific_internal = df['E_int'] / df['M_tot']
    specific_magnetic = df['E_mag'] / df['M_tot']

    # define subplots
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)

    #plot kinetic energy vs mass
    ax1.scatter(specific_kinetic, df['M_tot'], alpha=0.5)
    ax1.set_xlabel(r'Specific Kinetic Energy (energy/$M_\odot$)')
    ax1.set_ylabel(r'Total Mass ($M_\odot$)')
    ax1.set_title('Specific Kinetic Energy vs Total Mass of Accreted Gas')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    #plot gravitational energy vs mass (magnitude due to negative sign)
    ax2 = fig.add_subplot(222)
    ax2.scatter(-specific_gravitational, df['M_tot'], alpha=0.5)
    ax2.set_xlabel(r'Specific Gravitational Energy (energy/$M_\odot$)')
    ax2.set_ylabel(r'Total Mass ($M_\odot$)')
    ax2.set_title('Specific Gravitational Energy vs Total Mass of Accreted Gas')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    #plot internal energy vs mass
    ax3 = fig.add_subplot(223)
    ax3.scatter(specific_internal, df['M_tot'], alpha=0.5)
    ax3.set_xlabel(r'Specific Internal Energy (energy/$M_\odot$)')
    ax3.set_ylabel(r'Total Mass ($M_\odot$)')
    ax3.set_title('Specific Internal Energy vs Total Mass of Accreted Gas')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    #plot magnetic energy vs mass
    ax4 = fig.add_subplot(224)
    ax4.scatter(specific_magnetic, df['M_tot'], alpha=0.5)
    ax4.set_xlabel(r'Specific Magnetic Energy (energy/$M_\odot$)')
    ax4.set_ylabel(r'Total Mass ($M_\odot$)')
    ax4.set_title('Specific Magnetic Energy vs Total Mass of Accreted Gas') 
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    plt.savefig('specific_energies_vs_mass.pdf', dpi=300, bbox_inches='tight')

    plt.show()


def plot_strong_correlation(df):
    '''
    Plots the relationship between two properties of the accreted gas that show a strong correlation in the heatmap.
    '''

    # Calculate specific kinetic and magnetic energies
    df['sp_kin'] = df['E_kin'] / df['M_tot']
    df['sp_mag'] = df['E_mag'] / df['M_tot']

    # Generate and plot the correlation heatmap for found strong correlations
    correlation_matrix = np.log10(df[['M_tot','sp_kin', 'sp_mag', 'sig_3D', 'Ne', 'Ne_med']]).corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Accreted Gas Properties')
    plt.show()


def plot_velocity_correlation(df):
    '''
    Plots the relationship between the velocity dispersion of the accreted gas and its specific kinetic energy.
    '''

    # Generate and plot the correlation heatmap for found strong correlations
    correlation_matrix = np.log10(df[['M_tot', 'sig_3D', 'v_cm_1', 'v_cm_2', 'v_cm_3', 'L_vec_1', 'L_vec_2', 'L_vec_3']]).corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of velocities and mass')
    plt.show()


def morphology_correlation(df):
    '''
    Plots the relationship between the morphology of the accreted gas and its total mass.
    '''

    df['stretch1'] = df['R_p_1'] / df['R_p_2']
    df['stretch2'] = df['R_p_3'] / df['R_p_2']


    correlation_matrix = np.log10(df[['M_tot', 'R_eff', 'stretch1', 'stretch2','R_p_1', 'R_p_2', 'rho']]).corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Morphology')
    plt.show()


def piecewise_sp_eng_fit(df, energy_type='mag'):
    
    if energy_type not in ['kin', 'mag']:
        raise ValueError("energy_type must either be kinetic, \"kin\", or magnetic, \"mag\".")

    # Define specific energies
    df['sp_kin'] = df['E_kin'] / df['M_tot']
    df['sp_mag'] = df['E_mag'] / df['M_tot']

    energy_string = 'sp_' + energy_type

    # extract the subset of large specific energies
    large_stars_x = df[energy_string][df[energy_string] > 10**5]
    large_stars_y = df['M_tot'][df[energy_string] > 10**5]

    # extract the subset of small specific energies
    small_stars_x = df[energy_string][df[energy_string] <= 10**5]
    small_stars_y = df['M_tot'][df[energy_string] <= 10**5]

    # initialize and fit the model for the large specific energies
    model_large = LinearRegression()
    model_large.fit(np.log10(large_stars_x).values.reshape(-1, 1), np.log10(large_stars_y).values)
    large_r2 = model_large.score(np.log10(large_stars_x).values.reshape(-1, 1), np.log10(large_stars_y).values)

    # initialize and fit the model for the small specific energies
    model_small = LinearRegression()
    model_small.fit(np.log10(small_stars_x).values.reshape(-1, 1), np.log10(small_stars_y).values)
    small_r2 = model_small.score(np.log10(small_stars_x).values.reshape(-1, 1), np.log10(small_stars_y).values)


    # plot the scatter plot and lines of the specific energies vs total mass
    plt.scatter(df[energy_string], df['M_tot'], s=10, alpha=0.7)

    large_dash_x = np.linspace(10**5, max(large_stars_x), 50)
    large_dash_y = 10**model_large.predict(np.log10(large_dash_x).reshape(-1, 1))
    plt.plot(large_dash_x, large_dash_y, linestyle='--', c='r')

    small_dash_x = np.linspace(min(small_stars_x), 10**5, 50)
    small_dash_y = 10**model_small.predict(np.log10(small_dash_x).reshape(-1, 1))
    plt.plot(small_dash_x, small_dash_y, linestyle='--', c='r')
    plt.xscale('log')
    plt.yscale('log')
    print(f'large_slope = {model_large.coef_[0]}\nsmall_slope = {model_small.coef_[0]}')
    print(f"r2 values\nsmall: {small_r2}\nlarge: {large_r2}")
    plt.title(f"Specific energy vs total mass piecewise fit")

    plt.show()


if __name__ == "__main__":

    # Parser options for different analyses
    parser = argparse.ArgumentParser(description='Perform data analysis on accreted gas properties.')
    parser.add_argument('--heatmap', "-hm", action='store_true', help='Plot correlation heatmap')
    parser.add_argument('--magnetic_energy', "-me", action='store_true', help='Plot magnetic energy vs mass')
    parser.add_argument('--position', "-pos", action='store_true', help='Plot position vs mass')
    parser.add_argument('--dispersion', "-disp", action='store_true', help='Plot velocity dispersion vs mass')
    parser.add_argument('--electron_density', "-ne", action='store_true', help='Plot electron density vs mass')
    parser.add_argument('--star_locations', "-sl", action='store_true', help='Plot 3D positions of accreted gas particles colored by total mass')
    parser.add_argument('--temperature_vs_ne', "-temp", action='store_true', help='Plot temperature vs electron density')
    parser.add_argument('--specific_energies', "-se", action='store_true', help='Plot specific energies vs mass')
    parser.add_argument('--strong_correlation', "-sc", action='store_true', help='Plot properties with strong correlation')
    parser.add_argument('--velocity_correlation', "-vc", action='store_true', help='Plot velocity properties correlation')
    parser.add_argument('--morphology_correlation', "-mc", action='store_true', help='Plot morphology properties correlation')
    parser.add_argument('--piecewise_energy_fit', '-pw', action='store_true', help='Plot piecewise fit of specific energy')
    args = parser.parse_args()

    # if statements for each analysis option
    if args.heatmap:
        plot_correlation_heatmap(df)
    if args.magnetic_energy:
        plot_magnetic_energy_vs_mass(df)
    if args.position:
        plot_position_vs_mass(df)
    if args.dispersion:
        plot_dispersion_vs_mass(df)
    if args.electron_density:
        plot_Ne_vs_mass(df)
    if args.star_locations:
        plot_star_locations(df)
    if args.temperature_vs_ne:
        plot_temperature_vs_ne(df)
    if args.specific_energies:
        plot_specific_energies_vs_mass(df)
    if args.strong_correlation:
        plot_strong_correlation(df)
    if args.velocity_correlation:
        plot_velocity_correlation(df)
    if args.morphology_correlation:
        morphology_correlation(df)
    if args.piecewise_energy_fit:
        piecewise_sp_eng_fit(df)

    # If no arguments provided, print a message    
    if sys.argv[1:] == []:
        print("No analysis option provided. Use --help for available options.")
