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
df = pd.read_parquet('star_data/accreted_gas_properties_pre_stellar.pq')

# add distance from center metric to the DataFrame
df['distance_from_center'] = np.sqrt((df['x_cm_1']-50)**2 + (df['x_cm_2']-50)**2 + (df['x_cm_3']-50)**2)
df['sp_kin'] = df['E_kin']/df['M_tot']
df['sp_mag'] = df['E_mag']/df['M_tot']
df['sp_int'] = df['E_int']/df['M_tot']
df['L_magnitude'] = np.sqrt(df['L_vec_1']**2 + df['L_vec_2']**2 + df['L_vec_3']**2)

# Investigation of data structure and properties

# Reset index to acccess snap
df = df.reset_index()

#print("Active Columns:", list(df.columns))
#print(df.head().T)

#print(df.shape)

def plot_correlation_heatmap(df):
    '''
    Plots a heatmap of the correlation matrix for the properties of the accreted gas.
    '''

    chosen_log_columns = ['M_tot', 'sp_mag', 'sp_kin', 'sp_int', 'R_eff', 'sig_3D', 'T',
                         'Ne', 'B', 'rho', 'L_magnitude', 'distance_from_center']
    df_log = np.log10(df[chosen_log_columns])
    df_log['snap'] = df['snap']

    correlation_matrix = df_log.corr()
    plt.figure(figsize=(10, 8))
    heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Accreted Gas Properties (all log except snap)')
    plt.savefig('gas_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_intrinsic_gas_heatmap(df):
    '''
    Plots a heatmap of the correlation matrix for the distribution of intrensic gas properties and mass.
    '''
        

    chosen_columns = ['M_tot', 'T', 'T_med', 'T_16', 'T_84', 'T_std', 'Ne', 'Ne_med', 'Ne_16', 'Ne_84', 'Ne_std',
                     'B', 'B_med', 'B_16', 'B_84', 'B_std', 'rho', 'rho_med', 'rho_16', 'rho_84', 'rho_std', 'snap'] 

    correlation_matrix = np.log10(df[chosen_columns]).corr()
    plt.figure(figsize=(12, 10))
    heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Intrinsic Gas Properties')
    plt.tight_layout()
    #plt.savefig('intinsic_prop_correlation_heatmap_numberless.pdf', dpi=300, bbox_inches='tight')
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



def plot_feature_vs_mass(df, feature):
    '''
    Plots the relationship between the electron density of the accreted gas and its total mass.
    '''
    plt.scatter(df[feature], df['M_tot'], alpha=0.5)
    plt.xlabel(feature)
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
    df['sp_kin'] = df['E_kin'] / df['M_tot']
    df['sp_grav'] = df['E_grav'] / df['M_tot']
    df['sp_int'] = df['E_int'] / df['M_tot']
    df['sp_mag'] = df['E_mag'] / df['M_tot']

    # define subplots
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(221)

    #plot kinetic energy vs mass
    ax1.scatter(df['sp_kin'], df['M_tot'], alpha=0.5, s=15)
    ax1.set_xlabel(r'Specific Kinetic Energy (energy/$M_\odot$)')
    ax1.set_ylabel(r'Total Mass ($M_\odot$)')
    ax1.set_title('Specific Kinetic Energy vs Total Mass of Accreted Gas')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # plot kin linear model
    kin_model = LinearRegression()
    kin_model.fit(np.log10(df['sp_kin']).values.reshape(-1, 1), np.log10(df['M_tot']).values)
    kin_score = kin_model.score(np.log10(df['sp_kin']).values.reshape(-1, 1), np.log10(df['M_tot']).values)
    kin_xs = np.linspace(min(np.log10(df['sp_kin'])), max(np.log10(df['sp_kin'])), 100)
    kin_ys = kin_model.predict(kin_xs.reshape(-1, 1))
    ax1.plot(10**kin_xs, 10**kin_ys, linestyle='--', c='r', label=f'fit $r^2$ = {kin_score:.2f}')
    ax1.legend()

    
    #plot gravitational energy vs mass (magnitude due to negative sign)
    ax2 = fig.add_subplot(222)
    ax2.scatter(-df['sp_grav'], df['M_tot'], alpha=0.5, s=15)
    ax2.set_xlabel(r'Specific Gravitational Energy (energy/$M_\odot$)')
    ax2.set_ylabel(r'Total Mass ($M_\odot$)')
    ax2.set_title('Specific Gravitational Energy vs Total Mass of Accreted Gas')
    ax2.set_xscale('log')
    ax2.set_yscale('log')


    #plot internal energy vs mass
    ax3 = fig.add_subplot(223)
    ax3.scatter(df['sp_int'], df['M_tot'], alpha=0.5, s=15)
    ax3.set_xlabel(r'Specific Internal Energy (energy/$M_\odot$)')
    ax3.set_ylabel(r'Total Mass ($M_\odot$)')
    ax3.set_title('Specific Internal Energy vs Total Mass of Accreted Gas')
    ax3.set_xscale('log')
    ax3.set_yscale('log')

    #plot magnetic energy vs mass
    ax4 = fig.add_subplot(224)
    ax4.scatter(df['sp_mag'], df['M_tot'], alpha=0.5, s=15)
    ax4.set_xlabel(r'Specific Magnetic Energy (energy/$M_\odot$)')
    ax4.set_ylabel(r'Total Mass ($M_\odot$)')
    ax4.set_title('Specific Magnetic Energy vs Total Mass of Accreted Gas') 
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    # plot magnetic linear model
    mag_model = LinearRegression()
    mag_model.fit(np.log10(df['sp_mag']).values.reshape(-1, 1), np.log10(df['M_tot']).values)
    mag_score = mag_model.score(np.log10(df['sp_mag']).values.reshape(-1, 1), np.log10(df['M_tot']).values)
    mag_xs = np.linspace(min(np.log10(df['sp_mag'])), max(np.log10(df['sp_mag'])), 100)
    mag_ys = mag_model.predict(mag_xs.reshape(-1, 1))
    ax4.plot(10**mag_xs, 10**mag_ys, linestyle='--', c='r', label=f'fit $r^2$ = {mag_score:.2f}')
    ax4.legend()

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
    print(f"\nr2 values:\nsmall: {small_r2}\nlarge: {large_r2}")
    plt.title(f"Specific energy vs total mass piecewise fit")

    plt.show()


def radians_vs_mass(df):
    '''
    Compares the "angular velocity" to mass of a star
    '''
    # create the "angular velocity" statistic
    df['ang'] = np.sqrt(df['L_vec_1']**2 + df['L_vec_2']**2 + df['L_vec_3']**2)/(df['R_eff'])

    # plot "radians per second" vs mass
    plt.scatter(df['ang'], df['M_tot'], s=10, alpha=0.7)
    plt.xlabel(r'Magnitude of angular momentum divided by $M_{tot}$ and $R_{eff}$')
    plt.ylabel(r'Mass ($M_\odot$)')
    plt.title('\"Angular Velocity\" vs Mass')
    plt.xscale('log')
    plt.yscale('log')

    # fit linear model
    model = LinearRegression()
    model.fit(np.log10(df['ang']).values.reshape(-1, 1), np.log10(df['M_tot']).values)
    r2_score = model.score(np.log10(df['ang']).values.reshape(-1, 1), np.log10(df['M_tot']).values)

    # plot linear fit
    predline_xs = np.linspace(min(df['ang']), max(df['ang']), 100)
    predline_ys = 10**model.predict(np.log10(predline_xs).reshape(-1, 1))
    plt.plot(predline_xs, predline_ys, label=f'Linear fit, $r^2$ = {r2_score:.2f}', linestyle='--', c='r')
    plt.legend()
    print('Power-law slope of the fit:', model.coef_[0])

    plt.show()



if __name__ == "__main__":

    # Parser options for different analyses
    parser = argparse.ArgumentParser(description='Perform data analysis on accreted gas properties.')
    parser.add_argument('--heatmap', "-hm", action='store_true', help='Plot correlation heatmap')
    parser.add_argument('--heatmap_intrinsic', "-hmint", action='store_true', help='Plot intrensic correlation heatmap')
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
    parser.add_argument('--radians_ps', '-rad', action='store_true', help='Plot approx angular velocity vs mass')
    args = parser.parse_args()

    # if statements for each analysis option
    if args.heatmap:
        plot_correlation_heatmap(df)
    if args.heatmap_intrinsic:
        plot_intrinsic_gas_heatmap(df)
    if args.magnetic_energy:
        plot_magnetic_energy_vs_mass(df)
    if args.position:
        plot_position_vs_mass(df)
    if args.dispersion:
        plot_dispersion_vs_mass(df)
    if args.electron_density:
        plot_feature_vs_mass(df)
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
    if args.radians_ps:
        radians_vs_mass(df)
    # If no arguments provided, print a message    
    if sys.argv[1:] == []:
        print("No analysis option provided. Use --help for available options.")
