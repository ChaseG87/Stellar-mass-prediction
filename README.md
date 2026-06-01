This document describes three different data tables with star and gas data from a particular starforge simulation.


# accreted_gas_properties.pq summary.

Each row shows the physical attributes  of gas that will accrete onto a sink (star) particle at a particular moment in time. Gas that has already accreted is not included.

## Table  Overview 
- **Index Key:** `sink_id`
- **Index Key** `snap`
- **Spatial information:** Center of mass positions ($x, y, z$) and velocities ($u, v, w$). Specific angular momentum components.
Velocity dispesion. Principal axes.
- **Average gas properties:** Mass-weighted averages, sample standard deviations (`_std`), and distribution percentiles ($16^{\text{th}}$, $50^{\text{th}}$/median, and $84^{\text{th}}$) for different gas properties (temperature, density, magnetic field, etc.)

## Identification & Bulk Properties

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `sink_id` | `int64` | Dimensionless | Unique identifier of the sink particle (representing a star) within the simulation. |
| `snap` | `int64` | Dimensionless | Snapshot number. Shapshots are spaced by 2.4703e4 years. |
| `M_tot` | `float64` | code_mass ($M_\odot$) | Total mass of the unique, non-feedback gas particles that will accreted by the sink. |
| `R_eff` | `float64` | code_length (pc) | Effective Radius ($R_{\text{eff}}$). This is the radius of the sphere with the same volume as the gas: $R_{\rm eff} = \left(\frac{3 \cdot \text{Vol}}{4\pi}\right)^{1/3}$, where $\text{Vol} = \Sigma \frac{m}{\rho}$. $m$ and $\rho$ are the mass density of each gas particle.|

## Center of Mass & Kinematics

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `x_cm_1` | `float64` | code_length (pc) | x coordinate of the gas center of mass. |
| `x_cm_2` | `float64` | code_length (pc) | y coordinate of the gas center of mass. |
| `x_cm_3` | `float64` | code_length (pc) | z coordinate of the gas center of mass. |
| `v_cm_1` | `float64` | code_velocity (m/s) | x velocity component ($u$) of the gas center of mass. |
| `v_cm_2` | `float64` | code_velocity (m/s) | y velocity component ($v$) of the gas center of mass. |
| `v_cm_3` | `float64` | code_velocity (m/s) | z velocity component ($w$) of the gas center of mass. |
| `L_vec_1` | `float64` | code units | x component of the angular momentum vector (with respect to gas center of mass) |
| `L_vec_2` | `float64` | code units | y component of the angular momentum vector |
| `L_vec_3` | `float64` | code units | z component of the angular momentum vector |
| `sig_3D` | `float64` | code_velocity (m/s) | 3D Velocity Dispersion. |

## Spatial Morphology

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `R_p_1` | `float64` | code_length (pc) | 1st principal component length |
| `R_p_2` | `float64` | code_length (pc) | 2nd principal component length |
| `R_p_3` | `float64` | code_length (pc) | 3rd principal component length |

## Thermodynamics, magnetic field, density

All statistics are mass weighted

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `T` | `float64` | Kelvin (K) | average gas temperature. |
| `T_std` | `float64` | Kelvin (K) | standard deviation of the gas temperature. |
| `T_16` | `float64` | Kelvin (K) | 16th percentile of gas temperature distribution. |
| `T_med` | `float64` | Kelvin (K) | 50th percentile (median) of gas temperature distribution. |
| `T_84` | `float64` | Kelvin (K) | 84th percentile of gas temperature distribution. |
| `B` | `float64` | Gauss (G) | average magnetic field magnitude ($B_{\text{mag}} = \sqrt{B_x^2 + B_y^2 + B_z^2}$). |
| `B_std` | `float64` | Gauss (G) | standard deviation of the magnetic field magnitude. |
| `B_16` | `float64` | Gauss (G) | 16th percentile of magnetic field magnitude. |
| `B_med` | `float64` | Gauss (G) | 50th percentile (median) of magnetic field magnitude. |
| `B_84` | `float64` | Gauss (G) | 84th percentile of magnetic field magnitude. |
| `Ne` | `float64` | Dimensionless | average electron abundance (ratio of free electrons per baryon). |
| `Ne_std` | `float64` | Dimensionless | standard deviation of electron abundance. |
| `Ne_16` | `float64` | Dimensionless | 16th percentile of electron abundance. |
| `Ne_med` | `float64` | Dimensionless | 50th percentile (median) of electron abundance. |
| `Ne_84` | `float64` | Dimensionless | 84th percentile of electron abundance. |
| `rho` | `float64` | code_density | average gas volume density ($\rho$). |
| `rho_std` | `float64` | code_density | standard deviation of gas density. |
| `rho_16` | `float64` | code_density | 16th percentile of gas density. |
| `rho_med` | `float64` | code_density | 50th percentile (median) of gas density. |
| `rho_84` | `float64` | code_density | 84th percentile of gas density. |

## Energetics

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `E_grav` | `float64` | code_units | Gravitational Potential Energy ($E_{\text{pot}}$). Evaluated with  `pytreegrav.Potential` using particle masses, positions, and  smoothing lengths ($h$). |
| `E_kin` | `float64` | code_units | Kinetic Energy ($E_{\text{kin}}$) in the center of mass frame of the gas particles. |
| `E_mag` | `float64` | code_units | Magnetic Energy ($E_{\text{mag}}$). Integrated over local gas cell volumes: $E_{\text{mag}} = \frac{1}{8\pi} \sum B^2 V$. |
| `E_int` | `float64` | code_units | Internal Thermal Energy ($E_{\text{int}}$). A summation of gas masses multiplied by their specific internal energies. |

## Particle Tracking Counters

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `N_inc` | `int64` | Count | Number of unique, non-feedback gas particles that will accrete onto the target sink. |
| `N_fb` | `int64` | Count | Number of feedback gas cells encountered and excluded from analysis. |


# accreted_gas_properties_pre_stellar.pq summary 

Same as accreted_gas_properties.pq, except that there is only one row per star, corresponding to the snapshot before that star appeared.

# first_snap_table.pq 

Table containing initial positions, final masses, and initial masses for all the stars

## Data

| Column Name | Data Type | Physical Units | Field Description |
| :--- | :--- | :--- | :--- |
| `pid` | `int64` |  Dimensionless   | star id |
| `snap` | `int64` | Dimensionless |
| `x` | `float64` | code_length (pc) | x coordinate of star at initial snapshot.  |
| `y` | `float64` | code_length (pc) | y coordinate of star at initial snapshot. |
| `z` | `float64` | code_length (pc) | z coordinate of star at initial snapshot. |
| `mf` | `float64` | code_mass ($M_\odot$) | Final mass of star (at end of simulation or at star death--whichever comes first). |
| `mstar` | `float64` | code_mass ($M_\odot$) | Mass of the star at the initial snapshot |
