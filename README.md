# Stellar-mass-prediction
This is a working repository analyzing relationship between pre-stellar gas properties and the final mass of stars using STARFORGE simulations of GMCs.

## Files and Usage

#### star_data/data_properties.pdf
Contains a breakdown of the specific contents of stellar data provided in different parquet files.

#### star_data/accreted_gas_properties_pre_stellar.pq
Parquet file containing pre-stellar gas properties and statistics associated with each star the snapshot before a sink point is produced by that star in the STARFORGE simulation.
This simulation is used as the training data for predictive models and data analysis.

#### star_data/accreted_gas_properties_pre_stellar_test.pq
Parquet file containing pre-stellar gas properties and statisics associated with each star the snapshot before a sink point is produced by that star in a STARFORGE simulation with 
the same hyperparmeters but different initial conditions. This dataset is used as the validation set for predictive models.
