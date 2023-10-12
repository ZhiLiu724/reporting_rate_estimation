# Estimating reporting rate in crowdsourcing systems

This repository helps reproduce the results of [Liu, Bhandaram, and Garg(2023)](https://arxiv.org/abs/2204.08620) on other crowdsourcing settings. Assuming that reports are generated according to a Poisson process model with unknown rate, we leverage duplicate reports about one incident to learn the reporting rate as a function of incident characteristics, and further estimate the average reporting delay (time between the incident taking place and the first report came in) for each incident.

#### Prerequisite

###### Installing the required packages

```
conda create --name <env> --file requirements.txt
conda activate <env>
```

These commands will create a new `conda` environment with all the packages needed. Tested on a machine with `conda 23.1.0`. 

###### Preparing raw reports data

The raw data containing the reports needs to be minimally processed. Crucially, the following columns are required:

- A column that identifies unique incidents: multiple reports on the same incident should have the same ID along this column. 
- A column that contains the time that each reports were received. 
- A column that contains at least one time stamp for each incident, that we know after this time no reports will be submitted about such incident. This could be the actual resolution time of the incident. 
- Column(s) that indicate the various characteristics of an incident, that may influence the reporting rate. These could be either categorical or continuous.

For an example, see `./sample_data/sample_reports_df.csv`, which contains service requests made to NYC Parks Department on street trees through the NYC311 system.

#### Running the reproduction

See `sample_usage.ipynb` for an extended demonstration. We provide xxx main functions along the reproduction pipeline:

`create_incidents_df`: from the raw reports data, generates a dataframe such that one row corresponds to one incident, and contains the observation interval length for the incident, the number of duplicate reports observed in that interval, and various covariates of the incident.

`prepare_data_for_regression`: using the incidents dataframe, prepares and returns the data that could then be used as input to the Poisson regression model that will be trained.

`train_model`: using the data prepared, trains and returns Poisson regression model (either standard or zero-inflated) to learn the association of each covariate on the reporting rate.

`generate_predicted_reporting_delay`: using the training data and the trained model, estimates a mean reporting delay for each incident.

