# Estimating reporting rate in crowdsourcing systems [![DOI](https://zenodo.org/badge/700630223.svg)](https://zenodo.org/doi/10.5281/zenodo.10086345)

This repository contains code to apply the method of [Liu, Bhandaram, and Garg(2023)](https://arxiv.org/abs/2204.08620) to other crowdsourcing settings. If you use this package in a research paper, please cite the above paper (and email us -- we'd love to know!).

Assuming that reports are generated according to a Poisson process model with unknown rate, we leverage duplicate reports about one incident to learn the reporting rate as a function of incident characteristics, and further estimate the average reporting delay (time between the incident taking place and the first report came in) for each incident.

#### Usage

###### Installing the required packages

```
conda create --name <env> --file requirements.txt
conda activate <env>
```

These commands will create a new `conda` environment with all the packages needed. Tested on a machine with `conda 23.1.0`. 

###### Preparing raw reports data
You can start with a data frame that contains the raw reports. Each report is a row, and the following columns are required:

- A column that identifies unique incidents: multiple reports on the same incident should have the same ID along this column. 
- A column that contains the time that each report was received. 
- A column that contains at least one time stamp for each incident, that we know after this time no reports will be submitted about such incident. This could be the actual resolution time of the incident. If such a column is not available, you may add a constant time length after the first report for that unique incident.
- Column(s) that indicate the various characteristics of an incident, that may influence the reporting rate. These could be either categorical or continuous.

For an example, see `./sample_data/sample_reports_df.csv`, which contains service requests made to NYC Parks Department on street trees through the NYC311 system.

#### Applying the method

See `sample_usage.ipynb` for an extended demonstration. We provide 4 main functions along the usage pipeline:

`create_incidents_df`: from the raw reports data, generates a dataframe such that one row corresponds to one incident, and contains the observation interval length for the incident, the number of duplicate reports observed in that interval, and various covariates of the incident.

`prepare_data_for_regression`: using the incidents dataframe, prepares and returns the data that could then be used as input to the Poisson regression model that will be trained.

`train_model`: using the data prepared, trains and returns Poisson regression model (either standard or zero-inflated) to learn the association of each covariate on the reporting rate.

`generate_predicted_reporting_delay`: using the training data and the trained model, estimates a mean reporting delay for each incident.

