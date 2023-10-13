import pandas as pd
from patsy import dmatrices
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def calculate_observation_start_end_time(df, 
                                        incident_identifier_col, 
                                        observation_start_col,
                                        observation_start_agg_method,
                                        observation_end_col, 
                                        observation_end_agg_method, 
                                        max_duration):
    """
    Calculate the observation end time for each incident.
    ----------
    Parameters
    ----------
    df : pandas.DataFrame
        the raw reports dataframe
    incident_identifier_col : list of str
        the column name for the incident identifier
    observation_end_col : list of str
        the column name for the observation end time
    max_duration : str
        the maximum duration of the observation interval, in days
    """
    cols_to_agg = observation_start_col + observation_end_col

    agg_methods = observation_start_agg_method + observation_end_agg_method
    agg_dict = dict(zip(cols_to_agg, agg_methods))

    grouped = df.groupby(incident_identifier_col).agg(agg_dict).reset_index()

    # the observation start time is the minimum of the observation start time columns
    grouped['observation_start_time'] = grouped[observation_start_col].min(axis=1)

    # add a columns for max duration, then calculate the observation end time
    grouped['max_duration'] = grouped['observation_start_time'] + pd.Timedelta(max_duration, unit='d')
    observation_end_col.append('max_duration')
    grouped['observation_end_time'] = grouped[observation_end_col].min(axis=1)

    # drop the columns that are no longer needed
    cols_to_keep = incident_identifier_col +['observation_start_time', 'observation_end_time']
    grouped = grouped[cols_to_keep]

    return grouped


def create_incidents_df(reports_df,
                        incident_identifier_col,
                        observation_start_col,
                        observation_end_col, 
                        reports_identifier_col = None,
                        observation_start_agg_method = None,
                        observation_end_agg_method = None,
                        max_duration = 100,
                        covariates_cont = None,
                        covariates_cont_agg_method = None,
                        covariates_cat = None,
                        covariates_cat_agg_method = None,
                        dropna = True,
                        filter_short_duration = True,
                        standardize_cont = True):
    """
    Create the incidents dataframe from the raw reports dataframe.
    ----------
    Parameters
    ----------
    reports_df: pandas.DataFrame
        the raw reports dataframe
    reports_identifier_col: str, or list of str
        the column name of the report identifier; if not specified, the index of the dataframe will be used
    incident_identifier_col: str, or list of str
        the column name of the incident identifier, needs to be specified to identify unique incidents
    observation_start_col: str, or list of str
        the column name of the observation start time, needs to be specified to identify start time of observation interval, within a unique incident
    observation_start_agg_method: str, or list of str
        the aggregation method for the observation start time, available options are 'min', 'max', 'mean', 'median', 'first', 'last'
        if not specified, 'min' will be used for all columns
    observation_end_col: str, or list of str
        the column name of the observation end time, needs to be specified to identify end time of observation interval, within a unique incident
    observation_end_agg_method: str, or list of str
        the aggregation method for the observation end time, available options are 'min', 'max', 'mean', 'median', 'first', 'last'
        if not specified, 'min' will be used for all columns
    max_duration: int
        the maximum duration of the observation interval in days, default is 100 days
    covariates_cont: str, or list of str
        the column names of the continuous covariates included in the model
    covariates_cont_agg_method: str, or list of str
        the aggregation method for the continuous covariates, available options are 'min', 'max', 'mean', 'median', 'first', 'last'
        if not specified, 'first' will be used for all columns
    covariates_cat: str, or list of str
        the column names of the categorical covariates included in the model
    covariates_cat_agg_method: str, or list of str
        the aggregation method for the categorical covariates, available options are 'first', 'last', 'mode'
        if not specified, 'first' will be used for all columns
    dropna: bool
        whether to drop rows with missing values in the covariates columns, before aggregating the covariates
    filter_short_duration: bool or float
        whether to filter out incidents with duration shorter than 0.1 day, or the specified duration in days
    """

    assert isinstance(reports_df, pd.DataFrame), "reports_df must be a pandas.DataFrame"
    assert isinstance(incident_identifier_col, list) or isinstance(incident_identifier_col, str),  "incident_identifier_col must be a string or a list of strs"
    if isinstance(incident_identifier_col, str):
        incident_identifier_col = [incident_identifier_col]

    assert isinstance(observation_start_col, list) or isinstance(observation_start_col, str),"observation_start_col must be a str or list of strs"
    if isinstance(observation_start_col, str):
        observation_start_col = [observation_start_col]

    if observation_start_agg_method is None:
        observation_start_agg_method = ['min'] * len(observation_start_col)
    
    assert isinstance(observation_end_col, list) or isinstance(observation_end_col, str), "observation_end_col must be a str or list of strs"
    if isinstance(observation_end_col, str):
        observation_end_col = [observation_end_col]
    
    if observation_end_agg_method is None:
        observation_end_agg_method = ['min'] * len(observation_end_col)
    
    assert isinstance(max_duration, int), "max_duration must be an integer"
    assert covariates_cat is not None or covariates_cont is not None, "at least one covariate must be specified"

    if covariates_cont is not None:
        assert isinstance(covariates_cont, list) or isinstance(covariates_cont, str), "covariates_cont must be a str or list of strs"
        if isinstance(covariates_cont, str):
            covariates_cont = [covariates_cont]
        if covariates_cont_agg_method is None:
            covariates_cont_agg_method = ['first'] * len(covariates_cont)
    
    if covariates_cat is not None:
        assert isinstance(covariates_cat, list) or isinstance(covariates_cat, str), "covariates_cat must be a str or list of strs"
        if isinstance(covariates_cat, str):
            covariates_cat = [covariates_cat]
        if covariates_cat_agg_method is None:
            covariates_cat_agg_method = ['first'] * len(covariates_cat)
    
    assert isinstance(dropna, bool), "dropna must be a boolean"
    assert isinstance(filter_short_duration, bool) or isinstance(filter_short_duration, int) or isinstance(filter_short_duration, float), "filter_short_duration must be a boolean, integer, or float"

    assert standardize_cont <= dropna, "only if dropna is True, standardize_cont can be True"


    # if reports_identifier_col is not specified, use the index of the dataframe
    if reports_identifier_col is None:
        reports_df = reports_df.reset_index()
        reports_identifier_col = 'index'

    # calculate the observation start and end time for each incident
    df = reports_df.copy()
    grouped = calculate_observation_start_end_time(df, 
                                                   incident_identifier_col, 
                                                   observation_start_col,
                                                   observation_start_agg_method,
                                                   observation_end_col, 
                                                   observation_end_agg_method, 
                                                   max_duration)

    df = df.merge(grouped, on=incident_identifier_col, how='left')

    # evaluate if the report is within the observation interval
    df['is_within_observation_interval'] = (df['SRCreatedDate'] > df['observation_start_time']) & (df['SRCreatedDate'] <= df['observation_end_time'])

    # aggregate the incidents
    dfmain = df.groupby(incident_identifier_col).agg({'is_within_observation_interval': 'sum',
                                                   'observation_start_time': 'first',
                                                   'observation_end_time': 'first'}).reset_index()
    dfmain.rename(columns={'is_within_observation_interval': 'num_duplicates'}, inplace=True)
    dfmain['num_duplicates'] = dfmain['num_duplicates'].astype(int)
    dfmain['duration'] = (dfmain['observation_end_time'] - dfmain['observation_start_time']).dt.total_seconds() / (24 * 60 * 60)
    if isinstance(filter_short_duration, bool):
        if filter_short_duration is True:
            dfmain = dfmain.query('duration > 0.1')
    elif isinstance(filter_short_duration, (int, float)):
        dfmain = dfmain.query('duration > @filter_short_duration')
    dfmaincols_to_keep = incident_identifier_col + ['num_duplicates', 'duration']
    dfmain = dfmain[dfmaincols_to_keep]


    # aggregate the continuous covariates 
    if covariates_cont is not None:
        dfcov_cont = df.groupby(incident_identifier_col).agg(dict(zip(covariates_cont, covariates_cont_agg_method))).reset_index()
        dfmain = dfmain.merge(dfcov_cont, on=incident_identifier_col, how='left')
    if covariates_cat is not None:
        dfcov_cat = df.groupby(incident_identifier_col).agg(dict(zip(covariates_cat, covariates_cat_agg_method))).reset_index()
        dfmain = dfmain.merge(dfcov_cat, on=incident_identifier_col, how='left')
    if dropna:
        dfmain.dropna(inplace=True)
        if standardize_cont:
            dfmain[covariates_cont] = (dfmain[covariates_cont] - dfmain[covariates_cont].mean()) / dfmain[covariates_cont].std()
    
    return dfmain


def prepare_data_for_regression(incidents_df, 
                                covariate_cat, 
                                covariate_cont):
    """
    Prepare the data for regression.
    ----------
    Parameters
    ----------
    incidents_df: pandas.DataFrame
        the incidents dataframe, generated by create_incidents_df
    covariate_cont: str, or list of str
        the column names of the continuous covariates included in the model
    covariate_cat: str, or list of str
        the column names of the categorical covariates included in the model
    """
    assert isinstance(incidents_df, pd.DataFrame), "incidents_df must be a pandas.DataFrame"
    assert isinstance(covariate_cat, str) or isinstance(covariate_cat, list), "covariate_cat must be a string or list"
    if isinstance(covariate_cat, str):
        covariate_cat = [covariate_cat]
    assert isinstance(covariate_cont, str) or isinstance(covariate_cont, list), "covariate_cont must be a string or list"
    if isinstance(covariate_cont, str):
        covariate_cont = [covariate_cont]
    assert covariate_cat is not None or covariate_cont is not None, "at least one covariate must be specified"
    assert all([col in incidents_df.columns for col in covariate_cat]), "covariate_cat must be a column in incidents_df"
    assert all([col in incidents_df.columns for col in covariate_cont]), "covariate_cont must be a column in incidents_df"

    # prepare the formula for regression
    formula = 'num_duplicates ~ 1 + '
    if covariate_cont is not None:
        if isinstance(covariate_cont, str):
            formula += covariate_cont
        else:
            formula += ' + '.join(covariate_cont)
    if covariate_cat is not None:
        if isinstance(covariate_cat, str):
            formula += ' + C(' + covariate_cat + ')'
        else:
            formula += ' + ' + ' + '.join(['C(' + col + ')' for col in covariate_cat])
    
    # prepare the data for regression
    y, X = dmatrices(formula, incidents_df, return_type='dataframe')
    duration = incidents_df['duration']
    
    return y, X, duration


def train_model(y, X, duration, model_type = 'standard'):
    """
    Train the model.
    ----------
    Parameters
    ----------
    y: pandas.DataFrame
        the response variable
    X: pandas.DataFrame
        the covariates
    duration: pandas.Series
        the duration of the incidents
    model_type: str
        the type of the model, available options are 'standard' and 'zeroinflated'
    """
    assert isinstance(y, pd.DataFrame), "y must be a pandas.Series"
    assert isinstance(X, pd.DataFrame), "X must be a pandas.DataFrame"
    assert isinstance(duration, pd.Series), "duration must be a pandas.Series"

    # fit the model
    if model_type == 'standard':
        res = sm.Poisson(y, X, exposure = duration).fit(maxiter = 1000, method = 'bfgs')
    elif model_type == 'zeroinflated':
        res = sm.ZeroInflatedPoisson(y, X, exog_infl = None, exposure = duration, 
                                                          ).fit(maxiter = 1000, method = 'bfgs')
    print(res.summary())

    return res


def generate_predicted_reporting_delay(res, X):
    """
    Generate the predicted reporting delay.
    ----------
    Parameters
    ----------
    res: the fitted model
    X: pandas.DataFrame
        the covariates
    """
    
    # generate the predicted reporting delay
    predicted_reporting_rate = res.predict(X)
    predicted_reporting_delay = 1 / predicted_reporting_rate

    return predicted_reporting_delay


def append_delays_to_incidents_df(incidents_df, delays):
    """
    Append the predicted reporting delay to the incidents dataframe.
    ----------
    Parameters
    ----------
    incidents_df: pandas.DataFrame
        the incidents dataframe, generated by create_incidents_df
    delays: the predicted reporting delay
    """
    assert isinstance(incidents_df, pd.DataFrame), "incidents_df must be a pandas.DataFrame"
    assert len(delays) == len(incidents_df), "the length of delays must be the same as the length of incidents_df"

    incidents_df['predicted_reporting_delay'] = delays

    return incidents_df

