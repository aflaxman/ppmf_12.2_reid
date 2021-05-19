""" Methods for data wrangling for PPMF re-identification experiment"""

import glob
import pandas as pd

def synth_fnames(state, county):
    """ get list of fnames for synthetic population for given state/county

    Parameters
    ----------
    state : str, two-letters, e.g. 'mn'
    county : int, FIPS code without state digits (e.g. 53)
    """
    fname_list = glob.glob(f'/ihme/scratch/users/beatrixh/synthetic_pop/pyomo/best/{state.lower()}/*county{county:03d}*.csv')
    return fname_list

def read_synth_data(state, county):
    """ load pd.DataFrame of synthetic population for given state/county

    Parameters
    ----------
    state : str, two-letters, e.g. 'mn'
    county : int, FIPS code without state digits (e.g. 53)
    """
    fname_list = synth_fnames(state, county)

    df_list = []
    for i, fname in enumerate(fname_list):
        df = pd.read_csv(fname)
        df_list.append(df)
    df_synth = pd.concat(df_list, ignore_index=True)
    return df_synth

def read_ppmf_data(state, county):
    """ load pd.DataFrame of PPMF-12.2 person file for given state/county

    Parameters
    ----------
    state : int, FIPS code for state
    county : int, FIPS code without state digits (e.g. 53)
    """
    fname = f'/share/scratch/users/abie/ppmf/ppmf_20210428_eps12-2_P_{state:02d}{county:03d}.csv.gz'
    df_ppmf = pd.read_csv(fname)
    return df_ppmf

def simulate_commercial_data(df_synth, race_eth_cols=[], race_eth_query=''):
    """create simulated commercial data by redacting some of the
    population synthesised columns

    Parameters
    ----------
    df_synth : pd.DataFrame, from e.g. read_synth_data
    race_eth_cols : list, optional, race and ethnicity columns *not* to redact
    race_eth_query : str, optional, query string used to subset df_synth
    """
    if race_eth_query:
        race_eth_query += ' and '

    # relationship 16 is "Institutionalized group quarters population"
    # relationship 17 is "Noninstitutionalized group quarters population"
    race_eth_query += 'relationship != 16 and relationship != 17'

    df_synth_filtered = df_synth.query(race_eth_query)

    df_sim_commercial = df_synth_filtered.filter(['state', 'county', 'tract', 'block', 'age', 'sex_id'] + race_eth_cols)
    df_sim_commercial['voting_age'] = (df_sim_commercial.age >= 18)

    return df_sim_commercial

def generate_test_data(df_synth, df_sim_commercial):
    """create testing data of race and ethnicity columns to compare to the
    results of re-identification algorithms

    Parameters
    ----------
    df_synth : pd.DataFrame, from e.g. read_synth_data
    race_eth_cols : list, optional, race and ethnicity columns *not* to redact
    race_eth_query : str, optional, query string used to subset df_synth
    """
    test_rows = df_sim_commercial.index
    race_cols = ['racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor']
    df_test = df_synth.loc[test_rows].filter(['hispanic'] + race_cols)
    df_test['racmulti'] = (df_test[race_cols].sum(axis=1)>1)

    return df_test
