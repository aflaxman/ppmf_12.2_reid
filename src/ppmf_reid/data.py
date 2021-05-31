""" Methods for data wrangling for PPMF re-identification experiment"""

import glob
import numpy as np
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

def transform_race_ethnicity(df):
    df = df.copy()
    # map race and ethnicity data to mutually exclusive categories
    race_cols = ['racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor']

    # first make anyone with hispanic flag set have no race flags set
    df.loc[df.hispanic == 1, race_cols] = 0

    # then make anyone who still has multiple race flags set instead
    # have a single multiracial flag set
    racmulti_rows = (df[race_cols].sum(axis=1) > 1)
    df['racmulti'] = racmulti_rows.astype(int)
    df.loc[racmulti_rows, race_cols] = 0

    # confirm that everyone has a unique race/ethnicity now
    assert np.allclose(df.filter(['hispanic', 'racmulti'] + race_cols).sum(axis=1), 1)
    return df

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

    df_synth = transform_race_ethnicity(df_synth)

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
    df_ppmf = transform_race_ethnicity(df_ppmf)
    df_ppmf['pweight'] = 1.0
    return df_ppmf

def simulate_ppmf_epsilon_infinity(df_synth):
    """ simulate PPMF with epsilon of infinity (where it would match df_synth exactly)

    Parameters
    ----------
    df_synth : pd.DataFrame, from e.g. read_synth_data
    """
    t = df_synth.filter([
        'state', 'county', 'tract', 'block', 'hispanic',
        'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti',
    ])
    t['voting_age'] = (df_synth.age >= 18).astype(int)
    t['pweight'] = 1.0
    return t

def simulate_ppmf_epsilon(df_synth, epsilon):
    """ simulate PPMF with given epsilon

    Parameters
    ----------
    df_synth : pd.DataFrame, from e.g. read_synth_data
    """
    t = df_synth.filter([
        'state', 'county', 'tract', 'block', 'hispanic',
        'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor',
    ])
    t['pweight'] = 1
    t['voting_age'] = (df_synth.age >= 18).astype(int)
    s_hist = t.groupby(['state', 'county', 'tract', 'block', 
                        'hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor',
                        'voting_age']).pweight.sum()
    index_names = s_hist.index.names
    s_hist = pd.Series(s_hist, index=pd.MultiIndex.from_product((t.state.unique(), t.county.unique(), t.tract.unique(), t.block.unique(),
                                                                 [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])))
    s_hist = s_hist.fillna(0)
    s_hist.index.names = index_names  # FIXME: there must be a cooler way to fill in the missing rows in the multi-index and keep the names

    dp_noise = np.random.laplace(scale=1/(2*epsilon), size=len(s_hist))
    noisy_hist = np.clip(s_hist+dp_noise, 0, np.inf)
    #noisy_hist *= len(df_synth)/noisy_hist.sum()  # rescale to keep total population invariant  (TODO: rescale all counties simulataneously to match state count)
    noisy_hist = np.round(noisy_hist) # round to have integral number of people (TODO: ensure the total count is still invariant)
    
    df_ppmf = noisy_hist.reset_index()
    df_ppmf = df_ppmf[df_ppmf.pweight > 0]

    # df_ppmf (and s_hist) includes rows for individuals with no race, which is not allowed, filter them out now
    df_ppmf = df_ppmf.query('not (racwht == 0 and racblk == 0 and racaian == 0 and racasn == 0 and racnhpi == 0 and racsor == 0)')

    df_ppmf = transform_race_ethnicity(df_ppmf)

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
    race_eth_cols = ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']
    df_test = df_synth.loc[test_rows, race_eth_cols]

    return df_test
