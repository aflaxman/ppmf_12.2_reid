import numpy as np
import pandas as pd

race_eth_cols = ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']

def match_summary(df):
    """ summarize the race and ethnicity information in the dataframe

    Parameters
    ----------
    df : pd.DataFrame including columns from race_eth_cols
    """

    s = pd.Series(dtype=float)
    s['n_match'] = len(df)
    for col in race_eth_cols:
        if col in df.columns:
            s[col] = np.sum(df.pweight * df[col]) / df.pweight.sum()

    return s

def link_records(df_sim_commercial, df_ppmf):
    """merge summary of ppmf race/ethnicity data into simulated
    commercial data

    Parameters
    ----------
    df_sim_commercial : pd.DataFrame, including geographic columns, and voting_age column
    df_ppmf : pd.DataFrame, including geographic columns, voting_age column, and race/ethnicity columns
    """
    # for faster merging and summarizing, caculate and summarize the info for each strata in the ppmf
    df_ppmf_strata = df_ppmf.groupby(['state', 'county', 'tract', 'block', 'voting_age']
                                ).apply(match_summary)

    # merge in the pre-computed summary for each linked record
    df_linked = pd.merge(df_sim_commercial, df_ppmf_strata,
                         left_on=['state', 'county', 'tract', 'block', 'voting_age'],
                         right_index=True, how='left')

    # fill in the blanks with n_match value 0
    df_linked['n_match'] = df_linked.n_match.fillna(0).astype(int)

    return df_linked

def load_and_link(state, state_fips, county_fips):
    import ppmf_reid.data

    df_synth = ppmf_reid.data.read_synth_data(state, county_fips)
    df_sim_commercial = ppmf_reid.data.simulate_commercial_data(df_synth)
    df_test = ppmf_reid.data.generate_test_data(df_synth, df_sim_commercial)
    assert np.all(df_sim_commercial.index == df_test.index)

    df_ppmf_12 = ppmf_reid.data.read_ppmf_data(state_fips, county_fips)
    df_ppmf_inf = ppmf_reid.data.simulate_ppmf_epsilon_infinity(df_synth)
    df_sim_ppmf = {}
    for eps in [.05, .1, .2]:
        df_sim_ppmf[f'sim_{eps:.02f}'] = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, eps)

    df_ppmf = df_sim_ppmf.copy()
    df_ppmf.update({'12.2':df_ppmf_12,
                    'inf':df_ppmf_inf})

    df_linked = {}
    for key in df_ppmf.keys():
        df_linked[key] = link_records(df_sim_commercial, df_ppmf[key])

    return locals()

def summarize_results(results):
    summary = pd.Series(dtype='object')
    summary['state'] = results['state']
    summary['state_fips'] = results['state_fips']
    summary['county_fips'] = results['county_fips']
    
    df_test = results['df_test']

    for eps, df_linked in results['df_linked'].items():
        summary[f'n_unique_{eps}'] = (df_linked.n_match == 1).sum()
        for col in ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']:
            summary[f'n_unique_{col}_{eps}'] = (df_linked[col] == 1).sum()
            df_unique = df_linked[(df_linked[col] == 1)]
            s_correct_match = (df_unique[col] == df_test.loc[df_unique.index, col])
            summary[f'n_matched_{col}_{eps}'] = s_correct_match.sum()
    return summary
