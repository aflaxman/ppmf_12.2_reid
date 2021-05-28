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
    s['n_match'] = df.pweight.sum()
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

def simple_impute_records(df_sim_commercial):
    """merge non-hispanic, white, of voting-age into simulated
    commercial data

    Parameters
    ----------
    df_sim_commercial : pd.DataFrame, including geographic columns, and voting_age column
    """
    df_linked = df_sim_commercial.copy()

    for col in race_eth_cols:
        df_linked[col] = 0
    df_linked['racwht'] = 1


    # fill in the blanks with n_match value 0
    df_linked['n_match'] = 0

    return df_linked

def load_and_link(state, state_fips, county_fips, n_chunks=1, chunk_i=0):
    """ load and simulate data and link it

    Parameters
    ----------
    state : str, e.g. 'WA'
    state_fips : int, e.g. 53 (note: redundant, but convenient)
    county_fips : int, e.g. 53
    n_chunks : int, to break up large counties
    chunk_i : int, < n_chunks
    """

    import ppmf_reid.data

    df_synth = ppmf_reid.data.read_synth_data(state, county_fips)
    df_synth = df_synth[df_synth.tract % n_chunks == chunk_i]
    df_sim_commercial = ppmf_reid.data.simulate_commercial_data(df_synth)
    df_test = ppmf_reid.data.generate_test_data(df_synth, df_sim_commercial)
    assert np.all(df_sim_commercial.index == df_test.index)

    df_ppmf_12 = ppmf_reid.data.read_ppmf_data(state_fips, county_fips)
    df_ppmf_12 = df_ppmf_12[df_ppmf_12.tract % n_chunks == chunk_i]

    #df_ppmf_4 = ppmf_reid.data.read_ppmf_data_4(state_fips, county_fips)
    df_ppmf_inf = ppmf_reid.data.simulate_ppmf_epsilon_infinity(df_synth)
    df_sim_ppmf = {}
    for eps in [.01, .1, 1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.6, 1.8] + list(range(2,9,2)):
        df_sim_ppmf[f'sim_{eps:.02f}'] = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, eps)

    df_ppmf = df_sim_ppmf.copy()
    df_ppmf.update({'12.2':df_ppmf_12,
                    'inf':df_ppmf_inf})

    df_linked = {}
    if len(df_sim_commercial) > 0:
        df_linked['baseline'] = simple_impute_records(df_sim_commercial)
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
        summary[f'n_unique_match_eps_{eps}'] = (df_linked.n_match == 1).sum()

        df_unique_impute = df_linked[np.all(df_linked[race_eth_cols]%1 == 0, axis=1)]
        summary[f'n_unique_impute_all_eps_{eps}'] = len(df_unique_impute)

        summary[f'n_correct_impute_all_eps_{eps}'] = np.all(df_unique_impute[race_eth_cols] == df_test.loc[df_unique_impute.index, race_eth_cols], axis=1).sum()

        for col in ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']:
            summary[f'n_unique_impute_attribute_{col}_eps_{eps}'] = (df_linked[col] == 1).sum()
            df_unique_impute = df_linked[(df_linked[col] == 1)]
            s_correct_impute = (df_unique_impute[col] == df_test.loc[df_unique_impute.index, col])
            summary[f'n_correct_impute_attribute_{col}_eps_{eps}'] = s_correct_impute.sum()

            df_unique_match = df_linked[(df_linked.n_match == 1)]
            s_correct_match = (df_unique_match[col] == df_test.loc[df_unique_match.index, col])
            summary[f'n_unique_match_correct_impute_attribute_{col}_eps_{eps}'] = s_correct_match.sum()

        # tally counts for non-majority re-id
        summary[f'n_unique_impute_nonmajority_{eps}'] = 0
        summary[f'n_correct_impute_nonmajority_eps_{eps}'] = 0
        for i, df_sim_commercial_i in results['df_sim_commercial'].groupby(['state', 'county', 'tract']):
            df_test_i = df_test.loc[df_sim_commercial_i.index]
            s_rac_eth_cnts = df_test_i.sum()
            majority_race_eth = s_rac_eth_cnts.sort_values(ascending=False).index[0]
            non_majority_rows = df_test_i[majority_race_eth] == 0
            df_unique_impute = df_linked[np.all(df_linked[race_eth_cols]%1 == 0, axis=1)&non_majority_rows]
            summary[f'n_unique_impute_nonmajority_{eps}'] += len(df_unique_impute)
            summary[f'n_correct_impute_nonmajority_eps_{eps}'
                   ] += np.all(df_unique_impute[race_eth_cols] == df_test_i.loc[df_unique_impute.index, race_eth_cols], axis=1).sum()

        # add total and race/eth alone or in combination counts for convenience
        for col in ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']:
            summary[f'n_{col}'] = df_test[col].sum()
        summary['n_total'] = len(df_test)


    return summary
