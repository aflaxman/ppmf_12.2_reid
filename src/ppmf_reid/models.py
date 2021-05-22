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

