import numpy as np
import ppmf_reid.data, ppmf_reid.models

state, state_fips, county_fips = 'MN', 27, 51  # TODO: make this an input that is easy to iterate over

df_synth = ppmf_reid.data.read_synth_data(state, county_fips)
df_sim_commercial = ppmf_reid.data.simulate_commercial_data(df_synth)
df_test = ppmf_reid.data.generate_test_data(df_synth, df_sim_commercial)

df_ppmf = ppmf_reid.data.read_ppmf_data(state_fips, county_fips)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon_infinity(df_synth)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, .01)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, .1)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, 1)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, 12.2)
#df_ppmf = ppmf_reid.data.simulate_ppmf_epsilon(df_synth, 1_000_000)

df_linked = ppmf_reid.models.link_records(df_sim_commercial, df_ppmf)
assert np.all(df_sim_commercial.index == df_test.index)

#print(df_linked)
#print(df_linked.n_match.value_counts())
#print()

print('Number of unique matches:', (df_linked.n_match == 1).sum())
for col in ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']:
    print('Number of matches with unique value for', col, (df_linked[col] == 1).sum())
print()

for col in ['hispanic', 'racwht', 'racblk', 'racaian', 'racasn', 'racnhpi', 'racsor', 'racmulti']:
    df_unique = df_linked[(df_linked[col] == 1)]
    s_correct_match = (df_unique[col] == df_test.loc[df_unique.index, col])
    print('Number of matches where unique value was correct', col, s_correct_match.sum(), 'out of', (df_linked[col] == 1).sum())

# TODO: save this information in a place where it can be aggregated later
