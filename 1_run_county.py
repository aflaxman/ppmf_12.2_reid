import sys
import numpy as np, pandas as pd
import ppmf_reid.data, ppmf_reid.models

assert len(sys.argv) in [4,6], 'usage: run_on_cluster.sh 1_run_county.py [state] [state_fips] [county_fips] [[n_chunks]] [[chunk_i]]'
if len(sys.argv) == 4:
    _, state, state_fips, county_fips = sys.argv
    n_chunks, chunk_i = 1, 0
else:
    _, state, state_fips, county_fips, n_chunks, chunk_i = sys.argv
    
state = state.lower()
state_fips, county_fips = int(state_fips), int(county_fips)
n_chunks, chunk_i = int(n_chunks), int(chunk_i)

results = ppmf_reid.models.load_and_link(state, state_fips, county_fips, n_chunks, chunk_i)
summary = ppmf_reid.models.summarize_results(results)

#import pdb; pdb.set_trace()
fname = f'/share/scratch/users/abie/ppmf/results_{state_fips}_{county_fips}_{chunk_i}.csv.gz'
pd.DataFrame(summary).T.to_csv(fname, index=False)

t = pd.read_csv(fname)
assert np.all(t.iloc[0] == summary)
#print(t)
#print(summary)

