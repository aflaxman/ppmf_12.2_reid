import sys
import numpy as np, pandas as pd
import ppmf_reid.data, ppmf_reid.models

assert len(sys.argv) == 4, 'usage: run_on_cluster.sh 1_run_county.py [state] [state_fips] [county_fips]'
_, state, state_fips, county_fips = sys.argv
state = state.lower()
state_fips, county_fips = int(state_fips), int(county_fips)

results = ppmf_reid.models.load_and_link(state, state_fips, county_fips)
summary = ppmf_reid.models.summarize_results(results)

#import pdb; pdb.set_trace()
fname = f'/share/scratch/users/abie/ppmf/results_{state_fips}_{county_fips}.csv.gz'
pd.DataFrame(summary).T.to_csv(fname, index=False)

t = pd.read_csv(fname)
assert np.all(t.iloc[0] == summary)
#print(t)
#print(summary)

