# Simulation study of re-identification risk in ppmf_20210428_eps12-2_P

Plan of Work:

For each county, just for the people in households (not gq):

1. Load individual person data synthesized for that county in re-id
   exercise (`df_synth`)

2. Split `df_synth` it into (a) simulated external data (`df_sim`) and
   (b) hold-out validation data (`df_test`)

3. Load corresponding county of privacy protected person data from
   PL-74 demonstration product (`df_ppmf`)

4. merge `df_sim` and `df_ppmf` on their common fields, e.g. track,
   block, and voting_age

5. see how many individuals in `df_sim` have a unique match, and how
   many individuals in `df_sim` have a unique race or ethnicity in the
   matched ppmf data

6. for the individuals in `df_sim` with a unique match or with matches
   with a unique race/ethnicity, see how often this linked data is the
   same as the value in the validation data in `df_test`

Since the PPMF is based on the pre-swapped data, step (6) is perhaps
not meaningful.  If instead of loading df_ppmf, I simulate it from the
same df_synth data, I can make it meaningful, and also sweep across
values of epsilon.  And then compare at epsilon empirical found from
PPMF.