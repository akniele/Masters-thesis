import pandas as pd
import pingouin as pg
import sys


iv = sys.argv[1]  # 'class_type' or 'trans_type'

df_data = pd.read_parquet(f'df_anova_{iv}.parquet.gzip')

aov = pg.rm_anova(dv='distance', within=iv, subject='subject', data=df_data, effsize='np2')

print(aov)

post_hoc = pg.pairwise_tests(dv='distance', within=iv, subject='subject', data=df_data, padjust='bonf', effsize='eta-square')

print(post_hoc)