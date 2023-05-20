import pandas as pd
import sys

iv = sys.argv[1]

df_long = pd.read_parquet(f'df_anova_{iv}.parquet.gzip')

df_wide = df_long.pivot_table(index='subject', columns=iv, values='distance')

df_wide.to_csv(f'df_anova_{iv}_wide.csv')