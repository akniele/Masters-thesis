import os
import pickle
import numpy as np
import pandas as pd
import fnmatch

directory = 'distances/'  
subjects_per_file = 32_000
df_data = False

 
"""
Create a table for a one-way ANOVA
dependent variable: weighted Manhattan distance
independent variable: transformation type (entropy, top-p, bucket, None, baseline)
"""
    
for file in os.listdir(directory):
    print(file)
    trans_value = None
        
    #if file == "dist_unchanged.pkl" or file == "dist_baseline.pkl":
    #   continue
       
    if fnmatch.fnmatch(file, '*top_p*.pkl'):
        trans_value = 1 
    elif fnmatch.fnmatch(file, '*entropy*.pkl'):
        trans_value = 2  
    elif fnmatch.fnmatch(file, '*bucket*.pkl'):
        trans_value = 3
    elif fnmatch.fnmatch(file, '*baseline*.pkl'):
        trans_value = 4    
    elif fnmatch.fnmatch(file, '*unchanged*.pkl'):
        trans_value = 5
    else:
        raise ValueError("trans_value has to be one of the values 1, 2, 3, 4 or 5!")
 

    with open(f"{directory}{file}", 'rb') as file:
        data = pickle.load(file)
        data = np.reshape(data, (subjects_per_file,))
        subjects = np.arange(0, subjects_per_file)
            
        transformation = np.full((subjects_per_file,), trans_value)

        df_tmp_data = pd.DataFrame({'subject': subjects, 'trans_type': transformation, 'distance': data})
                                    
        df_tmp_data['trans_type'] = df_tmp_data['trans_type'].astype('category')

                                    
        print(f"data frame:\n {df_tmp_data.head(5)}")                            

        if type(df_data) is bool:
            df_data = df_tmp_data

        else:
            df_data = pd.concat([df_data, df_tmp_data], axis=0, ignore_index=True)
        
print(f"number of rows in dataframe: {df_data.shape[0]}")

df_data.to_parquet('df_anova_trans_type.parquet.gzip', compression='gzip')