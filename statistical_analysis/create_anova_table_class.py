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
independent variable: classification type (classifier, random labels, no classification)
only do this for top-p transformation and entropy transformation!
"""

patterns = ["*_2_*", "*_3_*", "*_4_*", "*_5_*"]

trans_patterns = ["*bucket*", "*unchanged*", "*baseline*"]
    
for file in os.listdir(directory):
    class_value = None
    print(file)
        
    if any(fnmatch.fnmatch(file, pattern) for pattern in trans_patterns):
        continue
       
    if any(fnmatch.fnmatch(file, pattern) for pattern in patterns) and not fnmatch.fnmatch(file, '*random*.pkl'):  
        class_value = 1
    elif fnmatch.fnmatch(file, '*random*.pkl'):
        class_value = 2
    elif fnmatch.fnmatch(file, '*None*.pkl'):
        class_value = 3
    else:
        raise ValueError("class_value has to be one of the values 1, 2 or 3!")
 

    with open(f"{directory}{file}", 'rb') as file:
        data = pickle.load(file)
        data = np.reshape(data, (subjects_per_file,))
        subjects = np.arange(0, subjects_per_file)
            
        transformation = np.full((subjects_per_file,), class_value)

        df_tmp_data = pd.DataFrame({'subject': subjects, 'class_type': transformation, 'distance': data})
                                    
        df_tmp_data['class_type'] = df_tmp_data['class_type'].astype('category')

                                    
        print(f"data frame:\n {df_tmp_data.head(5)}")                            

        if type(df_data) is bool:
            df_data = df_tmp_data

        else:
            df_data = pd.concat([df_data, df_tmp_data], axis=0, ignore_index=True)

        
print(f"number of rows in dataframe: {df_data.shape[0]}")

df_data.to_parquet('df_anova_class_type.parquet.gzip', compression='gzip')