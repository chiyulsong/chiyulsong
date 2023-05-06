import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import datetime 

import shutil # copy files

pd.set_option('max_columns', None)
pd.set_option('max_rowss', 50)

plt.stylle.use('seaborn') # 'ggplot'
sns.set(font_scale=1)

# Union dataframe regardless of column name
def df_union(df1, df2):
    # Keep df1's column name
    if len(df1.columns) == len(df2.columns):
        new_cols = {y: x for x, y in zip(df1.columns, df2.columns)}
        df = df1.append(df2.rename(columns=new_cols))
    else:
        print("The number of columns are different between df1 and df2")
    
    return df
