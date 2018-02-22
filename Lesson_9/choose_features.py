import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

data_dict = pd.read_pickle('final_project_dataset.pkl')
df = pd.DataFrame.from_dict(data_dict, orient='index')
for i in df:
    df[i] = df[i].apply(lambda x: 0 if isinstance(x, str) else x)
    df[i] = df[i][df[i] != 0]
df = df.drop('TOTAL')
# df = df[['salary', 'bonus', 'total_payments', 'restricted_stock', 'shared_receipt_with_poi', 'deferral_payments',
#          'total_stock_value']]
# print(df[df['salary'] > 1000000])
# scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')
# plt.show()
df = df['salary']
print(max(df))
print(min(df))