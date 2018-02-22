import pandas as pd
import numpy as np


def feature_format(data_dict, features, remove_all_zeros=True):
    df = pd.DataFrame.from_dict(data_dict, orient='index')
    res_df = df[features]
    if remove_all_zeros:
        for i in range(len(features)):
            try:
                res_df = res_df.loc[(res_df[features[i]] != 0) & (res_df[features[i]] != 'NaN')]
            except:
                pass
    return_array = []
    for i in features:
        return_array.append(res_df.as_matrix(columns=[i]))
    return return_array
