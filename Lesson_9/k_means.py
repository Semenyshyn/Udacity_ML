from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tools.feature_format import feature_format


def Draw(pred, features, poi, mark_poi=False, f1_name='feature 1', f2_name='feature 2'):
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.show()


data_dict = pd.read_pickle('final_project_dataset.pkl')
df = pd.DataFrame.from_dict(data_dict, orient='index')
data_dict.pop('TOTAL', 0)
features_list = ['poi', 'salary', 'exercised_stock_options']
data = feature_format(data_dict, features_list)

# feature target split
poi = data[0]

finance_features = np.column_stack((data[1], data[2]))


for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

clt = KMeans(n_clusters=2)
clt.fit(finance_features)
pred = clt.predict(finance_features)




try:
    Draw(pred, finance_features, poi, mark_poi=False, f1_name=features_list[1], f2_name=features_list[2])
except NameError:
    print("no predictions object named pred found, no clusters to plot")
