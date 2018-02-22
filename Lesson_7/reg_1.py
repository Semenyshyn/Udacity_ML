import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

data_dict = pd.read_pickle('final_project_dataset_modified.pkl')
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df[['bonus', 'salary']].reset_index().rename(columns={'index': 'name'})
df = df[(df['bonus'] != 'NaN') & (df['salary'] != 'NaN')]
df = df[(df.T != 0).any()]
df['bonus'] = df.bonus.apply(lambda x: float(x))
df['salary'] = df.salary.apply(lambda x: float(x))
target, features = np.array(df['bonus']), np.array(df['salary'])

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

# import matplotlib.pyplot as plt
#
# for feature, target in zip(feature_test, target_test):
#     plt.scatter(feature, target, color=test_color)
# for feature, target in zip(feature_train, target_train):
#     plt.scatter(feature, target, color=train_color)
#
# plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
# plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")
# plt.show()

reg = LinearRegression()
reg.fit(feature_train, target_train)
