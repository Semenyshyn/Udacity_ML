from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tools.feature_format import featureFormat, targetFeatureSplit
import pandas as pd

data_dict = pd.read_pickle(r'C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\Lesson_9\final_project_dataset.pkl')
features_list = ["poi", "salary"]
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))


