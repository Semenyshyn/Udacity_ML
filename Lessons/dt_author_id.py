from Lessons.nb_preparing_data import data_preparing
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

features_train, features_test, labels_train, labels_test = data_preparing()

clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

accuracy = accuracy_score(pred, labels_test)
print(accuracy)

# print(len(features_train[0]))
