from Lessons.nb_preparing_data import data_preparing
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


features_train, features_test, labels_train, labels_test = data_preparing()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

accuracy = accuracy_score(pred, labels_test)
print(accuracy)


