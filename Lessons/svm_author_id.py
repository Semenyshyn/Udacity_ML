from Lessons.nb_preparing_data import data_preparing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# clf = SVC(kernel='linear')
clf = SVC(kernel='rbf', C=10000)
features_train, features_test, labels_train, labels_test = data_preparing()

# features_train = features_train[:int(len(features_train) / 100)]
# labels_train = labels_train[:int(len(labels_train) / 100)]
print(features_test)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
#
print(accuracy)
# C=10 - 0.616040955631
# C=100 - 0.616040955631
# C=1000 - 0.821387940842
# C=10000 - 0.892491467577

from collections import Counter
print(Counter(pred))
