import matplotlib.pyplot as plt
from Choose_Algorithm.prep_terrain_data import makeTerrainData
from Choose_Algorithm.class_vis import prettyPicture, output_image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = makeTerrainData()

models = [
    ('KNN', KNeighborsClassifier()),
    ('ADC', AdaBoostClassifier(n_estimators=90, learning_rate=1, )),
    ('RFC', RandomForestClassifier(min_samples_split=2)),
    ('DTC', DecisionTreeClassifier()),
    ('GNB', GaussianNB()),
    ('SVC', SVC())
]

for name, model in models:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    print(name, ' -- ', accuracy)
    # prettyPicture(model, name, X_test, y_test)
