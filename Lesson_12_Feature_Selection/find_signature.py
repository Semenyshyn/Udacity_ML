import pickle
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(42)
words_file = r'C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\text_learning\your_word_data.pkl'
authors_file = r'C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\text_learning\your_email_authors.pkl'

word_data = pickle.load(open(words_file, 'rb'))
authors = pickle.load(open(authors_file, 'rb'))

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1,
                                                                            random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()

features_train = features_train[:150].toarray()
labels_train = labels_train[:150]
print(len(features_train))
clf = DecisionTreeClassifier()
clf.fit_transform(features_train, labels_train)
pred = clf.predict(features_test)

print(accuracy_score(pred, labels_test))

for i in range(len(clf.feature_importances_)):
    if clf.feature_importances_[i] > 0.2:
        print(i)
        print(clf.feature_importances_[i])
        print(vectorizer.get_feature_names()[i])

        # houectect