from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.cross_validation import train_test_split


def preprocess(words_file=r"C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\text_learning\your_word_data.pkl",
               authors_file=r"C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\text_learning\your_email_authors.pkl"):
    authors_file_handler = open(authors_file, "rb")
    authors = pickle.load(authors_file_handler)
    authors_file_handler.close()

    words_file_handler = open(words_file, 'rb')
    word_data = pickle.load(words_file_handler)
    words_file_handler.close()

    features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors,
                                                                                test_size=0.1,
                                                                                random_state=42)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    print("no. of Chris training emails:", sum(labels_train))
    print("no. of Sara training emails:", len(labels_train) - sum(labels_train))

    return features_train_transformed, features_test_transformed, labels_train, labels_test


def main():
    preprocess()


if __name__ == '__main__':
    main()
