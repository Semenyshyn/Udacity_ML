from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
print(stemmer.stem('responsiveness'))
print(stemmer.stem('responsivity'))
print(stemmer.stem('unresponsive'))

# sw = stopwords.words('english')
# print(len(sw))


