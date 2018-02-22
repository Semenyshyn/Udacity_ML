import os
import pickle
import re
import sys
from tools.parse_out_email_text import parseOutText
from sklearn.feature_extraction.text import TfidfVectorizer

base_enron_dir = r'C:\Users\IVAN.SEMENYSHYN\Desktop\MyProjects\ENRON' + '\\'
from_sara = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

stop_words = ['sara', 'shackleton', 'chris', 'germani', 'sshacklensf', 'cgermannsf']

from_data = []
word_data = []

temp_counter = 0
o = 0
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        o += 1
        print(path)
        print(o)
        temp_counter += 1
        path = base_enron_dir + path.replace('/', '\\')[:-1]
        email = open(path, "r")
        text = parseOutText(email)
        res = ' '.join([word for word in text.split() if word.lower() not in stop_words])
        word_data.append(res)
        if name == 'sara':
            from_data.append(0)
        elif name == 'chris':
            from_data.append(1)
        email.close()

print("emails processed")
from_sara.close()
from_chris.close()

pickle.dump(word_data, open("your_word_data.pkl", "wb"))
pickle.dump(from_data, open("your_email_authors.pkl", "wb"))

vect = TfidfVectorizer(stop_words="english")
vect.fit_transform(word_data)
print(len(vect.get_feature_names()))
print(vect.get_feature_names()[34597])
