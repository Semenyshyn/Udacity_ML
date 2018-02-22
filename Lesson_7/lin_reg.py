from Lesson_6.feature_format import feature_format, target_feature_split
import pickle
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dictionary = pickle.load(
    open(r"C:\Users\IVAN.SEMENYSHYN\PycharmProjects\Udacity_ML\Lesson_8\final_project_dataset.pkl", "rb"))
features_list = ["bonus", "salary"]
data = feature_format(dictionary, features_list, remove_any_zeroes=True, sort_keys='python2_lesson06_keys.pkl')
print(len(data))
target, features = target_feature_split(data)
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

reg = LinearRegression()
reg.fit(feature_train, target_train)
# print(reg.coef_)
# print(reg.intercept_)
# print(reg.score(feature_test, target_test))
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

## draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass
plt.xlabel(features_list[1])
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="g")
plt.ylabel(features_list[0])
plt.legend()
plt.show()
print(reg.coef_)
