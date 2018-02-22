from Lesson_6.feature_format import feature_format, target_feature_split
from Lesson_8.outlier_cleaner import OutlierCleaner
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ages = pd.read_pickle('practice_outliers_ages.pkl')
net_worths = pd.read_pickle('practice_outliers_net_worths.pkl')

ages = np.reshape(np.array(ages), (len(ages), 1))
net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))

ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1,
                                                                            random_state=42)
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
# slope print(reg.coef_)
# score print(reg.score(ages_test, net_worths_test))
cleaned_data = OutlierCleaner(reg.predict(ages_train), ages_train, net_worths_train)
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = np.reshape(np.array(ages), (len(ages), 1))
    net_worths = np.reshape(np.array(net_worths), (len(net_worths), 1))
    try:
        reg.fit(ages, net_worths)
        print(reg.coef_)
        print(reg.score(ages_test, net_worths_test))
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print(
            '''you don't seem to have regression imported/created,
           or else your regression object isn't named reg"
           either way, only draw the scatter plot of the cleaned data''')
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()
else:
    print ("outlierCleaner() is returning an empty list, no refitting to be done")


# VISUALISATION
# plt.plot(ages, reg.predict(ages), color='r')
# plt.scatter(ages, net_worths)
# plt.show()
