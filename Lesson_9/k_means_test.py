from Lesson_6.feature_format import featureFormat, target_feature_split
import pickle
import matplotlib.pyplot as plt

data_dict = pickle.load(open("final_project_dataset.pkl", "rb"))
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
poi = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list, remove_all_zeroes=False, remove_any_zeroes=False, remove_NaN=False)
poi, finance_features = target_feature_split(data)
for f1, f2 in finance_features:
    print(f1, f2)
    plt.scatter(f1, f2)
plt.show()
