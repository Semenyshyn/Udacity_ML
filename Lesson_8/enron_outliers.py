import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from Lesson_8.outlier_cleaner import OutlierCleaner

pd.set_option('display.max_colwidth', -1)
data_dict = pd.read_pickle('final_project_dataset.pkl')
df1 = pd.DataFrame.from_dict(data_dict, orient='index').reset_index()

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df[['salary', 'bonus']]
df = df[['salary', 'bonus']][
    (df['salary'] != 'NaN') & (df['bonus'] != 'NaN')]

reg = LinearRegression()
salary = df.as_matrix(columns=['salary'])
bonus = df.as_matrix(columns=['bonus'])
reg.fit(salary, bonus)
# cleaned_data = OutlierCleaner(reg.predict(salary), salary, bonus)

plt.plot(salary, reg.predict(salary), color='red')
plt.scatter(df['salary'], df['bonus'])
plt.show()

# for i in cleaned_data[-5:]:
#     print(df1['index'][df1['salary'] == i[0][0]])
#     print(i[0][0])

df['salary'] = df.salary.apply(lambda x: int(x))
df['bonus'] = df.bonus.apply(lambda x: int(x))
print(df.dtypes)
print(df[(df['salary'] >= 1000000) & (df['bonus'] >= 5000000)])
