import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy import stats

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# read csv into dataframe
df = pd.read_csv("./weatherAUS.csv", parse_dates=['Date'])

print('Size of weather data frame is :', df.shape)

# drop columns with useless data (too many null values)
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'], axis=1)

# get rid of nulls
df = df.dropna(how='any')

# use "z-score" to remove outliers
z = np.abs(stats.zscore(df._get_numeric_data()))
df= df[(z < 3).all(axis=1)]

# replace binary columns with 0 and 1
df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

# change categorical columns into a one-hot encoded format
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
df = pd.get_dummies(df, columns=categorical_columns)

# standardize data
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)


df.to_csv("preprocessed.csv")
# separate data from target
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]

# k = 5 has 84.25%, k=5 has 84.01%, 85.19% on all data ( bad cols removed), with date: 85.17
# select the k most useful columns
#selector = SelectKBest(chi2, k=10)
#selector.fit(X, y)
#X_new = selector.transform(X)

# fit and evaluate using k_fold

model = XGBClassifier()
kfold = KFold(n_splits=5, random_state=7)
print("Cross eval starting")

results = cross_val_score(model, X, y, cv=kfold, verbose=3)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



