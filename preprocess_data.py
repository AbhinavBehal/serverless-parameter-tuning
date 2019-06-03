import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

# read csv into dataframe
df = pd.read_csv("data/weatherAUS.csv", parse_dates=['Date'])

# drop columns with useless data (too many null values)
df = df.drop(columns=['Sunshine', 'Evaporation', 'Cloud3pm', 'Cloud9am', 'Location', 'RISK_MM', 'Date'], axis=1)

# get rid of nulls
df = df.dropna(how='any')

# use "z-score" to remove outliers
z = np.abs(stats.zscore(df._get_numeric_data()))
df = df[(z < 3).all(axis=1)]

# replace binary columns with 0 and 1
df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)

# change categorical columns into a one-hot encoded format
categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
df = pd.get_dummies(df, columns=categorical_columns)

# standardize data
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

df.to_csv("data/preprocessed.csv")
