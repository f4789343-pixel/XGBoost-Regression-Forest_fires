import pandas as pd
import numpy as np

df = pd.read_csv('data/forestfires.csv')
print(df.columns)
print(df['area'].head(-5))
print(df.info())

X = df.drop(columns=['day','month','area'])
y = df['area']

np.random.seed(42)

indices = np.random.permutation(len(X))
test_size = int(len(X)*0.2)

train_indices = indices[test_size:]
test_indices = indices[:test_size]

x_train = X.iloc[train_indices]
x_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

y_train_log = np.log1p(y_train)
