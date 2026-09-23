from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import pandas as pd
import numpy as np

df = pd.read_csv('forestfires.csv')

X = df.drop(columns=['day','month','area'])
y = df['area']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = XGBRegressor(n_estimeters=200,learning_rate=0.1,max_depth=5,reg_lambda=1,gamma=0)
model.fit(x_train,y_train)
predictions = model.predict(x_test)

mse = mean_squared_error(y_test,predictions)
mae = mean_absolute_error(y_test,predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,predictions)

print('MAE:',mae)
print('MSE:',mse)
print('RMSE:',rmse)
print('R2 Score:',r2)

