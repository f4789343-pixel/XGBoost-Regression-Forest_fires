import numpy as np
from train import pred, y_test
mse = np.mean((pred - y_test)**2)
mae = np.mean(abs(pred - y_test))
rmse = np.sqrt(mse)
ss_total = np.sum((y_test - np.mean(y_test))**2)
ss_residual = np.sum((y_test - pred)**2)
r2 = 1 - (ss_residual / ss_total)

print('MAE:',mae)
print('MSE:',mse)
print('RMSE:',rmse)
print('R2 Score:',r2)