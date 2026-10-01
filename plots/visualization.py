import matplotlib.pyplot as plt
from tests.scikit_learn import y_test, predictions

plt.scatter(y_test, predictions)
plt.plot([y_test.min(), y_test.max()], [predictions.min(), predictions.max()], linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.savefig('actual_vs_prediction.png')
plt.show()

Errors = y_test - predictions

plt.scatter(predictions, Errors)
plt.axhline(0)
plt.xlabel("Predicted Values")
plt.ylabel("Errors")
plt.title("Error Plot")
plt.savefig('residual_plot.png')
plt.show()