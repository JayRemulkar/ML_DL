# Implement Linear Regression(Diabetes Dataset).

import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,r2_score

def LinearModel(diabetes_X_train,diabetes_X_test,diabetes_y_train,diabetes_y_test):
      
      regr = linear_model.LinearRegression()
      regr.fit(diabetes_X_train, diabetes_y_train)
      diabetes_y_pred = regr.predict(diabetes_X_test)

      print('Coefficients: \n', regr.coef_)
      print('Mean squared error: %.2f'% mean_squared_error(diabetes_y_test, diabetes_y_pred))
      print('Coefficient of determination: %.2f'% r2_score(diabetes_y_test, diabetes_y_pred)) 

      plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
      plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
      plt.title("Linear regression Diabeties Dataset")
      plt.show()

def main():

      diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
      diabetes_X = diabetes_X[:, np.newaxis, 2]

      diabetes_X_train = diabetes_X[:-20]
      diabetes_X_test = diabetes_X[-20:]
      diabetes_y_train = diabetes_y[:-20]
      diabetes_y_test = diabetes_y[-20:]

      LinearModel(diabetes_X_train,diabetes_X_test,diabetes_y_train,diabetes_y_test)

if __name__ == "__main__":
    main()