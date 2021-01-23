import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer # for encoding categorical to numerical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split # for splitting data into trains and tests
from sklearn.linear_model import LinearRegression # for training and predicting
from sklearn.preprocessing import PolynomialFeatures # for polynomial linear regression

# this function expects level 6.5 (between Region Manager and Partner) Salary
def PLR():
    # import data
    data = pd.read_csv('Position_Salaries.csv')
    print(data)

    # Position  = Level: So, position does not need
    # Independent variables
    x = data.iloc[:, 1:-1].values

    # dependent variable
    y = data.iloc[:, -1].values
    # print(x, y)

    plt.scatter(x, y, color='red')
    plt.title('Raw Data Results')
    plt.xlabel('Levels')
    plt.ylabel('Salary')
    plt.show()

    # skip this step because we want to maximized prediction values
    # splitting data into 4 parts x_train, x_test, y_train, y_test
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    
    # Simple Linear Regression #
    # training data
    regressor = LinearRegression()  # handle dummy traps and select best models like backward elimination automatically because it includes a constant variable (b0) = 0
    regressor.fit(x, y)

    # predicting Salary by trained data
    y_pred = regressor.predict(x)
    print('\t\t Predict vs Actual')
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y.reshape(len(y), 1)), axis=1))
    print('(Simple Linear) Predict Salary of level 6.5 is %d dollars\n' %regressor.predict([[6.5]]))
    
    # visualization the data 
    plt.scatter(x, y, color='red')
    plt.plot(x, y_pred, color='blue')
    plt.title('Simple Linear Regression')
    plt.xlabel('Levels')
    plt.ylabel('Salary')
    plt.show()

    # Polynomial Linear Regression
    regressor_poly = PolynomialFeatures(degree=4) # degree 2 is by power of 2: if I need to power of 3 degree = 3
    # degree 4 is more precisely
    x_poly = regressor_poly.fit_transform(x) # this values return power of 2 of X
    regressor2 = LinearRegression()
    regressor2.fit(x_poly, y)

    # predicting Salary by trained data
    y_pred2 = regressor2.predict(x_poly)
    print('\t\t Predict vs Actual')
    print(np.concatenate((y_pred2.reshape(len(y_pred2), 1), y.reshape(len(y), 1)), axis=1))
    print('(Polynomial Linear) Predict Salary of level 6.5 is %d dollars' % regressor2.predict(regressor_poly.fit_transform([[6.5]])))

    # visualization the data
    x_grid = np.arange(min(x), max(x), 0.1) # for smooth looks arrange(start, stop, range)
    x_grid = x_grid.reshape((len(x_grid), 1))  # for smooth looks: return/reshape the smoothness to x_grid
    plt.scatter(x, y, color='red')
    plt.plot(x_grid, regressor2.predict(regressor_poly.fit_transform(x_grid)), color='blue')
    # without smoothness plt.plot(x_poly, y_pred2, color='blue')
    plt.title('Polynomial Linear Regression')
    plt.xlabel('Levels')
    plt.ylabel('Salary')
    plt.show()

if __name__ == '__main__':
    PLR()

