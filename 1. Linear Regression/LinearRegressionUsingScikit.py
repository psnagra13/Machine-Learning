import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

print(y)
reg = LinearRegression().fit(X, y)

print ( reg.score(X, y) )

reg.coef_
reg.intercept_ 
reg.predict(np.array([[3, 5]]))
