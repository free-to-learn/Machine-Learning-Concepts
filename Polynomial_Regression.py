import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Data Set Creation bu using Numpy
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# Transformation of Data
transformer = PolynomialFeatures(degree=2, include_bias=False) 
transformer.fit(x,y)
x_ = transformer.transform(x)
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)


# Create a model and fit it
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)
print(x_)

model = LinearRegression(fit_intercept=False).fit(x_, y)
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

# Predication Resopons
_y= model.predict(x_)
print(_y)

# Test Data
x_new =  
