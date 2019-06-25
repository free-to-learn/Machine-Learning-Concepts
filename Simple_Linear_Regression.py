import numpy as np
from sklearn.linear_model import LinearRegression
x = np.array([5,15,25,35,35,55]).reshape((-1,1))
y = np.array([5,20,14,32,22,38])
#print(x,'\n',y)


# Model Selection
model = LinearRegression()
model.fit(x,y)
model=LinearRegression().fit(x, y)
print(model)

# find the Coefficents
r_sq = model.score(x,y)
print("Coefficents of X,Y",r_sq)

# find the interception
print("Intercept:",model.intercept_)
print("Slope:",model.coef_)


# Passing the Data to model

new_model = LinearRegression().fit(x,y.reshape(-1,1))
print("Intercepts:",new_model.intercept_)
print('slope',new_model.coef_)

# Find the Y values
y_pred=model.predict(x)
print('Predicted Response',y_pred)
# OR
y_pred=model.intercept_+model.coef_*x
print("Predicted Response",y_pred,sep='\n')


# New Input to Test
x_new=np.arange(6).reshape((-1,1))
#print(x_new)

y_new=model.predict(x_new)
print(y_new)

# Data Visulization
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x_new,y_new)
plt.show()
