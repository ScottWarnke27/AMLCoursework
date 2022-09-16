import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#read in the piazza data
piazza_data = pd.read_csv("./labData.csv")

#split the data into x and y
x = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values

#split out train and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=1693)

#import and run a linear regression model on the piazza data
from sklearn.linear_model import LinearRegression

#fitting a linear regression to the training data
regression = LinearRegression()
regression.fit(x_train, y_train)

#look at the data
plt.scatter(x_train, y_train)
plt.show()

#look at the data with the regression prediction (in red)
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regression.predict(x_train), color="red")
plt.show()

#the same look from above, but with labels
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regression.predict(x_train), color="red")
plt.title("Piazza Contributions and Grades (Training Data)")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.show()

#now show the training data (in black) and the testing data (in blue)
plt.scatter(x_train, y_train, color="black")
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, regression.predict(x_train), color="red")
plt.title("Piazza Contributions and Grades")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.show()

#now look at all of the variables for linear regression, multiple linear regression
x_mlr = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
y_mlr = piazza_data[["Grade"]].values
x_train_mlr, x_test_mlr, y_train_mlr, y_test_mlr = train_test_split(x_mlr, y_mlr, test_size=0.25, random_state=1693)

from sklearn.preprocessing import StandardScaler

scale_x_mlr = StandardScaler()
x_train_mlr = scale_x_mlr.fit_transform(x_train_mlr)
x_test_mlr = scale_x_mlr.transform(x_test_mlr)

multiple_regression = LinearRegression()
multiple_regression.fit(x_train_mlr, y_train_mlr)

#the next two lines will calculate the predictions based on mlr and compare it to the test ys
y_predictions_mlr = multiple_regression.predict(x_test_mlr)
[y_test_mlr, y_predictions_mlr]


#Now try polynomial regression
from sklearn.preprocessing import PolynomialFeatures
xpoly = piazza_data[["contributions", "days online", "views", "questions", "answers"]].values
ypoly = piazza_data[["Grade"]].values

poly_data = PolynomialFeatures(degree = 2)
x_poly = poly_data.fit_transform(xpoly)

x_train_poly, x_test_poly, y_train_poly, y_test_poly = train_test_split(x_poly, ypoly, test_size = 0.25, random_state = 1693)
poly_reg = LinearRegression()
poly_reg.fit(x_train_poly, y_train_poly)

y_predictions = poly_reg.predict(x_test_poly)



#we can also run linear regression and polynomial regression, then compare them to the truth
x = piazza_data[["contributions"]].values
y = piazza_data[["Grade"]].values

#split out training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=1693)

#scale the x data
scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)

#linear regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

#polynomial regression
poly_data = PolynomialFeatures(degree =2)
poly_reg = LinearRegression()
poly_reg.fit(poly_data.fit_transform(x_train), y_train)

#plot the results of the linear and polynomial regression to compare with the truth
plt.scatter(x_test, y_test, color="black", label = "Truth")
plt.scatter(x_test, lin_reg.predict(x_test), color = "green", label="Linear")
plt.scatter(x_test, poly_reg.predict(poly_data.fit_transform(x_test)), color="blue", label = "Poly")
plt.xlabel("Piazza Contributions")
plt.ylabel("Grade")
plt.title("Linear Regression v Polynomial Regression v Truth")
plt.legend()
plt.show()