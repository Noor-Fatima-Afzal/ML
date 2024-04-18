import pandas
from sklearn import linear_model

df = pandas.read_csv("ToyotaCorolla1.csv")

X = df[['Doors', 'Cylinders','Gears','Quarterly_Tax','Weight']]
y = df['Price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
# print(regr.coef_): This line prints the coefficients of the regression equation. These coefficients represent the change in the target variable for a one-unit change in the corresponding predictor variable, assuming all other predictor variables are held constant. For example, regr.coef_[0] is the change in 'Price' for a one-unit change in 'Doors'.