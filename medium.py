import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class SimpleLinearRegression:
    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.residual_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None
        
    def fit(self, x, y):
        # Calculate the mean of the input (x) and output data (y)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Calculate the terms needed for the slope (b1) and intercept (bo) of the regression line 
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        # Calculate the slope (b1) and intercept (bo) of the regression line (regression equation)
        self.slope_ = numerator / denominator
        self.intercept_ = y_mean - self.slope_ * x_mean
        
        y_pred = self.intercept_ + self.slope_ * x
        self.residual_ = y - y_pred
        
        self.RSS = np.sum(self.residual_ ** 2)
        self.TSS = np.sum((y - y_mean) ** 2)
        self.r2score_ = 1 - (self.RSS / self.TSS)
        
    def predict(self, x):
        return self.intercept_ + self.slope_ * x


d = {'x': [1, 2, 3, 4, 5], 'y': [0.9, 2.5, 3.6, 3.5, 4.6]}
df = pd.DataFrame(data=d)

print(type(df))
print(df.columns)
print(df.ndim)
print(df.shape)
print(df.dtypes)

df

X = df.iloc[:, :1].values  
y = df.iloc[:, -1].values

print(type(X))
print(X.ndim)
print(X.shape)

print(type(y))
print(y.ndim)
print(y.shape)

x = X.flatten()

model = SimpleLinearRegression()
model.fit(x, y)
pred = model.predict(x)

print(f"Simple linear equation: y = {model.intercept_:.2f} + {model.slope_:.2f}x")
print(f"Slope: {model.slope_:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Residual: {model.residual_}")
print(f"Residual sum of squares (RSS): {model.RSS:.2f}")
print(f"Total sum of squares (TSS): {model.TSS:.2f}")
print(f"Coefficient of determination (R^2): {model.r2score_:.2f}")

plt.scatter(x, y, label='Observed values')
plt.plot(x, pred, color='red', marker='o', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()