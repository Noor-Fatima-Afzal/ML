import tkinter as tk
import pandas as pd
from sklearn import linear_model
import tkinter.messagebox as tkmessagebox
import matplotlib.pyplot as plt
import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.residual_ = None
        self.RSS = None
        self.TSS = None
        self.Rsquared_ = None
        
    def fit(self, x, y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
    
    # ab slope ka formula lgana hai
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        self.slope_ = numerator / denominator
    # ab intercept ka formula lgana hai
        self.intercept_ = y_mean - self.slope_ * x_mean
    # ab y=mx+c ka formula lgana hai
        y_pred = self.intercept_ + self.slope_ * x
        self.residual_ = y - y_pred
    # ab RSS ka formula lgana hai
        self.RSS = np.sum(self.residual_ ** 2)
    # ab TSS ka formula lgana hai
        self.TSS = np.sum((y - y_mean) ** 2)
    # ab R^2 ka formula lgana hai
        self.Rsquared_ = 1 - (self.RSS / self.TSS)
    
    def predict(self, x):
        return self.intercept_ + self.slope_ * x
    
data = pd.read_csv("ToyotaCorolla1.csv")

x = data['KM'].values
y = data['Price'].values

model = SimpleLinearRegression()
model.fit(x, y)

pred = model.predict(x)

plt.scatter(x, y, label='Observed values')
plt.plot(x, pred, color='red', label='Regression line')
plt.xlabel('KM')
plt.ylabel('Price')
plt.legend()
plt.show()