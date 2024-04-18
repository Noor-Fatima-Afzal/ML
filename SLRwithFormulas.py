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

data={'x':[1,2,3,4,5],'y':[0.9,2.5,3.6,3.5,4.6]}
df=pd.DataFrame(data)

x=df.iloc[:, :1].values # saari rows aur sirf 1st column
x=x.flatten() # 1D array banane ke liye
y=df.iloc[:, -1].values # saari rows aur sirf last column
    
model = SimpleLinearRegression()
model.fit(x, y)
pred = model.predict(x)

plt.scatter(x, y, label='Observed values')
plt.plot(x, pred, color='red', marker='o', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
        
