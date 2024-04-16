import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("data.csv")

print(df.head())

reg=linear_model.LinearRegression()
reg.fit(df[['Area']], df.Price)
print(reg.predict([[3300]]))

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df.Area, df.Price, color='red', marker='*')
plt.plot(df.Area, reg.predict(df[['Area']]), color='blue')
plt.show()