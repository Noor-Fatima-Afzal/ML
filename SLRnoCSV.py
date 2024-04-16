import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = {
  "Area": [420, 380, 390],
  "Price": [50, 40, 45]
}
df = pd.DataFrame(data)
reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df['Price'])
predicted_price = reg.predict([[400]])
print(f"The predicted price is {predicted_price}")