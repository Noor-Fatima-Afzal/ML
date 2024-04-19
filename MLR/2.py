import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data_dict = {
    'Doors': [3, 4, 5, 3, 2],
    'Cylinders': [4, 6, 4, 4, 6],
    'Gears': [5, 6, 5, 6, 5],
    'Quarterly_Tax': [200, 250, 150, 220, 300],
    'Weight': [1000, 1500, 1200, 1100, 1600],
    'Price': [20000, 25000, 22000, 23000, 26000]
}

df = pd.DataFrame(data_dict)

X = df[['Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight']]
y = df['Price']

regr = LinearRegression()

regr.fit(X, y)

input_features = [[3, 4, 7, 230, 1150]]
predicted = regr.predict(input_features)

print(predicted)

fig, axs = plt.subplots(5, figsize=(5,20))

for i, feature in enumerate(['Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight']):
    axs[i].scatter(df[feature], y)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Price') # y axis par hmaisha Price hogi

plt.tight_layout()
plt.show()