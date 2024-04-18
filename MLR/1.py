import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt


df = pandas.read_csv("ToyotaCorolla1.csv")

X = df[['Doors', 'Cylinders','Gears','Quarterly_Tax','Weight']]
y = df['Price']

regr = linear_model.LinearRegression()
regr.fit(X, y)

predicted = regr.predict([[3,4,7,230,1150]])

print(predicted)

fig, axs = plt.subplots(5, figsize=(5,20))

for i, feature in enumerate(['Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight']):
    axs[i].scatter(df[feature], y)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Price') # y axis par hmaisha Price hogi

plt.tight_layout()
plt.show()