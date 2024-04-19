import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pandas.read_csv("ToyotaCorolla1.csv")

X = df[['Doors', 'Cylinders','Gears','Quarterly_Tax','Weight']]
y = df['Price']

## Missing thing is train test split --->
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## print the shape of X_train, X_test, y_train, y_test
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# (1148{sample},5{featers})

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
predicted = regr.predict(X_test)
print(predicted)

fig, axs = plt.subplots(5, figsize=(5,20))

for i, feature in enumerate(['Doors', 'Cylinders', 'Gears', 'Quarterly_Tax', 'Weight']):
    axs[i].scatter(df[feature], y)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Price') # y axis par hmaisha Price hogi

plt.tight_layout()
plt.show()