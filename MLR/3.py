import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 1: Data Collection
df = pd.read_csv("ToyotaCorolla1.csv")

# Step 2: Split the Data
X = df[['Doors', 'Cylinders','Gears','Quarterly_Tax','Weight']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection
model = LinearRegression()

# Step 4: Train the Model
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Step 6: Prediction
input_features = [[3,4,7,230,1150]]
predicted = model.predict(input_features)
print("Predicted price:", predicted)
