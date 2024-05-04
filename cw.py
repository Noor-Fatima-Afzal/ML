import tkinter as tk
import pandas as pd
from sklearn import linear_model
import tkinter.messagebox as tkmessagebox
import matplotlib.pyplot as plt
import numpy as np

window = tk.Tk()
window.title("ECAT Marks Prediction")
window.configure(background="light blue")
window.geometry("900x700")
window.resizable(0, 0)

# Add a title label
title_label = tk.Label(window, text="ML model to predict Ecat marks", font=("Arial", 20))
title_label.pack(pady=10)

df = pd.read_csv("Ecat_2024.csv")

# Standardize 'Gender' column
df['Gender'] = df['Gender'].replace(['male','Male ' 'M', 'm', 'MALE', 'Male'], 'Male')
df['Gender'] = df['Gender'].replace(['female','Female ' 'F', 'f', 'FEMALE', 'Female'], 'Female')

# Drop unnecessary columns
df = df.drop(columns=['Timestamp', 'Name', 'Registration Number'])

# Convert marks columns to numeric type and fill missing values with the mean of the column
marks_columns = ['9th Marks', '10th Marks', '1st Year Marks', '2nd Year Marks', 'ECAT Marks']
for column in marks_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column] = df[column].fillna(df[column].mean())

df['9th Marks'] = pd.to_numeric(df['9th Marks'], errors='coerce')
df['9th Marks'] = df['9th Marks'].astype(float)
df['10th Marks'] = df['10th Marks'].astype(float)
df['1st Year Marks'] = df['1st Year Marks'].astype(float)
df['2nd Year Marks'] = df['2nd Year Marks'].astype(float)

X = df[['9th Marks', '10th Marks', '1st Year Marks', '2nd Year Marks']]
y = df['ECAT Marks']

reg = linear_model.LinearRegression()
reg.fit(X, y)

# Create labels and entries for each feature
features = ['9th Marks', '10th Marks', '1st Year Marks', '2nd Year Marks']
entries = {}
for feature in features:
    label = tk.Label(window, text=f"Enter {feature}:")
    label.pack()
    entry = tk.Entry(window)
    entry.pack()
    entries[feature] = entry

def predict_marks():
    values = [float(entries[feature].get()) for feature in features]
    marks = reg.predict([values])
    result = f"The predicted ECAT marks are {marks}"
    tkmessagebox.showinfo("Predicted Marks", result)

button = tk.Button(window, text="Predict Ecat Marks", command=predict_marks)
button.pack(padx=10, pady=10)

window.mainloop()