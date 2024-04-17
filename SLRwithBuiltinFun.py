import tkinter as tk
import pandas as pd
from sklearn import linear_model
import tkinter.messagebox as tkmessagebox
import matplotlib.pyplot as plt

window = tk.Tk()
window.title("Simple Linear Regression")
window.configure(background="light blue")
window.geometry("900x700")
window.resizable(0, 0)

df = pd.read_csv("data.csv")
reg = linear_model.LinearRegression()
reg.fit(df[['Area']], df['Price'])

label = tk.Label(window, text="Enter Area:")
label.pack()
entry = tk.Entry(window)
entry.pack()

def predict_price():
    area = float(entry.get())
    price = reg.predict([[area]])
    result = f"The predicted price for an area of {area} is {price}"
    tkmessagebox.showinfo("Predicted Price", result)

def show_graph():
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.scatter(df.Area, df.Price, color='red', marker='*')
    plt.plot(df.Area, reg.predict(df[['Area']]), color='blue')
    plt.show()

button = tk.Button(window, text="Predict Price", command=predict_price)
button.pack(padx=10, pady=10)

button = tk.Button(window, text="Show graphically", command=show_graph)
button.pack(padx=10, pady=10)

window.mainloop()