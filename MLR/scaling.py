import pandas as pd
import matplotlib.pyplot as plt

def min_max_scaler(df):
    scaled_df = (df - df.min()) / (df.max() - df.min())
    return scaled_df

data = pd.read_csv("ToyotaCorolla1.csv")
# Select the columns to scale
columns_to_scale = data.columns[2:5]

# Use the min_max_scaler function
scaled_data = min_max_scaler(data[columns_to_scale])

print("Original data (sample):\n", data[columns_to_scale].head(3))
print("\nScaled data (sample):\n", scaled_data.head(3))

# Create subplots
fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(10, 3*len(columns_to_scale)))

# Plot original and scaled data
for i, column in enumerate(columns_to_scale):
    axs[i, 0].hist(data[column], bins=30)
    axs[i, 0].set_title(f"Original Data: {column}")

    axs[i, 1].hist(scaled_data[column], bins=30)
    axs[i, 1].set_title(f"Scaled Data: {column}")

plt.tight_layout()
plt.show()