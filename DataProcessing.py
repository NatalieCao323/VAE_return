import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data (assuming 'stock_returns.csv' has a column 'RETX' for returns)
data = pd.read_csv('stock_returns.csv', low_memory=False)

# Clean data
# This will be specific to your data. For example, you might need to:
data.dropna(inplace=True)  # Drop missing values
data = data[data['RETX'].between(-1, 1)]  # Filter outliers

# Normalize data
scaler = StandardScaler()
returns = data['RETX'].values.reshape(-1, 1)  # Reshape to 2D array for scaling
returns_scaled = scaler.fit_transform(returns)

# Add the scaled returns to the original DataFrame
data['return_scaled'] = returns_scaled.ravel()  # Flatten the 2D array back to 1D

# Assuming you have sequential data and you want to create a sequence length of 60
sequence_length = 60
samples = len(data) - sequence_length + 1
inputs = np.zeros((samples, sequence_length))
# Export the scaler, mean, and standard deviation
mean_value = scaler.mean_[0]
std_dev_value = np.sqrt(scaler.var_[0])

for i in range(samples):
    inputs[i] = data['return_scaled'].iloc[i:i+sequence_length]

# Now `inputs` is a numpy array that you can pass into the VAE
