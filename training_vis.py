import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

file_path = r'swin_training_results\swinn_bcss512_conv_training_results.txt'

# Read the data assuming it's CSV or tab-separated
df = pd.read_csv(file_path)

# Check if the data is loaded correctly
print(df.head())  # Optional: View the first few rows to verify

# Create subplots
fig, axes = plt.subplots(figsize=(10, 5))

# Plot Loss curves
axes.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='blue', marker='o')
axes.plot(df['Epoch'], df['Val Loss'], label='Validation Loss', color='red', marker='o')
axes.set_xlabel('Epoch')
axes.set_ylabel('Loss')
axes.set_title('BCSS512: Train vs Validation Loss')
axes.legend()

# Show the plot
plt.tight_layout()
plt.show()