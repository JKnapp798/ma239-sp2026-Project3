#Jake Knapp 
#MA239 Project 3
#Wine Quality Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv("WineQuality(ChosenDataSet)/WineQT.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- Cleanup ----

# The id column is not needed 
df = df.drop(columns=["Id"])

# Check for missing values
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True) #fill any empty spots with mean (all values are numertic so no mode needed)

# Separate features, quality is removed as it would be cheating otherwise
features = df.drop(columns=["quality"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- PCA Componet ---

pca = PCA()
pca_components = pca.fit_transform(scaled_features)

# Explained variance
explained_var = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_var)

# Plot explained variance
plt.figure(figsize=(10,6))
plt.scatter(pca_components[:,0], pca_components[:,1],
            c=df['quality'], cmap='viridis', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Wine Features Colored by Quality')
plt.colorbar(label='Quality')
plt.grid(True)
plt.show()

# --- Cumulative variable ---
cumulative_var = np.cumsum(explained_var)
n_components = np.argmax(cumulative_var >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components}")
print("Cumulative explained variance:", cumulative_var)