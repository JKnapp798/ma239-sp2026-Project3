#Jake Knapp 
#MA239 Project 3
#Wine Quality Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv("WineQuality(ChosenDataSet)/WineQT.csv")
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- Cleanup ----

# The id column is not needed 
df = df.drop(columns=["Id"])

# Check for missing values
print("\nThe Following Are Missing Values:\n")
print(df.isnull().sum())
df.fillna(df.mean(), inplace=True) #fill any empty spots with mean (all values are numertic so no mode needed)
print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# Separate features, quality is removed as it would be cheating otherwise
features = df.drop(columns=["quality"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- PCA Componet ---

pca = PCA() #same as PVA via SVD from inclass dataset
pca_components = pca.fit_transform(scaled_features)

# Explained variance
explained_var = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_var)

# Plot explained variance
# plt.figure(figsize=(10,6))
# plt.scatter(pca_components[:,0], pca_components[:,1],
#             c=df['quality'], cmap='viridis', alpha=0.5)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of Wine Features Colored by Quality')
# plt.colorbar(label='Quality')
# plt.grid(True)
# plt.show()

# --- Cumulative variable and plot---
cumulative_var = np.cumsum(explained_var)
n_components = np.argmax(cumulative_var >= 0.95) + 1
print(f"Number of components to explain 95% variance: {n_components}")
print("Cumulative explained variance:", cumulative_var)

# plt.figure(figsize=(8,5))
# plt.plot(np.cumsum(explained_var), marker='o', linestyle='--')
# plt.axhline(y=0.95, color='r', linestyle=':')  #95% line
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA Cumulative Explained Variance')
# plt.grid(True)
# plt.show()


# --- Train/Test split ---

# Features (scaled) and target
X = scaled_features  # already scaled
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


#--- Model Evaluation--- 

# 1. Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# 2. Store evaluation metrics
results = []

for name, model in models.items():
    # Fit model
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Condition number (only for linear models)
    if name in ["Linear Regression", "Ridge Regression"]:
        cond_number = np.linalg.cond(X_train)
    else:
        cond_number = np.nan  # Not defined for Random Forest
    
    # Append results
    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Condition Number": cond_number
    })

# 3. Create comparison table
results_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(results_df)