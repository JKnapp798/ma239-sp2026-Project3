#Jake Knapp 
#MA239 Project 3
#Wine Quality Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load wine quality CSV
df = pd.read_csv("WineQuality(ChosenDataSet)/WineQT.csv")

#test first few rows
print(df.head())