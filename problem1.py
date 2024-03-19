import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt

# read in the csv file
df_iris = pd.read_csv("iris.csv")

# store in data frames according to class
setosa = df_iris[df_iris['species']=='setosa']
versicolor = df_iris[df_iris['species']=='versicolor']
virginica = df_iris[df_iris['species']=='virginica']

# minimum of the features for each class
min_setosa = setosa.min(numeric_only=True)
min_versicolor = versicolor.min(numeric_only=True)
min_virginica = virginica.min(numeric_only=True)

print(f"Minimum:\nSetosa\n{min_setosa}\nVersicolor\n{min_versicolor}\nVirginica\n{min_virginica}\n")

# maximum of the features for each class
max_setosa = setosa.max(numeric_only=True)
max_versicolor = versicolor.max(numeric_only=True)
max_virginica = virginica.max(numeric_only=True)

print(f"Maximun:\nSetosa\n{max_setosa}\nVersicolor\n{max_versicolor}\nVirginica\n{max_virginica}\n")

# mean of the features for each class
mean_setosa = setosa.mean(numeric_only=True)
mean_versicolor = versicolor.mean(numeric_only=True)
mean_virginica = virginica.mean(numeric_only=True)

print(f"Mean:\nSetosa\n{mean_setosa}\nVersicolor\n{mean_versicolor}\nVirginica\n{mean_virginica}\n")

# 10% trimmed mean of the features for each class
setosa_features = setosa.select_dtypes(include=np.number)
versicolor_features = versicolor.select_dtypes(include=np.number)
virginia_features = virginica.select_dtypes(include=np.number)

trim_setosa = stats.trim_mean(setosa_features, 0.1)
trim_versicolor = stats.trim_mean(versicolor_features, 0.1)
trim_virginica = stats.trim_mean(virginia_features, 0.1)

print(f"10% Trimmed Mean:\nSetosa\n{trim_setosa}\nVersicolor\n{trim_versicolor}\nVirginica\n{trim_virginica}\n")

# standard deviation of the features for each class
std_setosa = setosa.std(numeric_only=True)
std_versicolor = versicolor.std(numeric_only=True)
std_virginica = virginica.std(numeric_only=True)

print(f"Standard Deviation:\nSetosa\n{std_setosa}\nVersicolor\n{std_versicolor}\nVirginica\n{std_virginica}\n")

# skewness of the features for each class
skew_setosa = setosa.skew( numeric_only=True)
skew_versicolor = versicolor.skew(numeric_only=True)
skew_virginica = virginica.skew(numeric_only=True)

print(f"Skewness:\nSetosa\n{skew_setosa}\nVersicolor\n{skew_versicolor}\nVirginica\n{skew_virginica}\n")

# kurtosis of the features for each class
kurt_setosa = setosa.kurt(numeric_only=True)
kurt_versicolor = versicolor.kurt(numeric_only=True)
kurt_virginica = virginica.kurt(numeric_only=True)

print(f"Kurtosis:\nSetosa\n{kurt_setosa}\nVersicolor\n{kurt_versicolor}\nVirginica\n{kurt_virginica}\n")


