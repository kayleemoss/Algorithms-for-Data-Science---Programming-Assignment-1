import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Plotting the data for each feature and saving the image
def plotByFeature(df, features):
    sns.pairplot(df, hue='species', vars=features)
    plt.savefig('features.png')

# Sorting the data
def sortData(df, features):
    sorted_data = df.sort_values(by=[features])
    sorted_data.reset_index(inplace=True, drop=True)
    return sorted_data

# Calculating the mahalanobis distance
def mahalanobis(species):
    m = np.mean(species, axis=0)
    diff = species - m
    cov = np.cov(species.T)
    inv_cov = np.linalg.inv(cov)
    mahal_mx = np.dot(np.dot(diff, inv_cov), diff.T)
    dist = mahal_mx.diagonal()
    return dist

# Calculating the critical value
def criticalValue(alpha, n, num_feat):
    p = 1 - alpha
    dfn = num_feat
    dfd = n - num_feat - 1
    inv_f = scipy.stats.f.ppf(p, dfn, dfd)
    num = dfn * (n - 1) ** 2
    denom = n * (dfd) + inv_f
    cv = num / denom
    return cv

# Function to remove the outliers using the critical value and mahalanobis distance
def removeOutliers(species, alpha, num_feat):
    dist = mahalanobis(species)
    cv = criticalValue(alpha, len(species), num_feat)
    outliers = np.where(dist > cv)[0]
    remaining = np.delete(species, outliers, 0)
    return remaining

# Normalizing the data
def normalize(all_species):
    norm = []
    for species in all_species:
        f_min = np.amin(species, axis=0)
        f_max = np.amax(species, axis=0)
        diff = f_max - f_min
        n = (species - f_min) / diff
        norm.append(n)
    return norm

# Calculate the Fisher’s Linear Discriminant Ratio
def fdr(species, features):
    indices = [*range(3)]
    fdr = 0
    for index1 in indices:
        iris1 = species[index1]
        for index2 in indices:
            iris2 = species[index2]
            if (np.array_equal(iris1, iris2)):
                continue
            else:
                fdr_n = (np.mean(iris1, axis=0) - np.mean(iris2, axis=0)) ** 2
                fdr_d = np.var(iris1, axis=0) + np.var(iris2, axis=0)
                fdr += fdr_n / fdr_d
        indices.pop(0)
    fdr_df = pd.DataFrame(fdr, index=features, columns=['FDR'])
    return fdr_df

# Read in iris data
df_iris = pd.read_csv("iris.csv")

# Storing the data in numpy arrays according to species
setosa = df_iris[df_iris['species']=='setosa']
versicolor = df_iris[df_iris['species']=='versicolor']
virginica = df_iris[df_iris['species']=='virginica']

setosa_values = setosa.select_dtypes(include='number')
df = setosa_values.to_numpy()

versi_values = versicolor.select_dtypes(include='number')
df2 = versi_values.to_numpy()

virg_values = virginica.select_dtypes(include='number')
df3 = virg_values.to_numpy()

feature_names = ['sepal_length','sepal_width','petal_length','petal_width']

# Sorting the data
for item in feature_names:
        sorted_iris = sortData(df_iris, item)

# Plot the data by feature
plotByFeature(df_iris,feature_names)

# finding obvious outliers with a boxplot
def graph(y):
    sns.boxplot(x="species", y=y, data=df_iris)
    plt.savefig('original.png')

plt.figure(figsize=(10, 10))

plt.subplot(221)
graph('sepal_length')

plt.subplot(222)
graph('sepal_width')

plt.subplot(223)
graph('petal_length')

plt.subplot(224)
graph('petal_width')

# Remove outliers for each species store in pd dataframe and numpy for calculations
ro1 = removeOutliers(df, 0.5, 4)
new_set = pd.DataFrame(ro1, columns=['sepal_length','sepal_width','petal_length','petal_width'])
np_set = new_set.to_numpy()
new_set['species']='setosa'

ro2 = removeOutliers(df2, 0.5, 4)
new_vers = pd.DataFrame(ro2, columns=['sepal_length','sepal_width','petal_length','petal_width'])
np_vers = new_vers.to_numpy()
new_vers['species']='versicolor'

ro3 = removeOutliers(df3, 0.5, 4)
new_virg = pd.DataFrame(ro3, columns=['sepal_length','sepal_width','petal_length','petal_width'])
np_virg = new_virg.to_numpy()
new_virg['species']='virginica'

no_outliers = [np_set, np_vers, np_virg]
pd_no_outliers = pd.concat([new_set, new_vers, new_virg], ignore_index = True)


# New boxplots with removed outliers
def graph(y):
    sns.boxplot(x="species", y=y, data=pd_no_outliers)
    plt.savefig('no_outliers.png')

plt.figure(figsize=(10, 10))

plt.subplot(221)
graph('sepal_length')

plt.subplot(222)
graph('sepal_width')

plt.subplot(223)
graph('petal_length')

plt.subplot(224)
graph('petal_width')

# normalizing the data and determining the FDR
normal_species = normalize(no_outliers)
feature_fdr = fdr(normal_species,feature_names)
feature_fdr.to_csv('fdr_results.csv')

# write the sorted results to a csv file
sorted_iris.to_csv('sorted_results.csv')
