'''
Лабораторная работа №1
метода k-средних и метода DBSCAN.
метод локтя.
RAND - обязателен при наличии целевой переменной.
Для методов из второй группы необходимо выбрать ε и другие параметры метода.
Требуется сравнить визуализацию данных с использованием методов PCA, MDS 
с t-SNE, UMAP с разметкой по полученным кластерам.

Lab #1
k-means method and DBSCAN method.
elbow method.
RAND - required if there is a target variable.
For methods from the second group, it is necessary to choose ε and other parameters of the method.
It is required to compare data visualization using PCA, MDS methods 
with t-SNE, UMAP with labeling based on received data clusters.

'''


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt


df = pd.read_csv('iris.csv')

# Separate the features and target variable into separate dataframes
X = df.drop(['Id','variety'], axis=1)
y = df['variety']

# scaling the features using StandardScaler
scaler = StandardScaler()
preprocessed_data = scaler.fit_transform(X)

print("\nPreprocessed Data After Scaling:\n")
print(preprocessed_data)
print("\n")

# separate the features and target variable into separate dataframes
X = df.iloc[:, 1:5] # select columns 1 through 4 (the features)
y = df.iloc[:, 5] # select column 5 (the target variable)

# print the first 5 rows of the features dataframe
print("\nThe Features:\n")
print(X.head())

# print the first 5 rows of the target variable dataframe
print("\nThe Target Variable:\n")
print(y.head())

#////////////////

from sklearn.decomposition import PCA 
 
pca = PCA(n_components=2) 
 
principalComponents = pca.fit_transform(preprocessed_data) 
 
principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2']) 
final_df = pd.concat([df['variety'], principalDf], axis=1) 
final_df = final_df.dropna() 
final_df[:10]
print(final_df)


#plt.scatter(x=final_df['PCA1'],y=final_df['PCA2'],c=final_df['variety'])
#////////////////



# Apply the elbow method to determine the optimal number of clusters for KMeans
# We will use the within-cluster sum of squares (wcss) as the evaluation metric
wcss = [] 


for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  #k-means++' is a popular method for initializing the centroids to help avoid poor convergence of the clustering algorithm.
        kmeans.fit(preprocessed_data)
        wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=3) 
kmeans.fit(preprocessed_data) 
df['kmeans']=kmeans.labels_ 
plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1],c= df['kmeans'])



optimal_num_clusters = int(input("Enter the optimal number of clusters: "))
print("\n")
kmeans = KMeans(n_clusters=optimal_num_clusters)
print("The optimal number of clusters is:", optimal_num_clusters)
print("\n")
# Step 4: Apply k-means clustering method to the preprocessed data using the optimal number of clusters determined 

kmeans2 = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
kmeans.fit(preprocessed_data)
cluster_labels = kmeans.labels_

# Print the cluster labels
print("\nCluster KMeans Labels:\n")
print(kmeans.labels_) # each point in dataset assign it to cluster label
print("\nThe Length are:\n")
print(len(kmeans.labels_))
print("\n")

print(preprocessed_data[:, 1])
plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('sepalLength')
plt.ylabel('sepalWidth')
plt.show()

#Feature 1 & 2
#represent the two most important principal components that were derived from the scaled data using PCA
#представляют два наиболее важных основных компонента, которые были получены из масштабированных данных с использованием PCA

#///////////////DBSCAN Method////////////////////////////////////

dbscan = DBSCAN(eps=0.5, min_samples=5) 
#eps: radius of each neighborhood around a point
#min_samples: sets the minimum number of points required in a neighborhood for it to be considered a core point
dbscan_labels = dbscan.fit_predict(preprocessed_data)

# plot the clusters
plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=dbscan_labels )
plt.title('DBSCAN Clustering')
plt.xlabel('sepalLength')
plt.ylabel('sepalWidth')
plt.show()

#/////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////

#Analysis (PCA) and Multidimensional Scaling (MDS) methods to visualize the data with clusters obtained from k-means and DBSCAN clustering algorithms


from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import numpy as np

#Create an instance of PCA and MDS:

pca = PCA(n_components=1) # To 2 Dimensions
mds = MDS(n_components=2)

#Fit and transform the preprocessed_data using PCA and MDS:

pca_transformed_data = pca.fit_transform(preprocessed_data)
mds_transformed_data = mds.fit_transform(preprocessed_data)

#Visualize the data with k-means clusters using PCA and MDS:

#plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=cluster_labels)
plt.scatter(pca_transformed_data[:, 0], np.zeros(len(preprocessed_data)), c=cluster_labels)

plt.title("PCA with k-means clusters")
plt.xlabel("PC-sepalLength")
#plt.ylabel("PC-sepalWidth")
plt.show()

plt.scatter(mds_transformed_data[:, 0], mds_transformed_data[:, 1], c=cluster_labels)
plt.title("MDS with k-means clusters")
plt.xlabel("MDS-sepalLength")
plt.ylabel("MDS-sepalWidth")
plt.show()

#Visualize the data with DBSCAN clusters using PCA and MDS:

#plt.scatter(pca_transformed_data[:, 0], pca_transformed_data[:, 1], c=dbscan_labels)
plt.scatter(pca_transformed_data[:, 0], np.zeros(len(preprocessed_data)), c=dbscan_labels)
plt.title("PCA with DBSCAN clusters")
plt.xlabel("PC-sepalLength")
plt.ylabel("PC-sepalWidth")
plt.show()

plt.scatter(mds_transformed_data[:, 0], mds_transformed_data[:, 1], c=dbscan_labels)
plt.title("MDS with DBSCAN clusters")
plt.xlabel("MDS-sepalLength")
plt.ylabel("MDS-sepalWidth")
plt.show()


#/////////////////////////////////////////////////////////////////
#/////////////////////////////////////////////////////////////////

#Use Uniform Manifold Approximation and Projection (UMAP) and t-Distributed Stochastic Neighbor Embedding (t-SNE) methods to visualize the data with clusters obtained from k-means and DBSCAN clustering algorithms

import umap
from sklearn.manifold import TSNE

#Fit and transform the preprocessed data using the k-means clustering algorithm:

kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(preprocessed_data)

#Fit and transform the preprocessed data using the DBSCAN clustering algorithm:

dbscan = DBSCAN(eps=.5, min_samples=5)
cluster_labels = dbscan.fit_predict(preprocessed_data)

#Fit and transform the preprocessed data using UMAP:

umap_data = umap.UMAP(n_neighbors=15, min_dist=.1, metric="euclidean").fit_transform(preprocessed_data)

#Fit and transform the preprocessed data using t-SNE:

tsne_data = TSNE(n_components=2, perplexity=30, metric="euclidean").fit_transform(preprocessed_data) #preplexity: controls the balance between local and global aspects of the data

#Plot the data with cluster labels using UMAP and t-SNE:

plt.scatter(umap_data[:,0], umap_data[:,1], c=cluster_labels, cmap='viridis')
plt.title('UMAP')
plt.show()

plt.scatter(tsne_data[:,0], tsne_data[:,1], c=cluster_labels, cmap='viridis')
plt.title('t-SNE')
plt.show()

#/////////////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Plot the first visualization in the top-left subplot
axs[0, 0].scatter(pca_transformed_data[:, 0], np.zeros(len(preprocessed_data)), c=cluster_labels)
axs[0, 0].set_title("PCA with k-means clusters")
axs[0, 0].set_xlabel("PC-sepalLength")
axs[0, 0].set_ylabel("PC-sepalWidth")

# Plot the second visualization in the top-right subplot
axs[0, 1].scatter(mds_transformed_data[:, 0], mds_transformed_data[:, 1], c=cluster_labels)
axs[0, 1].set_title("MDS with k-means clusters")
axs[0, 1].set_xlabel("MDS-sepalLength")
axs[0, 1].set_ylabel("MDS-sepalWidth")

# Plot the third visualization in the bottom-left subplot
axs[1, 0].scatter(umap_data[:, 0], umap_data[:, 1], c=cluster_labels)
axs[1, 0].set_title("UMAP with k-means clusters")
axs[1, 0].set_xlabel("UMAP-sepalLength")
axs[1, 0].set_ylabel("UMAP-sepalWidth")

# Plot the fourth visualization in the bottom-right subplot
axs[1, 1].scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_labels)
axs[1, 1].set_title("t-SNE with k-means clusters")
axs[1, 1].set_xlabel("t-SNE-sepalLength")
axs[1, 1].set_ylabel("t-SNE-sepalWidth")

plt.show()
