
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Load the iris dataset into a pandas dataframe
df = pd.read_csv('iris.csv')

# Step 1: Separate the features and target variable into separate dataframes
X = df.drop(['Id', 'variety'], axis=1)
y = df['variety']

# Step 2: Preprocess the data by scaling the features using StandardScaler
scaler = StandardScaler()
preprocessed_data = scaler.fit_transform(X)

# separate the features and target variable into separate dataframes
X = df.iloc[:, 1:5] # select columns 1 through 4 (the features)
y = df.iloc[:, 5] # select column 5 (the target variable)

# print the first 5 rows of the features dataframe
print(X.head())

# print the first 5 rows of the target variable dataframe
print(y.head())


# Step 3: Apply the elbow method to determine the optimal number of clusters for KMeans
# We will use the within-cluster sum of squares (wcss) as the evaluation metric
wcss = [] 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  #k-means++' is a popular method for initializing the centroids to help avoid poor convergence of the clustering algorithm.
    kmeans.fit(preprocessed_data)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_num_clusters = int(input("Enter the optimal number of clusters: "))

print("The optimal number of clusters is:", optimal_num_clusters)

# Step 4: Apply k-means clustering method to the preprocessed data using the optimal number of clusters determined 

kmeans2 = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)
kmeans.fit(preprocessed_data)
cluster_labels = kmeans.labels_

# Print the cluster labels
print(kmeans.labels_)

print(preprocessed_data[:, 1])
plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=cluster_labels, cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 1')
plt.show()

#Feature 1 & 2
#represent the two most important principal components that were derived from the scaled data using PCA
#представляют два наиболее важных основных компонента, которые были получены из масштабированных данных с использованием PCA

#///////////////DBSCAN Method////////////////////////////////////

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(preprocessed_data)

# plot the clusters
plt.scatter(preprocessed_data[:, 0], preprocessed_data[:, 1], c=dbscan_labels )
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
