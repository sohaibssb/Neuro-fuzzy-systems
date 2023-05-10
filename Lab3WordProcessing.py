# Lab #3
# Word processing
# vectorize texts using two methods:
# - frequency or tf*idf;
# - static or contextualized vector models.
# The resulting text vectors must either be clustered. On the resulting clusters, it is necessary to train the clustering method and check the accuracy of its work.
# It is necessary to compare the accuracy of the method with and without the use of morphological analysis.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import spacy

df = pd.read_csv('TextEmotion.csv')

#vectorizing the text using frequency, tf*idf methods

def vectorize_text(texts, method='frequency'):
    if method == 'frequency':
        vectorizer = CountVectorizer()
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
    
    vectorized_text = vectorizer.fit_transform(texts)
    
    return vectorized_text.toarray()

##vectorizing the text using static, contextualized vector models methods

def contextualize_text(texts):
    nlp = spacy.load('en_core_web_md')
    vectors = []
    
    for text in texts:
        doc = nlp(text)
        vector = doc.vector
        vectors.append(vector)
        
    return vectors

#vectorize the wine descriptions:

#Vectorize using frequency and tfidf
vectorized_text_freq = vectorize_text(df['content'], method='frequency')
vectorized_text_tfidf = vectorize_text(df['content'], method='tfidf')

# Vectorize using contextualized vector models
vectorized_text_contextualized = contextualize_text(df['content'])

#///////////////////////////////////////////////////////////////////
#cluster the resulting vectors using the KMeans clustering algorithm

def cluster_vectors(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    
    return labels

# Cluster the frequency or tfidf vectors
labels_freq = cluster_vectors(vectorized_text_freq, n_clusters=3)
labels_tfidf = cluster_vectors(vectorized_text_tfidf, n_clusters=3)

# Cluster the contextualized vectors
labels_contextualized = cluster_vectors(vectorized_text_contextualized, n_clusters=3)


df['class'] = labels_contextualized 

#////////////////////////////////////////////////////
import pandas as pd

# Load the iris dataset
df = pd.read_csv('TextEmotion.csv')

# Add a new column with random values
import random
df['class'] = [random.randint(1, 100) for _ in range(len(df))]
print("\nНабор данных TextEmotion\n")
# Print the dataset with the added column
print(df)
#////////////////////////////////////////////////////

# Evaluate accuracy of clustering method without morphological analysis
accuracy_without_morphology = accuracy_score(df['class'], labels_freq)

# Evaluate accuracy of clustering method with morphological analysis
accuracy_with_morphology = accuracy_score(df['class'], labels_contextualized)
print("\n")
print(f"Точность без морфологического анализа: {accuracy_without_morphology}")
print("")
print(f"Точность с морфологическим анализом: {accuracy_with_morphology}")
print("")

# Cluster the frequency or tfidf vectors
labels_freq = cluster_vectors(vectorized_text_freq, n_clusters=3)
labels_tfidf = cluster_vectors(vectorized_text_tfidf, n_clusters=3)

# Cluster the contextualized vectors
labels_contextualized = cluster_vectors(vectorized_text_contextualized, n_clusters=3)

# Print the labels for each method
#print("Labels for frequency method:", labels_freq)
#print("Labels for tfidf method:", labels_tfidf)
#print("Labels for contextualized method:", labels_contextualized)


# Assign the labels to a new column in the dataframe
df['labels_freq'] = labels_freq
df['labels_tfidf'] = labels_tfidf
df['labels_contextualized'] = labels_contextualized

# Print the dataframe with the new labels columns
print(df)

# import matplotlib.pyplot as plt

# # Plot the contextualized vectors with different colors for different clusters
# plt.scatter([v[0] for v in vectorized_text_contextualized], [v[1] for v in vectorized_text_contextualized], c=labels_contextualized)

# # Set the plot title and axes labels
# plt.title('Contextualized Vector Clustering')
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Choose the number of clusters
num_clusters = 3

# Cluster the contextualized vectors
kmeans = KMeans(n_clusters=num_clusters)
labels_contextualized = kmeans.fit_predict(vectorized_text_contextualized)

# # Plot the contextualized vectors with different colors for different clusters
# plt.scatter([v[0] for v in vectorized_text_contextualized], [v[1] for v in vectorized_text_contextualized], c=labels_contextualized)

# # Print the KMeans model
# print(kmeans)

# # Set the plot title and axes labels
# plt.title('Contextualized Vector Clustering')
# plt.xlabel('X')
# plt.ylabel('Y')

# # Show the plot
# plt.show()


#//////////////////////////////////////////////////////////////////


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Perform PCA on the contextualized vectors
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectorized_text_contextualized)

# # Plot the PCA result
# plt.scatter(pca_result[:, 0], pca_result[:, 1])
# plt.title('PCA Result')
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.show()

# Use the elbow method to choose the number of clusters
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(vectorized_text_contextualized)
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Искажение')
plt.title('Elbow Method')
plt.show()

# Ask the user for the number of clusters
num_clusters = int(input('Введите количество кластеров: '))

# Cluster the contextualized vectors
kmeans = KMeans(n_clusters=num_clusters)
labels_contextualized = kmeans.fit_predict(vectorized_text_contextualized)

# Plot the contextualized vectors with different colors for different clusters
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels_contextualized)
plt.title('Контекстуальная векторная кластеризация')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# Print the KMeans model
print(kmeans)


