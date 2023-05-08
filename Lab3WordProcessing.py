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

df = pd.read_csv('iris.csv')

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
vectorized_text_freq = vectorize_text(df['variety'], method='frequency')
vectorized_text_tfidf = vectorize_text(df['variety'], method='tfidf')

# Vectorize using contextualized vector models
vectorized_text_contextualized = contextualize_text(df['variety'])

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
df = pd.read_csv('iris.csv')

# Add a new column with random values
import random
df['class'] = [random.randint(1, 100) for _ in range(len(df))]
print("\nНабор данных IRIS\n")
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
