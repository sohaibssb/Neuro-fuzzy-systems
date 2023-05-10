
# Lab #3
# Word processing
# vectorize texts using two methods:
# - frequency or tf*idf;
# - static or contextualized vector models.
# The resulting text vectors must either be clustered. On the resulting clusters, it is necessary to train the clustering method and check the accuracy of its work.
# It is necessary to compare the accuracy of the method with and without the use of morphological analysis.

import pandas as pd
import string as st
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import numpy as np
import re
from nltk import PorterStemmer, WordNetLemmatizer

df = pd.read_csv('TextEmotion.csv')
df = df.drop(columns=['tweet_id'])
# print(df[:10])


def remove_punct(text):
    if pd.isna(text):
        return text
    return "".join([ch for ch in text if ch not in st.punctuation])

df['removed_punc'] = df['content'].apply(lambda x:remove_punct(x))
# print(df.head())

def tokenize(text):
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

df['tokens'] = df['removed_punc'].apply(lambda msg : tokenize(msg))
# print(df.head())

def remove_small_words(text):
    return [x for x in text if len(x) > 3 ]

df['larger_tokens'] = df['tokens'].apply(lambda x : remove_small_words(x))
# print(df.head())


def remove_stopwords(text):
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

df['clean_tokens'] = df['larger_tokens'].apply(lambda x : remove_stopwords(x))
# print(df.head())

def stemming(text):
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

df['stem_words'] = df['clean_tokens'].apply(lambda wrd: stemming(wrd))
# print(df.head())

def lemmatize(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

df['lemma_words'] = df['stem_words'].apply(lambda x : lemmatize(x))
# print(df.head())

def return_sentences(tokens):
    return " ".join([word for word in tokens])

df['clean_text'] = df['lemma_words'].apply(lambda x : return_sentences(x))
print(df.head())


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
text_train, text_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], random_state=0, train_size = .75)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(text_train)
X_test = tfidf.transform(text_test)
print(X_train.toarray())
print(X_train.shape)
print(X_test.toarray())
print(X_test.shape)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X_train.toarray())

principalDf = pd.DataFrame(data = principalComponents, columns = ['PCA1', 'PCA2'])
final_df = pd.concat([df['sentiment'], principalDf], axis=1)
final_df = final_df.dropna()
final_df.tail(10)

import matplotlib.pyplot as plt
def visualize(xlabel,ylabel,title,X):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)#, ,projection='3d'
    ax.set_xlabel(xlabel, fontsize = 15)
    ax.set_ylabel(ylabel, fontsize = 15)
    # ax.set_zlabel('PCA3', fontsize = 15) 
    ax.set_title(title, fontsize = 20)
    types = ['empty', 'sadness', 'enthusiasm', 'neutral', 'worry', 'surprise','love', 'fun', 'hate', 'happiness', 'boredom', 'relief', 'anger']
    colors = ['y', 'b', 'g', 'r', 'c', 'm', 'k', '#FFA07A', '#00CED1', '#800080', '#FFD700', '#008000']
    markers = ['o', 's', 'v', '^', '*', 'x', 'D', 'P', 'h', '8', '>', '<', 'p']
    for type,color,marker  in zip(types,colors,markers):
        indicesToKeep = X['sentiment'] == type
        ax.scatter(X.loc[indicesToKeep, xlabel]
                , X.loc[indicesToKeep, ylabel]
                # , final_df.loc[indicesToKeep, 'PCA3']
                , c = color
                ,marker = marker
                , s = 50)
    ax.legend(types)
    ax.grid()


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=3)
kmeans.fit(final_df[['PCA1','PCA2']])
final_df['kmeans']=kmeans.labels_
print(final_df)



plt.scatter(x=final_df['PCA1'],y=final_df['PCA2'],c=final_df['kmeans'])


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def optimise_k_means(data,max_k):
    means = []
    inertias = []
    
    for k in range(1,max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
    
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means,inertias,('o-'))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


    final_df.dropna()
optimise_k_means(final_df[['PCA1','PCA2']],10)


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=3)
kmeans.fit(final_df[['PCA1','PCA2']])
print(final_df)

plt.scatter(x=final_df['PCA1'],y=final_df['PCA2'],c=final_df['kmeans'])


