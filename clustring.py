
import numpy as np
import pandas as pd
import time
import string
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import collections, re
from collections import Counter
import math
import kmeans
import nltk
# from textblob import TextBlob as tb

data_file = 'dataset.csv'
# nltk.download()

min_features = 100
features_amount = 100

def read_data():
    print 'read data...'
    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:4]

    tokens = []
    for doc in df.FullDescription:
        tokens.append(get_tokens(doc))
    # print tokens

    filtered_tokens = []
    for token in tokens:
        filtered_token = []
        for w in token:
            if not w in ENGLISH_STOP_WORDS:
                filtered_token.append(w)
        filtered_tokens.append(filtered_token)
    # filtered_tokens = [w for token in tokens
    #                             for w in token
    #                                 if not w in ENGLISH_STOP_WORDS]
    return filtered_tokens

def get_tokens(doc):
    return nltk.word_tokenize(doc.lower().translate(None, string.punctuation))

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)


def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)


def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))


def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


def extract_features(data):
    print 'fetching features...'
    vectorizer = CountVectorizer(max_features=features_amount)

    features_mat = vectorizer.fit_transform(data)
    features_name = vectorizer.get_feature_names()

    return features_mat, features_name


def extract_features_manual(data):
    data_split = []
    # data_split = ' '.join([str(w) for doc in data
    #                                 for w in doc])
    for doc in data:
        data_split.append(' '.join(str(w) for w in doc))

    # print data_split
    bag_of_words =  [collections.Counter(re.findall(r'\w+', txt)).most_common(features_amount) for txt in data_split]

    features = []
    for txt in bag_of_words:
        if len(txt) >= min_features:
            doc_features = []
            for v in txt:
                doc_features.append(v[1])
            features.append(np.array(doc_features))
    return np.array(features)


def pca(data):
    pca = PCA(n_components=10)
    pca.fit(data)
    vars_ = pca.explained_variance_

    plt.plot(range(0, len(vars_)), vars_)
    plt.ylabel('n_components')
    plt.show()

    return vars_

def manual_pca(input_mat, num_of_reduced_features=2):
    cov = np.cov(input_mat)
    # print cov

    evals, evecs = np.linalg.eigh(cov)
    # print evals
    # print evecs

    # should be equal
    # print np.sum(evals)
    # print np.sum(np.diag(cov))

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :num_of_reduced_features]

    return np.dot(evecs.T, input_mat).T, evals, evecs


def kmeans_clustring(vectors, k=3):
    k = 3
    km = kmeans.KMeans(k, vectors)
    km.main_loop()

def fkmeans():
    pass


if __name__ == '__main__':
    data = read_data()
    # print data

    # features, features_name = extract_features(data)
    # print features_name
    # print 'amount of features: ', len(features_name)

    features = extract_features_manual(data)
    # print features

    # vars_ = pca(features_mat.toarray())
    # print vars_

    # vars_ = np.var(features_mat.toarray())
    # print vars_

    downsampled_data, evals, evecs = manual_pca(input_mat=features, num_of_reduced_features=3)
    # print downsampled_data

    kmeans_clustring(downsampled_data)