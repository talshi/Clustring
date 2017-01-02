
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


def read_data():
    print 'read data...'
    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:2]
    tokens = get_tokens(df)
    # print tokens
    filtered_tokens = [token for token in tokens
                                for w in token
                                    if not w in ENGLISH_STOP_WORDS]
    # data_list = []
    # for v in df.values:
    #     data_list.append(str(v))
    return filtered_tokens

def get_tokens(df):
    return [nltk.word_tokenize(doc.lower().translate(None, string.punctuation))
                       for doc in df.FullDescription]

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
    vectorizer = CountVectorizer(max_features=200)

    features_mat = vectorizer.fit_transform(data)
    features_name =  vectorizer.get_feature_names()

    return features_mat, features_name


def extract_features_manual(data):
    data_split = ' '.join([str(w) for doc in data
                                    for w in doc])
    # print data_split
    bag_of_words =  [collections.Counter(re.findall(r'\w+', txt)).most_common(10) for txt in data_split]
    return bag_of_words


def pca(data):
    pca = PCA(n_components=10)
    pca.fit(data)
    vars_ = pca.explained_variance_

    plt.plot(range(0, len(vars_)), vars_)
    plt.ylabel('n_components')
    plt.show()

    return vars_

def manual_pca(input_mat, num_of_reduced_features=2):
    cov = np.cov(input_mat.toarray())
    print cov

    evals, evecs = np.linalg.eigh(cov)
    print evals
    print evecs

    # should be equal
    print np.sum(evals)
    print evals/np.sum(np.diag(cov))

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :num_of_reduced_features]

    return np.dot(evecs.T, input_mat.T).T, evals, evecs


def kmeans(vectors, k=3):
    k = 3
    km = kmeans(k, vectors)
    km.main_loop()

def fkmeans():
    pass


if __name__ == '__main__':
    data = read_data()
    # print data

    # features_mat, features_name = extract_features(data)
    # print features_name
    # print 'amount of features: ', len(features_name)

    features = extract_features_manual(data)
    # print features

    # vars_ = pca(features_mat.toarray())
    # print vars_

    # vars_ = np.var(features_mat.toarray())
    # print vars_

    downsampled_data, evals, evecs = manual_pca(input_mat=features, num_of_reduced_features=2)
