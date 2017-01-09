
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
import kmeans as kms
import nltk
# from textblob import TextBlob as tb

data_file = 'dataset.csv'


def read_data():
    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:10]
    tokens = get_tokens(df)

    # remove stop words
    for doc in tokens:
        for token in doc:
            if token in ENGLISH_STOP_WORDS:
                doc.remove(token)

    # data_list = []
    # for doc in filtered_tokens:
    #     for token in doc:
    #         data_list.append(str(token))
    return tokens


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
    vectorizer = CountVectorizer(max_features=200)

    features_mat = vectorizer.fit_transform(data)
    features_name =  vectorizer.get_feature_names()

    return features_mat, features_name


def extract_features_manual(data):
    data_split = []
    for doc in data:
        doc_str = ' '.join([str(w) for w in doc])
        data_split.append(doc_str)
    # data_split = ' '.join([str(w) for doc in data
    #                                 for w in doc])
    # print data_split
    # bag_of_words =  [txt for txt in data_split]
    bag_of_words = [collections.Counter(re.findall(r'\w+', doc)) for doc in data_split]
    bag_of_words = [doc.most_common(20) for doc in bag_of_words]
    features_names = bag_of_words

    features = []
    for doc in bag_of_words:
        doc_features = []
        for t in doc:
            doc_features.append(t[1])
        features.append(doc_features)

    # bag_of_words = [w[1] for w in bag_of_words]
    return np.array(features), features_names


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
    # print evals/np.sum(np.diag(cov))

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :num_of_reduced_features]

    return np.array(np.dot(evecs.T, input_mat).T), evals, evecs


def kmeans(vectors, metric='euclidean', k=29):
    km = kms.KMeans(k, vectors, metric)
    km.main_loop()
    return km.get_clusers(), km.get_centers()


def fkmeans():
    pass


if __name__ == '__main__':
    print '####################################################################'
    print 'read data...'
    data = read_data()
    # print data
    print '####################################################################\n'

    # features, features_name = extract_features(data)
    # print features_name
    # print 'amount of features: ', len(features_name)

    print '####################################################################'
    print 'fetching features...'
    features, features_names = extract_features_manual(data)
    print features_names
    print features
    print '####################################################################\n'

    # vars_ = pca(features)
    # print vars_

    print '####################################################################'
    print 'performing pca...'
    downsampled_data, evals, evecs = manual_pca(input_mat=features, num_of_reduced_features=2)
    print '####################################################################\n'

    print '####################################################################'
    print 'performing kmeans...'
    clusters, centers = kmeans(downsampled_data, metric='euclidean', k=10)
    print len(clusters), 'clusters are:'
    for cluster in clusters:
        print cluster
    print '\n', len(centers), 'centers are:'
    for center in centers:
        print center
    print '####################################################################\n'