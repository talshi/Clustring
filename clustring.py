
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
import fcmeans as fcm
import nltk
# from textblob import TextBlob as tb

data_file = 'dataset.csv'
features_amount = 20
amount_of_data = 10


def read_data():
    df = pd.read_csv(data_file)
    df = df[['FullDescription']][:amount_of_data]
    tokens = get_tokens(df)
    tokens = remove_stop_words(tokens)
    return tokens


def get_tokens(df):
    return [nltk.word_tokenize(doc.lower().translate(None, string.punctuation))
                       for doc in df.FullDescription]


def remove_stop_words(tokens):
    for doc in tokens:
        for token in doc:
            if token in ENGLISH_STOP_WORDS:
                doc.remove(token)
    return tokens


# def tf(word, blob):
#     return blob.words.count(word) / len(blob.words)
#
#
# def n_containing(word, bloblist):
#     return sum(1 for blob in bloblist if word in blob.words)
#
#
# def idf(word, bloblist):
#     return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
#
# def tfidf(word, blob, bloblist):
#     return tf(word, blob) * idf(word, bloblist)


def extract_features(data):
    vectorizer = CountVectorizer(max_features=features_amount)

    features_mat = []
    for doc in data:
        features_per_doc = vectorizer.fit_transform(np.array(doc))
        features_mat.append(features_per_doc)
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
    bag_of_words = [doc.most_common(features_amount) for doc in bag_of_words]
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
    evals, evecs = np.linalg.eigh(cov)

    # should be equal
    assert np.sum(evals) != np.sum(np.diag(cov))
    # print np.sum(evals)
    # print evals/np.sum(np.diag(cov))

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :num_of_reduced_features]

    return np.array(np.dot(evecs.T, input_mat).T)


def kmeans(vectors, metric='euclidean', k=29):
    km = kms.KMeans(k, vectors, metric)
    km.main_loop()
    return km.get_clusers(), km.get_centers()


def kmeans_with_pca(vectors, metric='euclidean', k=29):
    print '####################################################################'
    print 'performing pca...'
    downsampled_data = manual_pca(input_mat=vectors, num_of_reduced_features=2)
    print '####################################################################\n'

    km = kms.KMeans(k, downsampled_data, metric)
    km.main_loop()
    return km.get_clusers(), km.get_centers()


def fcmeans(vectors, metric='euclidean', k=29):
    fcm.FCmeans()
    fcm.fuzzy_cmeans()
    return fcm.get_centers(), fcm.get_U()


def fcmeans_with_pca(vectors, metric='euclidean', k=29):
    print '####################################################################'
    print 'performing pca...'
    downsampled_data = manual_pca(input_mat=vectors, num_of_reduced_features=2)
    print '####################################################################\n'

    fcm.FCmeans()
    fcm.fuzzy_cmeans()
    return fcm.get_centers(), fcm.get_U()


if __name__ == '__main__':
    print '####################################################################'
    print 'read data...'
    data = read_data()
    # print data
    print '####################################################################\n'

    # print '####################################################################'
    # print 'fetching features...'
    # features, features_name = extract_features(data)
    # print features_name
    # print 'amount of features: ', len(features_name)
    # print '####################################################################\n'

    print '####################################################################'
    print 'fetching features...'
    features, features_names = extract_features_manual(data)
    print features_names
    print features
    print '####################################################################\n'

    # vars_ = pca(features)
    # print vars_

    # print '####################################################################'
    # print 'performing pca...'
    # downsampled_data = manual_pca(input_mat=features, num_of_reduced_features=2)
    # print '####################################################################\n'

    print '####################################################################'
    print 'performing kmeans...'
    # clusters, centers = kmeans(features, metric='euclidean', k=10)
    clusters, centers = kmeans_with_pca(features, metric='euclidean', k=10)
    print len(clusters), 'clusters are:'
    for cluster in clusters:
        print cluster
    print '\n', len(centers), 'centers are:'
    for center in centers:
        print center
    print '####################################################################\n'

    print '####################################################################'
    print 'performing fuzzy c means...'
    # clusters, centers = fcmeans(features, metric='euclidean', k=10)
    clusters, centers = kmeans_with_pca(features, metric='euclidean', k=10)
    print len(clusters), 'clusters are:'
    for cluster in clusters:
        print cluster
    print '\n', len(centers), 'centers are:'
    for center in centers:
        print center
    print '####################################################################\n'