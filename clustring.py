
import numpy as np
import pandas as pd
import time
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data_file = 'dataset.csv'


def read_data():
    print 'read data...'
    df = pd.read_csv(data_file)
    df = df[['FullDescription']]
    df = pd.DataFrame([doc.lower().translate(None, string.punctuation) for doc in df.FullDescription])

    data_list = []
    for v in df.values:
        data_list.append(str(v))
    return data_list


def extract_features(data):
    print 'fetching features...'
    vectorizer = CountVectorizer(max_features=200)

    features_mat = vectorizer.fit_transform(data)
    features_name =  vectorizer.get_feature_names()

    return features_mat, features_name


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

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :num_of_reduced_features]

    return np.dot(evecs.T, input_mat.T).T, evals, evecs


if __name__ == '__main__':
    data = read_data()
    # print data

    features_mat, features_name = extract_features(data)
    print features_name
    print 'amount of features: ', len(features_name)

    # vars_ = pca(features_mat.toarray())
    # print vars_

    # vars_ = np.var(features_mat.toarray())
    # print vars_

    downsampled_data, evals, evecs = manual_pca(input_mat=features_mat, num_of_reduced_features=2)
