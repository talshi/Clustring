
import numpy as np
import pandas as pd
import time
import string
from sklearn.feature_extraction.text import CountVectorizer

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
    vectorizer = CountVectorizer(max_features=2000)
    vectorizer.fit_transform(data)
    # analyze = vectorizer.build_analyzer()

    return vectorizer.get_feature_names()

if __name__ == '__main__':
    data = read_data()
    # print data

    features = extract_features(data)
    print features
    print 'amount of features: ', len(features)