
import numpy as np
import pandas as pd
import time
import string

data_file = 'dataset.csv'

def read_data():
    print 'read data...'
    df = pd.read_csv(data_file)
    df = df[['FullDescription']]
    df = pd.DataFrame([doc.lower().translate(None, string.punctuation) for doc in df.FullDescription])
    return df

def fetch_features():
    pass

if __name__ == '__main__':
    data = read_data()
    print data