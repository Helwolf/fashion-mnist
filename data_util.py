import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import zipfile


normalize = {'mean': 72.99545760204082,
             'std': 89.96534428078684}


def get_train_data(data_path='data/fashion_train.csv', test_size=0.1, one_hot=True):

    data = pd.read_csv(data_path)
    label = np.array(data['label'])
    data = data.drop('label', 1)
    data = np.array(data).astype(np.uint8)
    data = data.reshape(-1, 28, 28, 1)

    if one_hot:
        cn = np.max(label)
        label = np.eye(cn).astype(np.int)[label]

    return train_test_split(data, label, test_size=test_size)


def get_test_data(data_path='data/fashion_test.csv'):

    data = pd.read_csv(data_path)
    data = np.array(data).astype(np.uint8)
    data = data.reshape(-1, 28, 28, 1)
    return data


def create_result(result, data_path='data/fashion_resulf.csv'):
    result = pd.DataFrame({'label': result})
    result.to_csv(data_path, index=False)
    zf = zipfile.ZipFile('result.zip', mode='w')
    try:
        zf.write(data_path, 'submission.csv')
    finally:
        print('success!')
        zf.close()


