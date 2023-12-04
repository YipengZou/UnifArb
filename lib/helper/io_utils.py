import os
import pickle

def load_pickle_local(path):
    try:
        res = pickle.load(open(path, 'rb'))
    except Exception as e:
        print('Failed to load pickle from ' + path)
        return None
    return res

def save_pickle_local(path, data):
    pickle.dump(data, open(path, 'wb'))
