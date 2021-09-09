import os
import gzip
import numpy as np
from log import *
from path import *

def load_data(base_path):
    dirname='cache'
    files = [
        'train_label_idx1_ubyte.gz', 'train_data_idx3_ubyte.gz', 
        'test_label_idx1_ubyte.gz', 'test_data_idx3_ubyte.gz'
    ]
    
    paths = []
    for fname in files:
      paths.append(os.path.join(base_path, fname))
    
    with gzip.open(paths[0], 'rb') as lbpath:
      y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[1], 'rb') as imgpath:
      x_train = np.frombuffer(
          imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    
    with gzip.open(paths[2], 'rb') as lbpath:
      y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[3], 'rb') as imgpath:
      x_test = np.frombuffer(
          imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    base_path = os.path.join(base_path,'dst')
    (x_train, y_train), (x_test, y_test) = load_data(base_path)
    log_i(len(y_train), y_train)
    log_i(len(y_test), y_test)
