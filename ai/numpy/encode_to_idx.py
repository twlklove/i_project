import numpy as np
import os
import random
import gzip
import base64
import struct
import cv2
from log import *

def convert_to_binary_bytestr(*value, endian='<i4'):
    #np.frombuffer(np.array([1,2,3], dtype='>i1').tobytes(), dtype=np.uint8) 
    #value = bytes().fromhex('{:08x}'.format(value)) #hex(0x00000803).strip('0x'))
    log_d(*value, endian)
    value = np.array(*value, dtype=endian).tobytes()
    log_d(value)
    return value

def get_files(path):
    dirs=os.listdir(path)
    files=[]
    for dir in dirs:
        for dp, dn, fn in os.walk(os.path.join(path, dir)):
            log_d(dir, fn)
            random.shuffle(fn)
            for fn_tmp in fn :
                files.append([dir, dp, fn_tmp])
    
    #log_d(files)
    random.shuffle(files)
    log_d(files[:10])
    log_d(len(files))
    return files

def encode_data(src_data, file): 
    magic = 0x00000803 #0x00000801
    num = len(src_data) #0x00002710
    num_rows = 28#0x0000001c
    num_cols = 28#0x0000001c
    data = convert_to_binary_bytestr([magic, num, num_rows, num_cols], endian='>i4')
    with gzip.open(file, 'wb') as lb :
        lb.write(data)

    count=0
    with gzip.open(file, 'ab') as lb :
        for data in src_data:
            count += 1 
            dir = data[1]
            file_name = data[2]
            path = os.path.join(dir, file_name)
            #content = cv2.imread(path) 
            content = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            content = cv2.resize(content, (num_rows, num_cols))
            lb.write(content)
            log_d(count, path, ':', content.shape)
 
            #with open(path, 'rb') as f:
            #    content = f.read()
            #    lb.write(content)
            #lb.write(content)
            #log_d(count, path)


def encode_label(src_data, file):
    magic = 0x00000801 #0x00000801
    num = len(src_data) #0x00002710
    data = convert_to_binary_bytestr([magic, num], endian='>i4')
    with gzip.open(file, 'wb') as lb :
        lb.write(data)

    count=0
    with gzip.open(file, 'ab') as lb :
        for data in src_data:
            count += 1 
            label = np.array(data[0], dtype='>u1').tobytes()
            lb.write(label)
            log_d(count, ' : ', label)

def encode(path, data_file, label_file):
    files = get_files(path)
    encode_data(files, data_file)
    encode_label(files, label_file)


if __name__ == '__main__':
    base_path = '/mnt/hgfs/i_share/i_test'
    #base_path = os.path.join(base_path, 'fashion-mnist', 'train')
    path = os.path.join(base_path, 'src', 'train')

    train_data_file = os.path.join(base_path, 'train_data_idx_ubyte.gz')
    train_label_file = os.path.join(base_path, 'train_label_idx_ubyte.gz')

    encode(path, train_data_file, train_label_file)
    exit()
     
