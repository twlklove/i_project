import numpy as np
import gzip
import os
import struct
import cv2
import shutil
from log import *

def decode_idx3_ubyte(file):
    magic = 0
    image_num = 0
    num_rows = 0
    num_cols = 0
    fmt_header = '>4i' #'>iiii'    # 以大端法读取4个 unsinged int32
    offset = 0
    with gzip.open(file, 'rb') as lbpath :
        data = lbpath.read()
        log_d(len(data), data[0:16]) 
        magic, image_num, num_rows, num_cols = struct.unpack_from(fmt_header, data, offset)
        #magic = np.frombuffer(data, np.uint8, count=4, offset=0)
        #image_num = np.frombuffer(data, np.uint8, count=4, offset=4)
        #num_rows = np.frombuffer(data, np.uint8, count=4, offset=8)
        #num_cols = np.frombuffer(data, np.uint8, count=4, offset=12)
        #with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as lbpath :
        #    y_train = np.frombuffer(lbpath.read(), np.unit8, offset=16).reshape(28, 28)
 
    log_d('magic:{}, image_num:{}, num_rows:{}, num_cols:{}'.format(magic, image_num, num_rows, num_cols))
    
    offset += struct.calcsize(fmt_header)
    
    fmt_image = '>' + str(num_rows * num_cols) + 'B'
    images = np.empty((image_num, num_rows, num_cols))
    for i in range(image_num):
        im = struct.unpack_from(fmt_image, data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    
    return images

def decode_idx1_ubyte(file):
    magic = 0
    label_num = 0
    fmt_header = '>2i' 
    offset = 0
    with gzip.open(file, 'rb') as lbpath :
        data = lbpath.read()
        log_d(len(data), data[0:8]) 
        magic, label_num = struct.unpack_from(fmt_header, data, offset)
    
    log_d('magic:{}, label_num:{}'.format(magic, label_num))
    
    offset += struct.calcsize(fmt_header)
    
    fmt_label = '>1B'
    labels = []
    for i in range(label_num):
        label = struct.unpack_from(fmt_label, data, offset)
        labels.append(label[0])
        offset += struct.calcsize(fmt_label)
    
    return labels

def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        log_d(folder)
    else:
        if not os.path.isdir(folder):
            os.makedirs(folder)


def export_img(exp_dir, img_ubyte, lable_ubyte):
    check_folder(exp_dir)
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(lable_ubyte)

    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(exp_dir, str(labels[i]))
        check_folder(img_dir)
        img_file = os.path.join(img_dir, str(i)+'.png') # '.jfif')
        imarr = images[i]
        cv2.imwrite(img_file, imarr)
        #with open(img_file, 'wb') as f:
        #    f.write(imarr)


def parser_mnist_data(path):
    path='/mnt/hgfs/i_share/fashion-mnist/'
    train_dir = os.path.join(path, 'train')
    train_img_ubyte = os.path.join(path, 'train-images-idx3-ubyte.gz')
    train_label_ubyte = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    export_img(train_dir, train_img_ubyte, train_label_ubyte)

    test_dir = os.path.join(path, 'test')
    test_img_ubyte = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    test_label_ubyte = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    export_img(test_dir, test_img_ubyte, test_label_ubyte)

def decode(path, data_file, label_file, to_dir): 
    img_ubyte = os.path.join(path, data_file)
    label_ubyte = os.path.join(path, label_file)
    export_img(to_dir, img_ubyte, label_ubyte)

if __name__ == '__main__':    
    #parser_mnist_data(path)
    path='/mnt/hgfs/i_share/i_test'
    data_file = 'train_data_idx_ubyte.gz'
    label_file = 'train_label_idx_ubyte.gz'
    to_dir = os.path.join(path, 'train')
    shutil.rmtree(to_dir) #os.removedirs/os.rmdir
    decode(path, data_file, label_file, to_dir)

