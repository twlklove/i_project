import numpy as np
import os
import random
import gzip
import base64
import struct
import cv2

debug = 1
def log_d(*info):
    if debug:
        print(*info)


def convert_to_binary_bytestr(*value, endian='<i4'):
    #np.frombuffer(np.array([1,2,3], dtype='>i1').tobytes(), dtype=np.uint8) 
    #value = bytes().fromhex('{:08x}'.format(value)) #hex(0x00000803).strip('0x'))
    log_d(*value, endian)
    value = np.array(*value, dtype=endian).tobytes()
    log_d(value)
    return value

def test(path):
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

    magic = 0x00000803 #0x00000801
    num = len(files) #0x00002710
    num_rows = 0x0000001c
    num_cols = 0x0000001c
    data = convert_to_binary_bytestr([magic, num, num_rows, num_cols], endian='>i4')
    with gzip.open('data_idx_ubyte.gz', 'wb') as lb :
        lb.write(data)

    count=0
    with gzip.open('data_idx_ubyte.gz', 'ab') as lb :
        for file in files:
            count += 1 
            dir = file[1]
            file_name = file[2]
            path = os.path.join(dir, file_name)
            content = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            lb.write(content)
            log_d(count, path, ':', content.shape)


def encode_train(data_dir):
    files=[]
    path = os.path.join(data_dir, 'train')
    dirs = os.listdir(path)
    for dir in dirs:
        dir_tmp = os.path.join(path, dir)
        files.append(list(os.walk(path)))
    print(len(files), type(files))

    for file in files:
        print(file)


if __name__ == '__main__':
    base_path = '/mnt/hgfs/i_share/fashion-mnist/'
    #encode_train(base_path)
    test(os.path.join(base_path,'train'))
    exit()
     
