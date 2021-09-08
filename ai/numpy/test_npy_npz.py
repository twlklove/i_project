import numpy as np

def test_0(file, data):
    np.save(file, data)
    data =  np.load(file+'.npy')
    print('data is : ', data)

def test_1(file, data):
    data1 = np.arange(5)
    np.savez(file, a=data, b=data1)
    data_tmp = np.load(file+'.npz')
    print("data_tmp is ", type(data_tmp), list(data_tmp))
    print('data_tmp[a] is : ', data_tmp['a'])
    print('data_tmp[b] is : ', data_tmp['b'])

file='data'
data = np.arange(10)
print(data)

#test_0(file, data)
test_1(file, data)

