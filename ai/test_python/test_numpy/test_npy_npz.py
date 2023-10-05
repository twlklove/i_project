import numpy as np

def test_0(file):
    data = np.arange(10)
    print(data)
    np.save(file, data)
    data =  np.load(file+'.npy')
    print('data is : ', data)

def test_1(file):
    data = np.arange(10)
    data1 = np.arange(5)
    np.savez(file, a=data, b=data1)
    data_tmp = np.load(file+'.npz')
    print("data_tmp is ", type(data_tmp), list(data_tmp))
    print('data_tmp[a] is : ', data_tmp['a'])
    print('data_tmp[b] is : ', data_tmp['b'])

if __name__ == '__main__':
    file='train_test_data'
    
    #test_0(file)
    #test_1(file)

