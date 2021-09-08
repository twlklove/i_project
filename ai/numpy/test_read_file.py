
import os
def test():
    num = 0
    with open('/mnt/hgfs/i_share/apple/0.jfif', 'rb') as f:
        for line in f.readlines():
            num += 1
            print("line : ", num, line.strip())  #strip \n

def test_1(file):
    data=0
    with open(file, 'rb') as f:
        data = f.read()
        print(data)

    path = os.path.dirname(file)
    path = path + os.path.sep + 'xx.jpeg'
    with open(path, 'wb') as f:
        f.write(data[0:3999])

def test_2():
    os.listdir('/mnt/hgfs/i_share/')
    for i, j, k in os.walk('/mnt/hgfs/i_share/'):
        print(i, j, k)

if __name__ == '__main__':
    test_1('/mnt/hgfs/i_share/apple/0.jfif')
    test_1('/mnt/hgfs/i_share/t10k-images-idx3-ubyte')
    test_2()

