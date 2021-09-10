import numpy as np

a_img = np.arange(5)
a_label = np.arange(6)
b_img = np.arange(7)
b_label = np.arange(8)
np.savez('test', a_img=a_img, a_label=a_label, b_img=b_img, b_label=b_label)

def get_data():
    with np.load('test.npz') as data:
        print(data.files)
        img, label = data['a_img'], data['a_label']
        img_t, label_t = data['b_img'], data['b_label']
        return (img, label), (img_t, label_t)

(a, a1), (b, b1) = get_data()
print(a, a1, b, b1)
