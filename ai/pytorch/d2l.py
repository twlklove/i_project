#from __future__ import print_function
import torch
from torch.distributions import multinomial
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import os
import inspect
import torchvision
from torchvision import transforms
from torch.utils import data
import time
import hashlib
import tarfile
import zipfile
import requests
    
def test_0():
    print(inspect.currentframe())
    x = torch.empty(5, 3)
    print(x)
    
    x = torch.tensor([5.5, 3])
    print(x)
    
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)
    
    x = x.new_ones(5, 3, dtype=torch.double)
    # new_* methods take in sizes
    print(x)
    
    x = torch.randn_like(x, dtype=torch.float) # override dtype!
    print(x) # result has the same size
    print(x.size())
        
    x = torch.arange(12)
    print(x)
    print(x.shape, x.numel())
    print(x.reshape(3, 4))
    print(x.reshape(-1, 4))
    print(x.reshape(3, -1))
    print(torch.randn(3, 4))  #jz:0, bzc:1, gaosi
    print(torch.rand(5, 3))    

def test_1():
    print(inspect.currentframe())
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x+y)
    print(x-y)
    print(x*y)
    print(x/y)
    print(x**y)
    print(torch.exp(x))

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1 ]])
    print(torch.cat((x, y), dim=0))
    print(torch.cat((x,y), dim=1))
    print(x.sum())
    print(x==y) 

    # index ans slice: 0 is first, and -1 is last
    print(x[-1], x[1:3])
    x[1, 2] = 9
    print(x)

    x[0:2, :] = 12
    print(x)
 
    #broadcast
    a = torch.arange(3).reshape((3,1))
    b = torch.arange(2).reshape((1,2))
    print(a+b)

    #id(x) is addr, use x[:] = x + y or x += y in order to save memory
    before = id(y)
    y = y + x
    print(id(y) == before)

    before = id(x)
    x[:] = x + y
    print(before == id(x)) 

    x += y
    print(before == id(x)) 
   
    #tensor to numpy
    a = x.numpy()
    b = torch.tensor(a)
    print(type(a), type(b))

    c = torch.tensor([3.5])
    print(c, c.item(), float(c), int(c))

#data preprocess using pandas
def test_2():
    print(inspect.currentframe()) 
    os.makedirs(os.path.join('.', 'test_data'), exist_ok=True)
    data_file = os.path.join('.', 'test_data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178000\n')
        f.write('NA,NA,140000\n')

    # !pip3 install pandas 
    data = pd.read_csv(data_file)
    print(data)

    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    inputs = inputs.fillna(inputs.mean(numeric_only=True))
    print(inputs)
    
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    # Dataframe to tensor
    x = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(outputs.to_numpy(dtype=float))
    print(x)
    print(y)

### linear
def test_3():
    print(inspect.currentframe()) 
    #scalar
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print(x+y, x*y, x/y, x**y)
    
    #vector
    x = torch.arange(4)
    print(x)
    print(len(x), x.shape)

    #matrix
    A = torch.arange(20).reshape(5, 4)
    print(A)
    print(A.T)                                     #transpose
    
    B = torch.tensor([[1,2,3], [2,0,4], [3,4,5]])  #symmetric matrix
    print(B)
    print(B.T)

    #tensor
    x = torch.arange(24).reshape(2, 3, 4)
    print(x)

    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()
    print(A)
    print(A+B)
    print(A*B)

    a = 2
    print(a + A)
    print(a * A)
    print(A.shape)

    ###
    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print(A)
    print("sum is : %d" %(A.sum()))
    print(A.sum(axis=0))
    print(A.sum(axis=1))
    sum_A = A.sum(axis=1, keepdims=True)
    print(sum_A)
    print(A/sum_A)
    print(A.cumsum(axis=0)) #sum on rows

    print(A.mean(), A.sum()/A.numel())
    print(A.mean(axis=0), A.sum(axis=0)/A.shape[0])
    
    #### dot
    x = torch.arange(4, dtype=torch.float32)
    y = torch.ones(4, dtype=torch.float32)
    print(x)
    print(y)
    print(torch.sum(x*y))
    print(torch.dot(x, y))

    #### matrix multipule
    A = torch.ones(2, 3)
    B = torch.ones(3, 4)
    C = torch.mm(A, B)
    print(A.shape, B.shape, C.shape)
    print(A)
    print(B)
    print(C)

    ## Norm
    u = torch.tensor([3.0, -4.0])
    print(torch.norm(u))     # L2 Norm
    print(torch.abs(u).sum())  #L1 Norm
    print(torch.norm(torch.ones((2,3)))) # Frobenius norm for matrix

### differential calculus and integral calculus
def test_4():  
    print(inspect.currentframe())  
    def f(x):
        return 3 * x**2 - 4*x
    
    def numerical_lim(f, x, h) :
        return (f(x+h) - f(x)) / h
    h = 0.1
    for i in range(5) :
        print(f'h={h:0.5f}, numerial limit={numerical_lim(f, 1, h):.5f}')
        h *= 0.1
        
    x = np.arange(0, 3, 0.1)
    plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


def test_5():
    print(inspect.currentframe())  

    x = torch.arange(4.0, requires_grad=True)
    y = 2 * torch.dot(x, x)
    print(x) 
    print(y)

    y.backward()
    print(x.grad)
    print(x.grad == 4 *x)

    x.grad.zero_()  #clear grad before
    y = x.sum()
    y.backward()
    print(x.grad)

    ###############
    x.grad.zero_()
    y = x * x
    u = y.detach()  # detach 
    z = u * x
    z.sum().backward()
    print(x.grad == u)

    x.grad.zero_()
    y.sum().backward()
    print(x.grad == 2 * x)

## sampling and distribution, multinomial distribution
def test_6():
    print(inspect.currentframe())  

    fair_probs = torch.ones([6])
    print(multinomial.Multinomial(10, fair_probs).sample())

    counts = multinomial.Multinomial(1000, fair_probs).sample()
    estimates = counts / 1000
    print(estimates)

    counts = multinomial.Multinomial(10, fair_probs).sample((500,))
    print(counts)
    cum_counts = counts.cumsum(dim=0)
    print(cum_counts.shape)
    print(cum_counts)
    sum_on_dim1 = cum_counts.sum(dim=1, keepdims=True)
    print(sum_on_dim1.shape)
    print(sum_on_dim1[0:10, :])
    estimates = cum_counts / sum_on_dim1 
    print(estimates)

    fig = plt.figure(figsize=(6, 4.5)) 
    fig.set_facecolor('lightgray')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('lightyellow')
    ax.set_xlabel('Groups of experiments')
    ax.set_ylabel('Estimated probability')
    ax.set_ylim(0, 0.4)
    ax.axhline(y=0.167, color='black', linestyle='--')  #axvline(...)
    #plt.text(100, 0.167, '0.167')
    plt.plot(200, 0.167, 'ro')
    plt.annotate("this is poit(100, 0.167)", xy=(200, 0.167), xytext=(150, 0.25), color='green', arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3', color='red'))
    
    for i in range(6):
        plt.plot(estimates[:, i].numpy(), label=("P(die="+str(i+1)+")"))
    plt.legend()
    plt.grid(ls=':', color='gray', alpha=0.5)
    plt.text(200, 0.3, "hello, twlk", fontsize=10, color='gray', alpha=0.5)

    plt.show()

#######################################
def test_timer():
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])

    timer=Timer()
    c = a + b
    print(f'{timer.stop(): .5f} sec')
    
def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel) 
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    if legend:
        axes.legend(legend)
    axes.grid()

def plot(x, y=None, xlabel=None, ylabel=None, legend=None, xlim=None, 
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None) :
    if legend is None:
        legend=[]

    set_figsize(figsize)
    axes = axes if axes else plt.gca()        
    
    def has_one_axis(x):
        return (hasattr(x, "ndim") and x.ndim == 1 or isinstance(x, list) and not hasattr(x[0], "__len__"))
    
    if has_one_axis(x) :
        x = [x]

    if x is None:
        x, y = [[]] * len(x), x
    elif has_one_axis(y) :
        y = [y]

    if len(x) != len(y):
        x = x * len(y)
    
    axes.cla()
    for x, y, fmt in zip(x, y, fmts) :
        if len(x) :
            axes.plot(x, y, fmt)
        else :
            axes.plot(y, fmt)

    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()
    
class Timer:
    def __init__(self) :
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

## define sgd
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def load_array(data_arrays, batch_size, is_train=True):
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='./test_data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./test_data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())
    #accuracy(y_hat, y) / len(y)

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx) :
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
        if isinstance(net, torch.nn.Module):
            net.eval()  # set net into evaluate mode
        metric = Accumulator(2) #
        with torch.no_grad():
            for x, y in data_iter:
                metric.add(accuracy(net(x), y), y.numel())
        return metric[0] / metric[1]

class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, 
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)

        if nrows * ncols == 1:
            self.axes = [self.axes, ]

        self.config_axes = lambda:set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.x, self.y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.x:
            self.x = [[] for _ in range(n)]
        if not self.y:
            self.y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.x[i].append(a)
                self.y[i].append(b)
                
        self.axes[0].cla()
        for x, y, fmt in zip(self.x, self.y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display(self.fig, clear=True)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    fig = plt.figure(figsize=figsize)
    axes = fig.subplots(num_rows, num_cols)
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])
    plt.show()
    return axes;

def train_epoch_ch3(net, trainer_iter, loss, updater) :
        if isinstance(net, torch.nn.Module):
            net.train()

        metric = Accumulator(3)
        for x, y in trainer_iter:
            y_hat = net(x)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(x.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test_acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
def predict_ch3(net, test_iter, get_labels_func=None, n=6):
    if get_labels_func is None:
        get_labels_func = get_fashion_mnist_labels
        
    for x, y in test_iter:
        break
        
    trues = get_labels_func(y)
    preds = get_labels_func(net(x).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(x[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        
### get data set
def synthetic_data(w, b, num_examples):
    """y = xw + b + c"""
    x = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(x, w) + b      
    y += torch.normal(0, 0.001, y.shape)
    return x, y.reshape((-1, 1))

def test_load_data_fashion_mnist():
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    for x, y in train_iter:
        print(x.shape, x.dtype, y.shape, y.dtype)
        break
        
def test_fishion_mnist_data_set():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./test_data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./test_data", train=False, transform=trans, download=True)
    print(len(mnist_train), len(mnist_test), mnist_train[0][0].shape)

    x, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(x.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                 num_workers=get_dataloader_workers())
    timer = Timer()
    for x, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')
    
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('.', 'test_data')):
    assert name in DATA_HUB, f"{name} not exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)

    return fname

def download_extract(name, folder=None):
    fnmae = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False

    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    for name in DATA_HUB:
        download(name)

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda"{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
    test_0()
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_timer()
    test_load_data_fashion_mnist()
    test_fishion_mnist_data_set()

