import numpy as np
import math
import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import d2l
import pandas as pd
from torch.nn import functional as F
                      
def test_linear_regression():
    import random 
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    
    ## read data set using batch_size
    def data_iter(batch_size, features, labels):
        num_examples = len(features)
        indices = list(range(num_examples))
        random.shuffle(indices)

        for i in range(0, num_examples, batch_size) :
            batch_indices = torch.tensor(indices[i: min(i+batch_size, num_examples)])
            yield features[batch_indices], labels[batch_indices]
    
    batch_size = 10
    for x, y in data_iter(batch_size, features, labels) :
        print(x, '\n', y)
        break
 
    ## define net
    def linreg(x, w, b) :
        return torch.matmul(x, w) + b

    ## define loss
    def squared_loss(y_hat, y):
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

    ##init parameters 
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) 
    b = torch.zeros(1, requires_grad=True)

    ## tranning
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss
        
    for epoch in range(num_epochs) :
        for x, y in data_iter(batch_size, features, labels):
            l = loss(net(x, w, b), y)
            print('hello')
            print(w.grad)
            l.sum().backward()           # calculate grad
            print(w.grad)
            sgd([w, b], lr, batch_size)  #update parameters

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w error is xx: {true_w - w.reshape(true_w.shape)}')
    print(f'b error is xx: {true_b - b}')

################
def test_linear_regression_use_torch_api(): 
    # create data set
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    
    # get data
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    #print(next(iter(data_iter)))
   
    # define net
    net = nn.Sequential(nn.Linear(2, 1)) # input shape: 2, output shape: 1, Linear: full connected
    
    #init parameters
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # define loss
    loss = nn.MSELoss() #MESLoss is mean squared error, as is L2 normal 
    
    # define optimization algorithm
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    
    #training
    num_epochs = 3
    for epoch in range(num_epochs):
        for x, y in data_iter:
            l = loss(net(x), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch+1}, loss{l:f}')
   
    w = net[0].weight.data
    b = net[0].weight.data
    print('w error is : ', true_w - w.reshape(true_w.shape))
    print('b error is : ', true_b - b)

def test_softmax():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_inputs = 784
    num_outputs = 10
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    def softmax(x):
        x_exp = torch.exp(x)
        partition = x_exp.sum(1, keepdim=True)
        return x_exp / partition

    def net(x):
        y_hat = torch.matmul(x.reshape((-1, w.shape[0])), w) + b
        return softmax(y_hat)
    
    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])
    
    lr = 0.1
    def updater(batch_size):
        return d2l.sgd([w, b], lr, batch_size)

    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    d2l.predict_ch3(net, test_iter, d2l.get_fashion_mnist_labels)
    
def test_softmax_use_torch_api():
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weight)

    loss = nn.CrossEntropyLoss(reduction='none')

    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.predict_ch3(net, test_iter, d2l.get_fashion_mnist_labels)

def test_multiplayer_perceptron():
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    w1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [w1, b1, w2, b2]

    def relu(x):
        a = torch.zeros_like(x)
        return torch.max(x, a)

    def net(x):
        x = x.reshape((-1, num_inputs))
        h = relu(x@w1 + b1)
        return (h@w2 + b2)

    loss = nn.CrossEntropyLoss(reduction='none')

    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

    d2l.predict_ch3(net, test_iter)

def test_multiplayer_perceptron_use_torch_api():
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            
    batch_size, lr, num_epochs = 256, 0.1, 10       
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)        
    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.predict_ch3(net, test_iter)

def test_kaggle_house():
    d2l.DATA_HUB['kaggle_house_train'] = (d2l.DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
    d2l.DATA_HUB['kaggle_house_test'] = (d2l.DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

    #get data
    train_data = pd.read_csv(d2l.download('kaggle_house_train'))
    test_data = pd.read_csv(d2l.download('kaggle_house_test'))
    print(train_data.shape, test_data.shape)
    print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    #preprocess data
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    #print(all_features.dtypes)
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    #print(all_features.shape)
    
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

    loss = nn.MSELoss()
    in_features = train_features.shape[1]
    def get_net():
        net = nn.Sequential(nn.Linear(in_features, 1))
        return net
    def log_rmse(net, features, labels):
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()

    def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
        train_ls, test_ls = [], []
        train_iter = d2l.load_array((train_features, train_labels), batch_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            for x, y in train_iter:
                optimizer.zero_grad()
                l = loss(net(x), y)
                l.backward()
                optimizer.step()

            train_ls.append(log_rmse(net, train_features, train_labels))
            if test_labels is not None:
                test_ls.append(log_rmse(net, test_features, test_labels))
        return train_ls, test_ls

    def get_k_fold_data(k, i, x, y):
        assert k > 1
        fold_size = x.shape[0] // k
        x_train, y_train = None, None
        for j in range(k):
            idx = slice(j * fold_size, (j+1)*fold_size)
            x_part, y_part = x[idx, :], y[idx]
            if j == 1:
                x_valid, y_valid = x_part, y_part
            elif x_train is None:
                x_train, y_train = x_part, y_part
            else :
                x_train = torch.cat([x_train, x_part], 0)
                y_train = torch.cat([y_train, y_part], 0)

        return x_train, y_train, x_valid, y_valid

    def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        train_l_sum ,valid_l_sum = 0, 0
        for i in range(k):
            data = get_k_fold_data(k, i, x_train, y_train)
            net = get_net()
            train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]
            if i == 0:
                d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls], xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                         legend=['train', 'valid'], yscale='log')
            print(f'fold{i+1}, train log rmse {float(train_ls[-1]):f},'
                  f'valid log rmse {float(valid_ls[-1]):f}')
        return train_l_sum / k, valid_l_sum /k

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-fold valid: mean train log rmse: {float(train_l):f},' f'mean valid log rmse: {float(valid_l):f}')

    def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
        net = get_net()
        train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
        d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
        print(f'train log rmse:{float(train_ls[-1]):f}')

        preds = net(test_features).detach().numpy()
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)

    train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
    
def test_calculate():
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8,1))
    x = torch.rand(size=(2,4))
    net(x)
    print(net[2].state_dict())  # look up parameters
    print(type(net[2].bias))
    print(net[2].bias.data)
    print(net[2].weight.grad)
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    print(net.state_dict()['2.bias'].data)
    #print(net[0][1][0].bias.data) 
    
    #init parameters
    def init_normal(m):
        nn.init.normal(m.weight, mean=0, std=0.01)
        nn.init.zerors_(m.bias)
        
    def init_xavier(m):
        nn.init.xavier_uniform_(m.weight)      
    net[0].apply(init_xavier)

    #share parameters
    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4,8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8,1))
    net(x)
    print(net[2].weight.data[0] == net[4].weight.data[0])

    #define yourself layer
    class MyLinear(nn.Module):
        def __init__(self, in_units, units):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(in_units, units))
            self.bias = nn.Parameter(torch.randn(units,))
        def forward(self, x):
            linear = tourch.matmul(x, self.weight.data) + self.bias.data
            return F.relu(linear)

    linear = MyLinear(5, 3)
    print(linear.weight)
    
    # save and load net
    torch.save(x, 'x-file')
    x2 = torch.load('x-file')
    y = torch.zeros(4)
    torch.save([x, y], 'x-files')
    x2, y2 = torch.load('x-files')
    my_dict = {'x':x, 'y':y}
    torch.save(my_dict, 'mydict')
    my_dicts = torch.load('mydict')
    
    net = nn.Linear(20, 3)
    x = torch.randn(size=(2, 20))
    y = net(x)
    torch.save(net.state_dict(), 'net.params')
    net_clone = nn.Linear(20, 3)
    net_clone.load_state_dict(torch.load('net.params'))
    print(net_clone.eval())
    

if __name__ == '__main__':
    #test_linear_regression()
    #test_linear_regression_use_torch_api()
    #test_softmax()
    #test_softmax_use_torch_api()  
    #test_multiplayer_perceptron()
    #test_multiplayer_perceptron_use_torch_api()
    test_kaggle_house()
    #test_calculate()
