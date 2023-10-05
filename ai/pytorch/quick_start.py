#!/bin/bash/python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os

# TorchText, TorchVision, and TorchAudio

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(device, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def start_train(device, train_data, test_data, batch_size, epochs):       
    # Create data loaders.
    train_dataloader = DataLoader(train_data, batch_size=batch_size) 
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    '''
       The training process is conducted over several iterations (epochs). 
       During each epoch, the model learns parameters to make better predictions. 
       we’d like to see the accuracy increase and the loss decrease with every epoch.
    ''' 
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test(device, test_dataloader, model, loss_fn)
    print("Done!")
    
    ## save model
    torch.save(model.state_dict(), model_file)
    print("Saved PyTorch Model State to {}", model_file)


def predict(device,  test_data, index):
    '''
       load model and use the model to make predictions
    '''
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_file))   ## load model
    
    ## use model to make predictions.
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    
    model.eval()
    x, y = test_data[index][0], test_data[index][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    return device 

def main(is_train=True): 
    # Download training data from open datasets.
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    
    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    print(f"train data is : {len(train_data)}, and test data is:  {len(test_data)}")
    print(f"shape of test_data: {test_data[0][0].shape}")
    print(f"label of test_data[0] is : {test_data[0][1]}")
    #print(f"data of test_data[0] is : {test_data[0][0]}")
  
    if not os.path.exists('models'):
       os.makedirs('models')

    device = get_device()
    if is_train :
        batch_size = 64
        epochs = 5
        start_train(device, train_data, test_data, batch_size, epochs)

    print(f"begin to predict:")
    for i in range(10):
        predict(device, test_data, i)

model_file='models/quick_start_model.pth'
if __name__ == '__main__':
   main() 
