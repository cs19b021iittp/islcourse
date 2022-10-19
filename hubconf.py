import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
loss_fn = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"

def kali():
  print ('kali')
  
# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname
class cs19b021NN(nn.Module):
  def __init__(self):
        super(cs19b021NN, self).__init__()
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
        
        
  def load_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
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
    print('loading data')
    return training_data, test_data
        
      
        
    def create_dataloaders(training_data, test_data, batch_size=64):
  # Create data loaders.
  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)

  for X, y in test_dataloader:
      print(f"Shape of X [N, C, H, W]: {X.shape}")
      print(f"Shape of y: {y.shape} {y.dtype}")
      break
  print('returning dataloaders')
  return train_dataloader, test_dataloader
  
    
# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):

  model = cs19b021NN().to(device)

  return model

training_data, test_data = load_data();
train_dataloader, test_dataloader = create_dataloaders(training_data, test_data, batch_size=64)
  
  
  return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model_advanced(train_data_loader=None, n_epochs=10,lr=1e-4,config=None):
  model = get_model(train_data_loader=train_dataloader, n_epochs=10);

  # To train a model, we need a loss function and an optimizer.
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  size = len(train_dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(train_dataloader):
      X, y = X.to(device), y.to(device)

      # Compute prediction error
      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if batch % 100 == 0:
          loss, current = loss.item(), batch * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)
def test_model(model1=None, test_data_loader=None):

  accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
  # write your code here as per instructions
  # ... your code ...
  # ... your code ...
  # ... and so on ...
  # calculate accuracy, precision, recall and f1score
  
  print ('Returning metrics... (rollnumber: xx)')
  
  return accuracy_val, precision_val, recall_val, f1score_val
