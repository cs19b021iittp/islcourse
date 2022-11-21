import sklearn
import scipy
import seaborn
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as tn
import torchvision.transforms as tt
import torch.utils as utils

import matplotlib.pyplot as plt
import seaborn

from sklearn.datasets import make_blobs
def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  # write your code ...
  return X,y

from sklearn.datasets import make_circles
def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y = make_circles(n_samples=n_points, shuffle=True,factor=0.3, noise=0.05, random_state=0)
  # write your code ...
  return X,y

from sklearn.datasets import fetch_openml
def get_data_mnist():
  pass
  # write your code here
  # Refer to sklearn data sets

  mnist = fetch_openml('MNIST original')
  X=mnist.data
  y=mnist.target
  # write your code ...
  return X,y

from sklearn.cluster import KMeans 
def build_kmeans(X,k=10):
  pass
  # k is a variable, calling function can give a different number
  # Refer to sklearn KMeans method
   # this is the KMeans object
  km= KMeans(n_clusters=k,random_state=0) 
  km.fit(X)
  # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
  # centers = km.cluster_centers_
  # plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
  # write your code ...
  return km

def assign_kmeans(km=None,X=None):
  pass
  # For each of the points in X, assign one of the means
  # refer to predict() function of the KMeans in sklearn
  # write your code ...
  ypred = km.predict(X)
  return ypred

from sklearn import metrics
def compare_clusterings(ypred_1=None,ypred_2=None):
  pass
  # refer to sklearn documentation for homogeneity, completeness and vscore
  h=metrics.homogeneity_score(ypred_1, ypred_2)
  c=metrics.completeness_score(ypred_1, ypred_2)
  v=metrics.v_measure_score(ypred_1, ypred_2)
  return h,c,v

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_lr_model(X=None, y=None):
  # Build logistic regression, refer to sklearn
  lr_model = LogisticRegression(solver="liblinear",fit_intercept=False)
  lr_model.fit(X,y)
  return lr_model

def build_rf_model(X=None, y=None):
  # Build Random Forest classifier, refer to sklearn
  rf_model = RandomForestClassifier(random_state=400)
  rf_model.fit(X,y)
  return rf_model

def get_metrics(model=None,X=None,y=None):
  pass
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  acc, prec, rec, f1, auc = 0,0,0,0,0
  y_pred = model.predict(X)
  acc = accuracy_score(y, y_pred)
  prec = precision_score(y, y_pred, average='micro')
  rec =  recall_score(y, y_pred , average='micro')
  f1 =  f1_score(y, y_pred, average='micro' )
  auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr' )
  return acc, prec, rec, f1, auc

from sklearn.model_selection import GridSearchCV
def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2

   lr_param_grid = {
      "max_iter": [100, 200, 500],
      "penalty": ["l1","l2"],
      "solver" : ["liblinear"]
   }

   return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = { 
    'n_estimators' : [1, 10, 100],
    'max_depth' : [1,10,None],
    'criterion' :['gini', 'entropy'],

  }

  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  print(model.get_params().keys())
  top1_scores = []
  for scoring in metrics:
    grid_search_cv = GridSearchCV(model,param_grid, cv=cv, scoring=scoring)
    grid_search_cv.fit(X,y)
    top1_scores.append(grid_search_cv.best_score_)

  return top1_scores

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
device = "cuda" if torch.cuda.is_available() else "cpu"

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.fc_encoder = nn.Sequential(nn.Linear(in_features=inp_dim,out_features=hid_dim),)
    self.fc_decoder = nn.Sequential(nn.Linear(in_features=hid_dim,out_features=inp_dim),) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Sequential(nn.Linear(in_features=hid_dim,out_features=num_classes),) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax =nn.softmax() #write your code - softmax object

    
  def forward(self,x):
    x = nn.Flatten(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    x=(y_pred-yground)**2
    lc1 = torch.mean(x) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist\
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
  # convert to tensor
  trainset = datasets.FashionMNIST(root='.', train=True, transform=ToTensor(), download=True)
  testset = datasets.FashionMNIST(root='.', transform=ToTensor(), download=True)
  train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
  test_loader = DataLoader(testset, batch_size=32)
  X, y = trainset,testset
  # write your code
  return X,y

