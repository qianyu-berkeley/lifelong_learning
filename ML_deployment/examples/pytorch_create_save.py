# -*- coding: utf-8 -*-
"""pytorch_create_save.ipynb

Original file is a colab notebook
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

dataset = pd.read_csv('https://raw.githubusercontent.com/futurexskill/ml-model-deployment/main/storepurchasedata_large.csv')


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.20,random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Xtrain_ = torch.from_numpy(X_train).float()
Xtest_ = torch.from_numpy(X_test).float()

Xtrain_

ytrain_ = torch.from_numpy(y_train)
ytest_ = torch.from_numpy(y_test)

ytrain_

Xtrain_.shape, ytrain_.shape

Xtest_.shape, ytest_.shape

input_size=2
output_size=2
hidden_size=10

class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = torch.nn.Linear(input_size, hidden_size)
       self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
       self.fc3 = torch.nn.Linear(hidden_size, output_size)


   def forward(self, X):
       X = torch.relu((self.fc1(X)))
       X = torch.relu((self.fc2(X)))
       X = self.fc3(X)

       return F.log_softmax(X,dim=1)

model = Net()

import torch.optim as optim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.NLLLoss()

epochs = 100

for epoch in range(epochs):
  optimizer.zero_grad()
  Ypred = model(Xtrain_)
  loss = loss_fn(Ypred,  ytrain_)
  loss.backward()
  optimizer.step()
  print('Epoch',epoch, 'loss',loss.item())

list(model.parameters())
torch.from_numpy(sc.transform(np.array([[40,20000]]))).float()

y_cust = model(torch.from_numpy(sc.transform(np.array([[40,20000]]))).float())
_, predicted_20_40000 = torch.max(y_cust_20_40000.data,-1)

# save as .pt
torch.save(model,'customer_buy.pt')
restored_model = torch.load('customer_buy.pt')
y_cust = restored_model(torch.from_numpy(sc.transform(np.array([[40,20000]]))).float())
_, predicted = torch.max(y_cust_20_40000.data,-1)


# save as dictionary
torch.save(model.state_dict(),'customer_buy_state_dict')
new_predictor = Net()
y_cust = new_predictor(torch.from_numpy(sc.transform(np.array([[40,20000]]))).float())

#!zip -r customer_buy_state_dict.zip customer_buy_state_dict

from google.colab import files
files.download('customer_buy_state_dict.zip')

