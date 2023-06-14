from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import torch 
import torch.nn as nn
from torch.optim import Adam
from strlearn.streams import StreamGenerator
from matplotlib import pyplot as plt
from helpers import train_epoch, test_epoch, reset_weights
from pathlib import Path 

epochs = 1000

stream = StreamGenerator(
  n_classes=2,
  n_informative=2,
  n_redundant=0,
  n_repeated=0,
  n_features=2,
  random_state=105,
  n_chunks=100,
  chunk_size=500,
)

X, y = stream.get_chunk()
kf = KFold(n_splits=5, shuffle=True)

while stream.get_chunk():
  X, y = stream.get_chunk()
  kf = KFold(n_splits=5, shuffle=True)
  for i, (train_index, test_index) in enumerate(kf.split(X)):
      print(f'Fold: {i + 1}')
      X_train = torch.tensor(X[train_index], dtype=torch.float)
      y_train = torch.tensor(y[train_index], dtype=torch.float)
      X_test = torch.tensor(X[test_index], dtype=torch.float)
      y_test = torch.tensor(y[test_index], dtype=torch.float)

      model = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
      )

      optimizer = Adam(model.parameters(), lr=0.001 )
      loss_fn = nn.BCEWithLogitsLoss()

      for epoch in range(epochs):
          train_loss = train_epoch(model, X_train, y_train, loss_fn, optimizer)
          acc, test_loss = test_epoch(model, X_test, y_test, loss_fn)
          if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Test Accuracy: {acc}')
          
          