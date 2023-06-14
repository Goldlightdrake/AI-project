from sklearn.metrics import accuracy_score
import torch

def train_epoch(model, X_train, y_train, loss_fn, optimizer):
    model.train()

    y_logits = model(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    train_loss = loss_fn(y_pred, y_train) 

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    return train_loss

def test_epoch(model,X_test, y_test,loss_fn):
    model.eval()
    with torch.inference_mode():
      test_logits = model(X_test).squeeze()
      test_pred = torch.round(torch.sigmoid(test_logits))
      test_loss = loss_fn(test_pred, y_test.type(torch.float)) 
      acc = accuracy_score(y_test,test_pred)

    return acc, test_loss

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()