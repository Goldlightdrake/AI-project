import torch


def train_epoch(model, X_train, y_train, loss_fn, optimizer):
    model.train()

    y_pred = model(X_train).squeeze()

    train_loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    return train_loss


def test_epoch(model, X_test, y_test, loss_fn, metrics):
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_test).squeeze()
        y_pred = torch.round(y_logits)
        test_loss = loss_fn(y_pred, y_test)
        scores = []
        for metric in metrics:
            scores.append(metric(y_test, y_pred))

    return scores, test_loss
