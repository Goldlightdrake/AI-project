from streams import weights_streams, drift_streams, chunk_streams
from parameters import n_chunks, epochs, learning_rate, metrics
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from helpers import train_epoch, test_epoch
from neural_classificator import NeuralClassificator

data_stream = chunk_streams
scores = np.zeros((len(data_stream), n_chunks - 1, epochs, len(metrics)))
losses = np.zeros((len(data_stream), n_chunks - 1, epochs))

for stream_index, stream in enumerate(data_stream):
    classification_model = NeuralClassificator()
    optimizer = Adam(classification_model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    ######Feeding model########
    for chunk in range(n_chunks):
        X, y = stream.get_chunk()
        X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        print(f"Chunk: {chunk + 1}")
        for epoch in range(epochs):
            if chunk != 0:
                score_list, test_loss = test_epoch(
                    model=classification_model,
                    X_test=X,
                    y_test=y,
                    loss_fn=loss_fn,
                    metrics=metrics,
                )
                for score_index, score in enumerate(score_list):
                    scores[stream_index, chunk - 1, epoch, score_index] = score
                losses[stream_index, chunk - 1, epoch] = test_loss
                if epoch % 100 == 0:
                    print(f"epoch: {epoch}, Test Loss: {train_loss}, Acc: {score_list}")
            train_loss = train_epoch(
                model=classification_model,
                X_train=X,
                y_train=y,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )

np.savez("output", scores=scores, losses=losses)
