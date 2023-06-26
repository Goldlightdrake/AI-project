from streams import weights_streams, drift_streams, chunk_streams
from parameters import n_chunks, epochs, learning_rate, metrics
import numpy as np
from sea import MySEA
from strlearn.evaluators import TestThenTrain
from parameters import metrics

data_stream = chunk_streams
scores = np.zeros((len(data_stream), n_chunks - 1, len(metrics)))
losses = np.zeros((len(data_stream), n_chunks - 1))

for stream_index, stream in enumerate(data_stream):
    clf = MySEA()
    evaluator = TestThenTrain(metrics=metrics)

    evaluator.process(stream, clf)
    scores[stream_index, :] = evaluator.scores

np.savez("output", scores=scores)
