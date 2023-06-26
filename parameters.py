from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from imblearn.metrics import geometric_mean_score

n_chunks = 100
epochs = 1000
learning_rate = 0.006
metrics = [
  f1_score,
  balanced_accuracy_score,
  geometric_mean_score,
  recall_score
]