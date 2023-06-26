import numpy as np
from matplotlib import pyplot as plt
from parameters import n_chunks, metrics
from streams import weights_streams, drift_streams, chunk_streams
from scipy.stats import ttest_rel
from tabulate import tabulate

if __name__ == "__main__":
    data_stream = chunk_streams
    output = np.load("experiments/sea.npz")
    scores = output["scores"]
    t_scores = np.zeros((len(data_stream), len(data_stream)))
    p_scores = np.zeros((len(data_stream), len(data_stream)))
    better = np.full((len(data_stream), len(data_stream)), False)

    # for m_index, metric in enumerate(metrics):
    #     for s_index, stream in enumerate(data_stream):
    #         temp_scores = scores[:, s_index, :, m_index]
    #         mean_scores = np.mean(temp_scores, axis=1)
    #         for i in range(2):
    #             for j in range(2):
    #                 t_score, p_score = ttest_rel(temp_scores[i], temp_scores[j])
    #                 better[i, j] = (
    #                     np.mean(temp_scores, axis=1)[i]
    #                     > np.mean(temp_scores, axis=1)[j]
    #                 )
    #                 t_scores[i, j] = t_score
    #                 p_scores[i, j] = p_score

            # for i, _ in enumerate(data_stream):
            #   for j, _ in enumerate(data_stream):
            #      if(significant_better[i, j]):
            #         print(i, f'with score {np.mean(temp_scores, axis=1)[i]} better then', j, f'with score {np.mean(temp_scores, axis=1)[j]}')

            # print(significant_better)
    # significant = p_scores < 0.05
    # significant_better = significant * better
    # print(
    #     tabulate(
    #         significant_better,
    #         headers=["NN", "SEA"],
    #         showindex=["NN", "SEA"],
    #         tablefmt="grid",
    #     )
    # )

    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(17, 10)
    fig.tight_layout(pad=5.0)
    fig.suptitle("Metrics scores for unbalanced streams", fontsize=24)
   
    ax[0].plot(
        range(n_chunks - 1),
        scores[1, :],
        label=["f1", "balanced_accuracy", "G-mean", "recall"],
    )
    ax[1].plot(
        range(n_chunks - 1),
        scores[1, :],
        label=["f1", "balanced_accuracy", "G-mean", "recall"],
    )
    ax[2].plot(
        range(n_chunks - 1),
        scores[1, :],
        label=["f1", "balanced_accuracy", "G-mean", "recall"],
    )

    ax[0].set_title("Stream chunk_size 200")
    ax[1].set_title("Stream chunk_size 500")
    ax[2].set_title("Stream chunk_size 1000")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    [a.set_xlabel("chunk") for a in ax]

    [a.set_ylabel("scores") for a in ax]

    # [a.set_ylabel('losses') for a in ax[1]]

    # [a.set_ylabel('losses') for a in ax[1]]

    plt.show()
    fig.savefig('output_images/unbalanced_streams_sea_metrics.png', format='png')
