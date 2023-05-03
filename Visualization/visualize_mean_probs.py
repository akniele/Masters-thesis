import numpy as np
import matplotlib.pyplot as plt
import pickle

mean_array = np.zeros(256)  # for getting the mean probability per index in the training data
mean_sum = 0  # for getting the mean total probability mass in the top-256 of the training data

for i in range(9):
    with open(f"../train_data/big_10000_{i}.pkl", "rb") as f:
        bigprobs = pickle.load(f)
        mean_1 = np.mean(bigprobs, axis=0)
        mean_2 = np.mean(mean_1, axis=0)
        mean_array += mean_2

        sum_1 = np.sum(bigprobs, axis=-1)
        sum_2 = np.mean(sum_1)
        mean_sum += sum_2

final_means = mean_array / 9

mean_sum /= 9
print(f"The mean probability mass in the top 256 probabilities in the training data is {mean_sum*100}%")

x_values = np.arange(len(final_means))

plt.plot(x_values, final_means)

plt.title("Mean probabilities across the distributions in the training data")
plt.xlabel('Indices')
plt.ylabel('Mean probability')

plt.savefig(f"/home/ubuntu/pipeline/plots/try_plot.png")
plt.close()
