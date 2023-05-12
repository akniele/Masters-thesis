import sys
sys.path.insert(1, '../GetClusters')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
from differenceMetrics import sort_probs

max_index = 16

mean_array = torch.zeros(max_index)  # for getting the mean probability per index in the training data
mean_sum = 0  # for getting the mean total probability mass in the top-256 of the training data

mean_array_small = np.zeros(max_index)  # same but for small model
mean_sum_small = 0

for i in range(9):
    with open(f"../train_data/big_10000_{i}.pkl", "rb") as f:
        with open(f"../train_data/small_10000_{i}.pkl", "rb") as g:
            bigprobs = pickle.load(f)
            bigprobs = bigprobs[:, :, :max_index]
            mean_1 = torch.mean(bigprobs, dim=0)
            mean_2 = torch.mean(mean_1, dim=0)
            mean_array += mean_2

            sum_1 = torch.sum(bigprobs, dim=-1)
            sum_2 = torch.mean(sum_1)
            mean_sum += sum_2

            smallprobs = pickle.load(g)
            smallprobs = smallprobs[:, :, :max_index]

            _, smallprobs_sort_by_big, _ = sort_probs(bigprobs.numpy(), smallprobs.numpy())
            mean_1_small = np.mean(smallprobs_sort_by_big, axis=0)
            mean_2_small = np.mean(mean_1_small, axis=0)
            mean_array_small += mean_2_small

            sum_1_small = torch.sum(smallprobs, dim=-1)
            sum_2_small = torch.mean(sum_1_small)
            mean_sum_small += sum_2_small


final_means_big = mean_array / 9
final_means_small = mean_array_small / 9

mean_sum_small /= 9
mean_sum /= 9

print(f"For the big model, the mean probability mass in the top 256 probabilities in the training data is "
      f"{mean_sum*100}%")
print(f"For the small model, the mean probability mass in the top 256 probabilities in the training data is "
      f"{mean_sum_small*100}%")

x_values = torch.arange(len(final_means_big))

plt.plot(x_values, final_means_big, label="big model")
plt.plot(x_values, final_means_small, label="small model")
plt.legend(loc='upper right', bbox_to_anchor=(0.85, 0.55))

plt.title("Mean probabilities across the distributions in the training data")
plt.xlabel('Indices')
plt.ylabel('Mean probability')

plt.savefig(f"/home/ubuntu/pipeline/plots/try_plot.png")
plt.close()




#import sys
#sys.path.insert(1, '../GetClusters')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
#from differenceMetrics import sort_probs


# count_bigger_than_one = 0
# count_bigger_than_one_small = 0
# count_bigger_than_one_sort_small = 0
#
# for i in range(9):
#     with open(f"../train_data/big_10000_{i}.pkl", "rb") as f:
#         with open(f"../train_data/small_10000_{i}.pkl", "rb") as g:
#             with open(f"../train_data/small_10000_{i}sep.pkl", "rb") as h:
#                 small_sort_probs = pickle.load(h)
#                 bigprobs = pickle.load(f)
#                 smallprobs = pickle.load(g)
#                 sum_1 = torch.sum(bigprobs, dim=-1)
#                 sum_2 = torch.sum(smallprobs, dim=-1)
#                 sum_3 = torch.sum(small_sort_probs, dim=-1)
#                 count_bigger_than_one += torch.numel(sum_1[sum_1 > 1])
#                 count_bigger_than_one_small += torch.numel(sum_2[sum_2 > 1])
#                 count_bigger_than_one_sort_small += torch.numel(sum_3[sum_3 > 1])
#
#
# print(f"count bigger than one: {count_bigger_than_one}")
# print(f"count bigger than one small: {count_bigger_than_one_small}")
# print(f"count bigger than one small sort: {count_bigger_than_one_sort_small}")


# max_index = [2, 5, 10, 20, 30, 40, 50, 100]

#mean_array = torch.zeros(max_index)  # for getting the mean probability per index in the training data
# mean_sum = 0  # for getting the mean total probability mass in the top-256 of the training data

#mean_array_small = np.zeros(max_index)  # same but for small model
# mean_sum_small = 0
#
# mean_list = []
# mean_list_small = []
#
# for index in max_index:
#
# mean_array = torch.zeros()
# for i in range(9):
#     with open(f"../train_data/big_10000_{i}.pkl", "rb") as f:
#         with open(f"../train_data/small_10000_{i}sep.pkl", "rb") as g:
#             bigprobs = pickle.load(f)
#             #bigprobs = bigprobs[:, :, :index]
#             mean_1 = torch.mean(bigprobs, dim=0)
#             mean_2 = torch.mean(mean_1, dim=0)
#             mean_array += mean_2
#
#             # sum_1 = torch.sum(bigprobs, dim=-1)
#             # sum_2 = torch.mean(sum_1)
#             # mean_sum += sum_2
#
#             smallprobs = pickle.load(g)
#             #smallprobs = smallprobs[:, :, :index]
#
#             # _, smallprobs_sort_by_big, _ = sort_probs(bigprobs.numpy(), smallprobs.numpy())
#             # mean_1_small = np.mean(smallprobs_sort_by_big, axis=0)
#             # mean_2_small = np.mean(mean_1_small, axis=0)
#             # mean_array_small += mean_2_small
#
#             sum_1_small = torch.sum(smallprobs, dim=-1)
#             sum_2_small = torch.mean(sum_1_small)
#             mean_sum_small += sum_2_small
#
# mean_list_small.append(mean_sum_small / 9)
# mean_list.append(mean_sum / 9)
#
# mean_sum_small = 0
# mean_sum = 0
#
# final_means_big = mean_array / 9
# final_means_small = mean_array_small / 9
#
# print(f"For the big model, the mean probability mass in the top 256 probabilities in the training data is "
#       f"{mean_list}")
# print(f"For the small model, the mean probability mass in the top 256 probabilities in the training data is "
#       f"{mean_list_small}")
#
# x_values = torch.arange(len(final_means_big))
#
# plt.plot(x_values, final_means_big, label="big model")
# plt.plot(x_values, final_means_small, label="small model")
# plt.legend(loc='upper right', bbox_to_anchor=(0.85, 0.55))
#
# plt.title("Mean probabilities across the distributions in the training data")
# plt.xlabel('Indices')
# plt.ylabel('Mean probability')
#
# plt.savefig(f"/home/ubuntu/pipeline/plots/try_plot.png")
# plt.close()
