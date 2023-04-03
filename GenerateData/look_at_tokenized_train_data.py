import pickle
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
import sys

sys.path.append('../')
from GetClusters.differenceMetrics import bucket_diff_top_k

data_small = pickle.load(open("../new_data/small_10000_0.pkl", "rb"))

data_big = pickle.load(open("../new_data/big_10000_0.pkl", "rb"))

features = pickle.load(open("../new_data/features_10000_0.pkl", "rb"))

print(f"data small shape: {data_small.size()}")
print(f"data big shape: {data_big.size()}")
print(f"features shape: {features.size()}")

print(f"data small first distribution: {data_small[0][0][:50]}")
print(f"data big first distribution: {data_big[0][0][:50]}")

bucket_difference = torch.sum(data_big[0][1][:10]) - torch.sum(data_small[0][1][:10])

print(f"bucket difference: {bucket_difference}, should be the same as feature: {features[0][1][0]}")

print(f"first three feature vectors in file:\n {features[:1, :3]}")

bucket_diffs = bucket_diff_top_k(data_small, data_big, indices=[10, 35])

print(f"bucket diffs shape: {bucket_diffs.shape}")

print(f"data type bucket diffs: {type(bucket_diffs[0])}")

print(f"bucket diffs [0][0]: {bucket_diffs[0][0]}")

ind = np.nonzero(bucket_diffs == 0.012411028146743774)

print(f"index: {ind}")


# #data2 = pickle.load(open("../train_data/train_big_100000_1.pkl", "rb"))
#
# data3 = pickle.load(open("../train_data/train_big_tmp_1000_0.pkl", "rb"))
#
# for i in range(10):
#     if (data[i][1] == data3[i][1]).all():
#         print(i)
#         print(data[i][1][:10], data3[i][1][:10])


# print(data.shape)
# print(data[9999][63][:20])
# # # print(data[0][1][:5])
# # #
# print(data2.shape)
# print(data2[9999][63][:20])
# print(data2[0][1][:5])s



# ..\\code\\Non-Residual-GANN\\DataManagers\\WikitextDataset\\PreTokenized\\WikitextDataset-16384-64-ls-100-Valid-100.pkl
# data = pickle.load(open("WikitextDataset-16384-64-ls-100-Train-10pct.pkl", "rb"))
# data = pickle.load(open("scaled_features.pkl", "rb"))
#
# print(data.shape)
# print(type(data))
# print(data[0].shape)
#
# print(9000*64)

#data = pickle.load(open("WikitextDataset-16384-64-ls-100-Train-10pct.pkl", "rb"))
#token_dict = pickle.load(open("reverseVocab.p", "rb"))
#tokens = [token_dict[i] for i in range(len(token_dict))]

# for token in data[2]:
#     print(token_dict[token])

#print(len(token_dict.keys()))

#print(type(data))

#print(len(list(filter(lambda x: len(x) == 64, data))))

#

# TOKENIZED_DATA_PATH = "../Data/WikitextDataset-16384-64-ls-100-Train-10pct.pkl"
# print("  => LOADING PRE TOKENIZED DATASET")
# with open(TOKENIZED_DATA_PATH, "rb") as f:
#     data = pickle.load(f)
#
# print("  => FORMATTING DATA")
# new_data = [data[i] for i in range(len(data)) if len(data[i]) == 64][:100000]
# del data
#
# data = Dataset.from_pandas(pd.DataFrame({"data": new_data}))
# data.set_format("torch")
# loader = iter(torch.utils.data.DataLoader(data["data"], batch_size=1))
#
# token_dict = pickle.load(open("../Data/reverseVocab.p", "rb"))
#
# id_dict = dict()
#
#
# #for i in range(10):
# count = 0
# for j, example in enumerate(loader):
#     example = example.squeeze(0)
#     lst_example = [str(i) for i in example.tolist()]
#     str_example = "".join(lst_example)
#     if str_example in id_dict.keys():
#         count += 1
#         text = [token_dict[example[k].item()] for k in range(len(example))]
#         print(f"duplicate text: {text}")
#         id_dict[str_example].append(j)
#         print(f"other occurrences: {id_dict[str_example]} ")
#     else:
#         id_dict[str_example] = [j]
#
# print(f"count: {count}")
