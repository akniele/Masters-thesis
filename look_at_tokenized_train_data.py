import pickle

# data = pickle.load(open("train_data/train_big_100000_0.pkl", "rb"))
# data2 = pickle.load(open("train_data/train_big_100000_1.pkl", "rb"))
#
# # for i in range(500):
# #     print(i, (data[i][1] == data2[i][1]).all())
#
#
# print(data.shape)
# print(data[9999][63][:20])
# # # print(data[0][1][:5])
# # #
# print(data2.shape)
# print(data2[9999][63][:20])
# # print(data2[0][1][:5])s



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
print("  => LOADING PRE TOKENIZED DATASET")
with open(TOKENIZED_DATA_PATH, "rb") as f:
    data = pickle.load(f)

print("  => FORMATTING DATA")
new_data = [data[i] for i in range(len(data)) if len(data[i]) == 64][:20000]
del data

data = Dataset.from_pandas(pd.DataFrame({"data": new_data}))
data.set_format("torch")
loader = iter(torch.utils.data.DataLoader(data["data"], batch_size=1))

print(new_data[9986])

print(new_data[9986+10000])

i = 0
for example in loader:
    if i == 9986 or i == (9986+9999):
        print(example)
    i += 1
