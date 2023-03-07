import pickle


# ..\\code\\Non-Residual-GANN\\DataManagers\\WikitextDataset\\PreTokenized\\WikitextDataset-16384-64-ls-100-Valid-100.pkl
# data = pickle.load(open("WikitextDataset-16384-64-ls-100-Train-10pct.pkl", "rb"))
data = pickle.load(open("train_data/train_small_10.pkl", "rb"))

print(len(data))
print(type(data))
print(data[0].shape)

#data = pickle.load(open("WikitextDataset-16384-64-ls-100-Train-10pct.pkl", "rb"))
#token_dict = pickle.load(open("reverseVocab.p", "rb"))
#tokens = [token_dict[i] for i in range(len(token_dict))]

# for token in data[2]:
#     print(token_dict[token])

#print(len(token_dict.keys()))

#print(type(data))

#print(len(list(filter(lambda x: len(x) == 64, data))))

