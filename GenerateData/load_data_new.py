import sys
sys.path.insert(1, '../../Non-Residual-GANN')
sys.path.insert(2, '../GetClusters')
from datasets import load_dataset, Dataset
from SamplingComparissons import CompareModels
from Utils import loadGenerator
import pandas as pd
import pickle
from GANN.SamplingStrategies import SamplingTechniques
import torch
import tqdm
from torch.utils.data import Dataset as DatasetPytorch
from GetClusters.differenceMetrics import bucket_diff_top_k
import numpy as np
from sys import getsizeof
import psutil
import config

BIG_MODEL_PATH = "/home/ubuntu/Non-Residual-GANN/Models/MLE+VH8-BIG-BIG-A8-R2-1670318134.3765588/"
SMALL_MODEL_PATH = "/home/ubuntu/Non-Residual-GANN/Models/MLE+VH2-Mini-BIG-A8-R2-1670318134.2979872"
TOKENIZED_DATA_PATH = "../pipeline/train_data/WikitextDataset-16384-64-ls-100-Train-10pct.pkl"
TOKENIZER_PATH = "/home/ubuntu/Non-Residual-GANN/Tokenizers/WikitextTokenizer-16384-64-ls-100"
DEVICE = "cuda:0"
CHECKPOINT = "Epoch-8"
SEQUENCE_LENGTH = 64  # Of tokenized pretrained data.
VOCAB_LENGTH = 16384
NUM_ALTERATIONS = 8
NUM_SAMPLES = 100000  # should be divisible by 1000
TOPK = 256

SAVE = False
TRUNCATE = False


def generateData(functions, bucket_indices, tokenized_data_path=TOKENIZED_DATA_PATH, sequence_length=SEQUENCE_LENGTH,
                 num_samples=NUM_SAMPLES, truncate=False, topk=256, save=False):

    print("  => LOADING BIG MODEL")
    bigModel = loadGenerator(BIG_MODEL_PATH, CHECKPOINT).to(DEVICE)
    bigModel.eval()

    print("  => LOADING SMALL MODEL")
    smallModel = loadGenerator(SMALL_MODEL_PATH, CHECKPOINT).to(DEVICE)
    smallModel.eval()

    print("  => LOADING PRE TOKENIZED DATASET")
    with open(tokenized_data_path, "rb") as f:
        data = pickle.load(f)

    print("  => FORMATTING DATA")
    new_data = [data[i] for i in range(len(data)) if len(data[i]) == sequence_length][:num_samples]
    del data
    data = Dataset.from_pandas(pd.DataFrame({"data": new_data}))
    data.set_format("torch")
    loader = iter(torch.utils.data.DataLoader(data["data"], batch_size=1))  # added iter so it doesn't load the first data over and over

    num_features = sum([config.function_feature_dict[f"{function.__name__}"] for function in functions])

    for i in tqdm.tqdm(range(2)):
        print("  => INITIALIZING ARRAYS")
        big_probabilities = torch.zeros((num_samples//10, sequence_length, TOPK))
        small_probabilities = torch.zeros((num_samples//10, sequence_length, TOPK))

        big_indices = torch.zeros((num_samples//10, sequence_length, TOPK))
        small_indices = torch.zeros((num_samples//10, sequence_length, TOPK))

        tmp_big = torch.zeros((500, SEQUENCE_LENGTH, VOCAB_LENGTH))
        tmp_small = torch.zeros((500, SEQUENCE_LENGTH, VOCAB_LENGTH))

        if i < 9:  # only get features for the first 9 files, the 10th is test data
            features = torch.zeros((num_samples//10, sequence_length, num_features))

        print("  => DONE INITIALIZING ARRAYS")

        def getData(inputIDs):
            smallProb, _, _, _, _ = CompareModels._generateProbAndSamples(
                inputIDs=inputIDs,
                numAlterations=NUM_ALTERATIONS,
                model=smallModel,
                samplingStrat=SamplingTechniques.SAMPLE_NO_REPLACEMENT,
                samplingParams={}
            )
            bigProb, _, _, _, _ = CompareModels._generateProbAndSamples(
                inputIDs=inputIDs,
                numAlterations=NUM_ALTERATIONS,
                model=bigModel,
                samplingStrat=SamplingTechniques.SAMPLE_NO_REPLACEMENT,
                samplingParams={}
            )

            return smallProb, bigProb

        index = 0
        for example in tqdm.tqdm(loader):
            if index == num_samples//10:
                break
            inputIDs = (torch.tensor(example).to(DEVICE))  # size [batch_size, SEQUENCE_LENGTH, VOCAB_LENGTH]
            smallProb, bigProb = getData(inputIDs)

            tmp_small[index % 500] = smallProb.detach().cpu()
            tmp_big[index % 500] = bigProb.detach().cpu()

            index += 1

            if index % 500 == 0 and truncate:
                small_ordered, small_indx = torch.sort(tmp_small, descending=True)
                big_ordered, big_indx = torch.sort(tmp_big, descending=True)

                small_ordered, small_indx = small_ordered[:, :, :topk], small_indx[:, :, :topk]
                big_ordered, big_indx = big_ordered[:, :, :topk], big_indx[:, :, :topk]

                small_probabilities[index-500:index, :, :] = small_ordered
                big_probabilities[index - 500:index, :, :] = big_ordered

                small_indices[index-500:index, :, :] = small_indx
                big_indices[index - 500:index, :, :] = big_indx

                if i < 9:
                    # create feature vector!
                    num_features_function = 0
                    for j in range(len(functions)):
                        diffs = functions[j](tmp_small, tmp_big, bucket_indices)
                        if len(diffs.shape) == 2:
                            diffs = np.expand_dims(diffs, -1)

                        start = num_features_function
                        num_features_function += config.function_feature_dict[f"{functions[j].__name__}"]
                        print(f"shape bucket_diffs: {diffs.shape}")
                        features[index-500:index, :, start:num_features_function] = torch.from_numpy(diffs)
                        print(f"shape features: {features.shape}")

        if save:
            print("  => SAVING...")
            with open(f"../pipeline/new_data/big_{num_samples//10}_{i}.pkl", "wb") as f:
                pickle.dump(big_probabilities, f)

            with open(f"../pipeline/new_data/small_{num_samples//10}_{i}.pkl", "wb") as g:
                pickle.dump(small_probabilities, g)

            with open(f"../pipeline/new_data/indices_big_{num_samples//10}_{i}.pkl", "wb") as f:
                pickle.dump(big_indices, f)

            with open(f"../pipeline/new_data/indices_small_{num_samples//10}_{i}.pkl", "wb") as f:
                pickle.dump(small_indices, f)

            if i < 9:
                with open(f"../pipeline/new_data/features_{num_samples//10}_{i}.pkl", "wb") as g:
                    pickle.dump(features, g)

    return small_probabilities, big_probabilities, small_indices, big_indices, features


# if __name__ == "__main__":
    # print("  => GENERATING ALL DATA")
    # small_probs, big_probs, small_indices, big_indices = generateData(
    #     TOKENIZED_DATA_PATH, SEQUENCE_LENGTH, NUM_SAMPLES, truncate=TRUNCATE, save=SAVE)
    # print(len(small_probs))
    # print(len(small_indices))
    # print(small_probs[0][0])
    # print(len(small_probs[0][0]))
