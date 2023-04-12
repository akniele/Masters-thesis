import sys
sys.path.insert(1, '../Non-Residual-GANN')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#from datasets import load_dataset, Dataset
from SamplingComparissons.AlterationProbabilityOverlap import generateAlterationProbabilityDistribution
from SamplingComparissons import CompareModels
from Utils import loadGenerator
import pandas as pd
import pickle
from GANN.SamplingStrategies import SamplingTechniques
import torch
import torch.nn as nn
import tqdm
import numpy as np
import sys
import os
from functools import partialmethod
import json
#from torch.utils.data import Dataset as DatasetPytorch
from ClassifierFiles.trainingDataClassifier import prepare_training_data
from Transformation.fill_up_distributions import fill_multiple_distributions
from Transformation.transformation import get_distances, get_mean_distances
from sklearn.model_selection import train_test_split
import time
from config import timeit

"""
Sample Mapping Network is a neural network that maps
the probability distribution from one network to the inclusion distribution of another.
Maps the probability from big model to the inclusion distribution of the small model
"""


BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5
BIG_MODEL_PATH = "/raid/fredrikc/Non-Residual-GANN/MLE-Med-Med-A8-R2-SampleNoRep-ReallySmallData-1669347015.5838063/"
SMALL_MODEL_PATH = "/raid/fredrikc/Non-Residual-GANN/MLE-Mini-Big-A8-R2-SampleNoRep-ReallySmallData-1669346525.9906075/"
TOKENIZED_DATA_PATH = "/raid/fredrikc/Non-Residual-GANN/DataManagers/WikitextDataset/PreTokenized/WikitextDataset-16384-64-ls-100-Train-100pct.pkl"
TOKENIZER_PATH = "/raid/fredrikc/Non-Residual-GANN/Tokenizers/WikitextTokenizer-16384-64-ls-100"
DEVICE = "cuda:0"
CHECKPOINT = "Current"
SEQUENCE_LENGTH = 64  # Of tokenized pretrained data.
VOCAB_LENGTH = 16384
VOCAB_AFTER_REDUCTION = 257
NUM_ALTERATIONS = 8
DATA_PATH_NAME = "./sampleNetData.pickle"
N_TEST_SAMPLES = 500


def main():
    print("  => LOADING SAMPLE NET")
    sampleNet = MODEL(VOCAB_AFTER_REDUCTION).to(DEVICE)

    print("  => LOADING OPTIMIZER")
    optimizer = torch.optim.Adam(sampleNet.parameters(), lr=LEARNING_RATE)

    print("  => PREPARE TRAINING DATA FOR CLASSIFIER")

    df_big = False
    df_small = False

    NUM_TRAIN_SHEETS = 10_000

    for i in tqdm.tqdm(range(1)):
        with open(f"train_data/train_big_100000_{i}.pkl", "rb") as f:
            probs1 = pickle.load(f)
            probs1 = probs1
        #probs1 = probs1.numpy()
        with open(f"train_data/train_small_100000_{i}.pkl", "rb") as g:
            probs0 = pickle.load(g)
            probs0 = probs0
        #probs0 = probs0.numpy()

        with open(f"train_data/indices_big_100000_{i}.pkl", "rb") as h:
            indices1 = pickle.load(h)
            indices1 = indices1

        with open(f"train_data/indices_small_100000_{i}.pkl", "rb") as k:
            indices0 = pickle.load(k)
            indices0 = indices0

        print("  => STARTING THE CORRECTION OF TRAINING DATA:")

        correct_probs0 = train_data_baseline(probs0, indices1, indices0)

        print(f"correct_probs0.shape: {correct_probs0.size()}")

        tmp_zeros = torch.zeros((NUM_TRAIN_SHEETS, 64, 1))
        probs1 = torch.cat((probs1, tmp_zeros), dim=-1)  # add column of zeros for 257th element

        print(f"bigprobs shape after adding 257: {probs1.size()}")

        probs1 = add_sum_as_last_element(probs1)  # add 1 - the sum of the first 256 elements in the 257th slot
        correct_probs0 = add_sum_as_last_element(correct_probs0)

        print(f" probs1 first row: {probs1[:1, :1]}")
        print(f" correct_probs0 random row: {correct_probs0[16:17, :1]}")

        probs1 = probs1.numpy()
        correct_probs0 = correct_probs0.numpy()

        df_tmp_big = prepare_training_data(probs1, num_sheets=NUM_TRAIN_SHEETS)
        df_tmp_small = prepare_training_data(correct_probs0, num_sheets=NUM_TRAIN_SHEETS)
        if type(df_big) is bool:
            df_big = df_tmp_big
        else:
            df_big = pd.concat([df_big, df_tmp_big], axis=0, ignore_index=True)

        if type(df_small) is bool:
            df_small = df_tmp_small
        else:
            df_small = pd.concat([df_small, df_tmp_small], axis=0, ignore_index=True)

    df_small.columns = ['target_text']
    df_big.columns = ['source_text']

    df_final = pd.concat([df_big, df_small], axis=1, ignore_index=False)

    print(df_final.head())

    print(f"df_final.shape[0]: {df_final.shape[0]}")

    np.random.seed(112)
    torch.manual_seed(0)

    df_train, df_val = np.split(df_final.sample(frac=1, random_state=42),
                                [int(.8 * len(df_final))])

    trainDataset = Dataset(df_train)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)

    print("  => PREPARING TEST DATA FOR CLASSIFIER")

    with open(f"train_data/train_big_100000_9.pkl", "rb") as f:
        probs1 = pickle.load(f)
        probs1 = probs1[:N_TEST_SAMPLES]
    with open(f"train_data/train_small_100000_9.pkl", "rb") as g:
        probs0 = pickle.load(g)
        probs0 = probs0[:N_TEST_SAMPLES]
    with open(f"train_data/indices_big_100000_9.pkl", "rb") as h:
        indices1 = pickle.load(h)
        indices1 = indices1[:N_TEST_SAMPLES]
    with open(f"train_data/indices_small_100000_9.pkl", "rb") as k:
        indices0 = pickle.load(k)
        indices0 = indices0[:N_TEST_SAMPLES]

    print("  => STARTING THE CORRECTION OF TRAINING DATA:")

    correct_probs0 = train_data_baseline(probs0, indices1, indices0)

    print(f"correct_probs0.shape: {correct_probs0.size()}")

    tmp_zeros = torch.zeros((N_TEST_SAMPLES, 64, 1))
    probs1 = torch.cat((probs1, tmp_zeros), dim=-1)  # add column of zeros for 257th element

    print(f"bigprobs shape after adding 257: {probs1.size()}")

    probs1 = add_sum_as_last_element(probs1)  # add 1 - the sum of the first 256 elements in the 257th slot
    correct_probs0 = add_sum_as_last_element(correct_probs0)

    probs1 = probs1.numpy()
    df_big = prepare_training_data(probs1, num_sheets=NUM_TRAIN_SHEETS)
    correct_probs0 = correct_probs0.numpy()
    df_small = prepare_training_data(correct_probs0, num_sheets=NUM_TRAIN_SHEETS)
    indices1 = indices1.numpy()
    df_big_indx = prepare_training_data(indices1, num_sheets=NUM_TRAIN_SHEETS)
    indices0 = indices0.numpy()
    df_small_indx = prepare_training_data(indices0, num_sheets=NUM_TRAIN_SHEETS)

    df_small.columns = ['target_text']
    df_big.columns = ['source_text']
    df_big_indx.columns = ['source_index']
    df_small_indx.columns = ['target_index']

    df_test = pd.concat([df_big, df_small, df_big_indx, df_small_indx], axis=1, ignore_index=False)

    print(df_test.head())

    print(f"n rows of test data: {df_test.shape[0]}")
    testDataset = DatasetTest(df_test)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

    print("  => TRAINING")
    for epoch in range(EPOCHS):
        first_iteration = True
        print("EPOCH :", (epoch+1), "of", EPOCHS)
        for bigprobs, smallprobs in tqdm.tqdm(trainLoader):
            optimizer.zero_grad()

            bigprobs = bigprobs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
            smallprobs = smallprobs.squeeze().to(DEVICE)
            #samples_indices = samples_indices.to(DEVICE)
            #probs_indices = probs_indices.to(DEVICE)

            pred_samples = sampleNet.forward(bigprobs)

            #unsort_samples = samples.gather(1, samples_indices.argsort(1)).to(DEVICE)
            #unsort_pred_samples = pred_samples.gather(1, probs_indices.argsort(1)).to(DEVICE)

            overlaps = torch.abs(smallprobs - pred_samples)
            over_loss = overlaps.mean() / BATCH_SIZE

            over_loss.backward()
            optimizer.step()

            if first_iteration and epoch == 0:
                print(f"First iteration, overlap loss: {float(over_loss)}")
                first_iteration = False

        print("Overlap loss: ", float(over_loss), "  ")

    print("\n\n\n\n")
    print("--------------------")
    print("----- TESTING ----- ")
    print("--------------------")
    sampleNet.eval()
    with torch.no_grad():
        trans_distances = np.zeros((N_TEST_SAMPLES, 64, 1))
        original_distances = np.zeros((N_TEST_SAMPLES, 64, 1))
        for i, (bigprobs, smallprobs, bigindices, smallindices) in enumerate(tqdm.tqdm(testLoader)):
            bigprobs = bigprobs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
            smallprobs = smallprobs.squeeze().to(DEVICE)
            smallindices = smallindices.to(DEVICE)
            bigindices = bigindices.to(DEVICE)

            pred_samples = sampleNet.forward(bigprobs)

            overlaps = torch.abs(smallprobs - pred_samples)
            over_loss = overlaps.sum() / BATCH_SIZE

            print(f"bigprobs shape, bigindices shape: {bigprobs.shape}, {bigindices.shape}")

            bigprobs = bigprobs[:, :, :-1]
            smallprobs = smallprobs[:, :, :-1]
            pred_samples = pred_samples[:, :, :-1]

            print("  => FILL UP DISTRIBUTIONS")
            filled_up_big_probs = fill_multiple_distributions(bigprobs.cpu().numpy(), bigindices.cpu().numpy())
            filled_up_pred_probs = fill_multiple_distributions(pred_samples.cpu().numpy(), bigindices.cpu().numpy())
            print("  => DONE FILLING UP DISTRIBUTIONS")

            dist_trans_tmp, dist_big_tmp = get_distances(filled_up_pred_probs, filled_up_big_probs,
                                                         smallprobs.cpu().numpy(), bigindices.cpu().numpy())

            dist_trans_tmp = np.expand_dims(dist_trans_tmp, -1)
            dist_big_tmp = np.expand_dims(dist_big_tmp, -1)

            trans_distances[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = dist_trans_tmp
            original_distances[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = dist_big_tmp

            # unsort_samples = samples.gather(1, samples_indices.argsort(1)).to(DEVICE)
            # unsort_pred_samples = pred_samples.gather(1, probs_indices.argsort(1)).to(DEVICE)
            # word_differences = (unsort_samples - unsort_pred_samples).to(DEVICE)

        print("Overlap loss: ", float(over_loss), "  ")

        score = get_mean_distances(trans_distances, original_distances)

        print(f"final score: {score}")


# class DATASET(DatasetPytorch):
#     def __init__(self, x, y):
#         super(DATASET, self).__init__()
#         self.x = torch.FloatTensor(x)
#         self.y = torch.FloatTensor(y)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, index):
#         return self.x[index], self.y[index]


#@timeit
def train_data_baseline(smalltop, bigidx, smallidx):
    indx_tensor = torch.empty_like(smallidx)
    for i, column in enumerate(tqdm.tqdm(smallidx)):
        for j, row in enumerate(column):
            tmp = torch.isin(smallidx[i][j], bigidx[i][j])
            indx_tensor[i, j] = tmp

    print(f"index tensor size: {indx_tensor.size()}")
    print(f" index tensor first row: {indx_tensor[:1,:1]}")
    print(f" index tensor random row: {indx_tensor[16:17, :1]}")

    final_tensor = torch.zeros((smallidx.shape[0], smallidx.shape[1], smallidx.shape[2]+1))  # add column of zeros for 257th element
    for i, column in enumerate(tqdm.tqdm(indx_tensor)):
        for j, row in enumerate(column):
            for k, element in enumerate(row):
                if element == 1:
                    idx = torch.where(bigidx[i, j] == smallidx[i, j, k])[0]
                    final_tensor[i, j, idx] = smalltop[i, j, k].item()

    print(f"final_tensor: {final_tensor}")

    return final_tensor


def add_sum_as_last_element(probs):
    sums = torch.sum(probs, dim=-1)
    to_add = 1 - sums
    probs[:, :, -1] = to_add
    return probs


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.bigprobs = [text for text in df['source_text']]
        self.bigprobs = torch.FloatTensor(np.array(self.bigprobs))
        self.smallprobs = [text for text in df['target_text']]
        self.smallprobs = torch.FloatTensor(np.array(self.smallprobs))

    def __len__(self):
        return len(self.smallprobs)

    def get_bigprobs(self, idx):
        # Fetch a batch of raw input probs
        return self.bigprobs[idx]

    def get_smallprobs(self, idx):
        # Fetch a batch of raw input probs
        return self.smallprobs[idx]

    def __getitem__(self, idx):
        batch_probs_big = self.get_bigprobs(idx)
        batch_probs_small = self.get_smallprobs(idx)
        return batch_probs_big, batch_probs_small


class DatasetTest(torch.utils.data.Dataset):

    def __init__(self, df):
        self.bigprobs = [text for text in df['source_text']]
        self.bigprobs = torch.FloatTensor(np.array(self.bigprobs))
        self.smallprobs = [text for text in df['target_text']]
        self.smallprobs = torch.FloatTensor(np.array(self.smallprobs))
        self.bigindices = [indx for indx in df['source_index']]
        self.bigindices = torch.FloatTensor(np.array(self.bigindices))
        self.smallindices = [indx for indx in df['target_index']]
        self.smallindices = torch.FloatTensor(np.array(self.smallindices))

    def __len__(self):
        return len(self.smallprobs)

    def get_bigprobs(self, idx):
        # Fetch a batch of raw input probs
        return self.bigprobs[idx]

    def get_bigindices(self, idx):
        # Fetch a batch of raw input probs
        return self.bigindices[idx]

    def get_smallprobs(self, idx):
        # Fetch a batch of raw input probs
        return self.smallprobs[idx]

    def get_smallindices(self, idx):
        # Fetch a batch of raw input probs
        return self.smallindices[idx]

    def __getitem__(self, idx):
        batch_probs_big = self.get_bigprobs(idx)
        batch_probs_small = self.get_smallprobs(idx)
        batch_indices_big = self.get_bigindices(idx)
        batch_indices_small = self.get_smallindices(idx)
        return batch_probs_big, batch_probs_small, batch_indices_big, batch_indices_small


class MODEL(nn.Module):
    def __init__(self, size):
        super(MODEL, self).__init__()
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.3)
        self.l1 = torch.nn.Linear(size, 5000)
        self.l2 = torch.nn.Linear(5000, 10000)
        self.l3 = torch.nn.Linear(10000, 5000)
        self.l4 = torch.nn.Linear(5000, size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.dropout(x)
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.dropout(x)
        x = self.l4(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()
