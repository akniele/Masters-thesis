import sys
sys.path.insert(1, '../Non-Residual-GANN')

import pandas as pd
import pickle
import torch
import torch.nn as nn
import tqdm
import numpy as np
from collections import defaultdict
from ClassifierFiles.trainingDataClassifier import prepare_training_data
from Transformation.fill_up_distributions import fill_multiple_distributions
from Transformation.transformation import get_distances, get_mean_distances
import ClassifierFiles.plot_acc_and_loss as plot_acc_and_loss

"""
This baseline model maps transforms the probability distributions of one model to
more closely resemble the probability distributions of a different model.
"""


BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda:0"
SEQUENCE_LENGTH = 64
VOCAB_LENGTH = 16384
VOCAB_AFTER_REDUCTION = 257
N_TEST_SAMPLES = 500
LOSS_PLOT_NAME = f"loss_plot.png"
NUM_TRAIN_SHEETS = 10_000


def main():
    print("  => LOADING MODEL")
    model = MODEL(VOCAB_AFTER_REDUCTION).to(DEVICE)

    print("  => LOADING OPTIMIZER")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("  => PREPARE TRAINING DATA FOR CLASSIFIER")

    df_big = False
    df_small = False

    for i in tqdm.tqdm(range(9)):
        with open(f"train_data/big_10000_{i}sep.pkl", "rb") as f:  # This data is sorted by the probs of the big model
            probs1 = pickle.load(f)
        with open(f"train_data/small_10000_{i}sep.pkl", "rb") as g:
            probs0 = pickle.load(g)

        tmp_zeros = np.zeros((NUM_TRAIN_SHEETS, 64, 1))
        probs1 = np.concatenate((probs1, tmp_zeros), axis=-1)  # add column of zeros for 257th element

        probs0 = np.concatenate((probs0, tmp_zeros), axis=-1)  # add column of zeros for 257th element

        probs1 = add_sum_as_last_element(probs1)  # add 1 - the sum of the first 256 elements in the 257th slot
        probs0 = add_sum_as_last_element(probs0)

        df_tmp_big = prepare_training_data(probs1, num_sheets=NUM_TRAIN_SHEETS)
        df_tmp_small = prepare_training_data(probs0, num_sheets=NUM_TRAIN_SHEETS)

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

    np.random.seed(112)
    torch.manual_seed(0)

    df_train, df_val = np.split(df_final.sample(frac=1, random_state=42),
                                [int(.8 * len(df_final))])

    trainDataset = Dataset(df_train)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE)
    valDataset = Dataset(df_val)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size=BATCH_SIZE)

    print("  => PREPARING TEST DATA FOR CLASSIFIER")

    with open(f"train_data/train_big_10000_9.pkl", "rb") as f:  # the big and small probs are sorted separately
        probs1 = pickle.load(f)
        probs1 = probs1[:N_TEST_SAMPLES]
    with open(f"train_data/train_small_10000_9.pkl", "rb") as g:
        probs0 = pickle.load(g)
        probs0 = probs0[:N_TEST_SAMPLES]
    with open(f"train_data/indices_big_10000_9.pkl", "rb") as h:
        indices = pickle.load(h)
        indices = indices[:N_TEST_SAMPLES]
    with open(f"train_data/indices_small_10000_9.pkl", "rb") as k:
        indices0 = pickle.load(k)
        indices0 = indices0[:N_TEST_SAMPLES]

    with open(f"train_data/small_10000_9sep.pkl", "rb") as g:  # the small probs are sorted by the big probs
        test_small = pickle.load(g)
        test_small = test_small[:N_TEST_SAMPLES]

    tmp_zeros = np.zeros((N_TEST_SAMPLES, 64, 1))
    probs1 = np.concatenate((probs1, tmp_zeros), axis=-1)  # add column of zeros for 257th element
    probs0 = np.concatenate((probs0, tmp_zeros), axis=-1)  # add column of zeros for 257th element
    test_small = np.concatenate((test_small, tmp_zeros), axis=-1) # add column of zeros for 257th element

    probs1 = add_sum_as_last_element(probs1)  # add 1 - the sum of the first 256 elements in the 257th slot
    probs0 = add_sum_as_last_element(probs0)
    test_small = add_sum_as_last_element(test_small)

    df_big = prepare_training_data(probs1, num_sheets=NUM_TRAIN_SHEETS)
    df_small = prepare_training_data(probs0, num_sheets=NUM_TRAIN_SHEETS)
    indices = indices.numpy()
    df_indx = prepare_training_data(indices, num_sheets=NUM_TRAIN_SHEETS)
    indices0 = indices0.numpy()
    df_small_indx = prepare_training_data(indices0, num_sheets=NUM_TRAIN_SHEETS)

    df_test_small = prepare_training_data(test_small, num_sheets=NUM_TRAIN_SHEETS)

    df_small.columns = ['target_text']  # we need this for calculating the final score
                                        # (need full 256 probs, sorted independently)
    df_big.columns = ['source_text']
    df_indx.columns = ['source_index']
    df_small_indx.columns = ['target_index']
    df_test_small.columns = ['test_text']  # we need this data to calculate the test loss
                                           # (probs need to refer to same token)

    df_test = pd.concat([df_big, df_small, df_indx, df_small_indx, df_test_small], axis=1, ignore_index=False)

    testDataset = DatasetTest(df_test)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE)

    print("  => TRAINING")
    for epoch in range(EPOCHS):
        model.train()

        total_loss_train = 0
        for bigprobs, smallprobs in tqdm.tqdm(trainLoader):
            optimizer.zero_grad()

            bigprobs = bigprobs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
            smallprobs = smallprobs.squeeze().to(DEVICE)

            pred_samples = model.forward(bigprobs)

            overlaps = torch.abs(smallprobs - pred_samples)
            over_loss = overlaps.sum()

            total_loss_train += over_loss.item()

            over_loss = overlaps.mean()

            over_loss.backward()
            optimizer.step()

        model.train_loss.append(total_loss_train / (len(df_train)*64*VOCAB_AFTER_REDUCTION))

        model.eval()
        total_loss_val = 0
        with torch.no_grad():
            for bigprobs, smallprobs in valLoader:
                bigprobs = bigprobs.squeeze().to(DEVICE)
                smallprobs = smallprobs.squeeze().to(DEVICE)

                pred_samples = model.forward(bigprobs)

                overlaps = torch.abs(smallprobs - pred_samples)
                over_loss = overlaps.sum()

                total_loss_val += over_loss.item()

            model.val_loss.append(total_loss_val / (len(df_val)*64*VOCAB_AFTER_REDUCTION))

        print(
            f'Epochs: {epoch + 1} | Train Loss: {(total_loss_train / (len(df_train)*64*VOCAB_AFTER_REDUCTION)): .4f} | '
            f'Val Loss: {(total_loss_val / (len(df_val)*64*VOCAB_AFTER_REDUCTION)): .4f}')

    loss_dict = defaultdict()
    loss_dict["train_loss"] = model.train_loss
    loss_dict["val_loss"] = model.val_loss

    plot_acc_and_loss.loss_plot(loss_dict, LOSS_PLOT_NAME)

    print("\n\n\n\n")
    print("--------------------")
    print("----- TESTING ----- ")
    print("--------------------")
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        trans_distances = np.zeros((N_TEST_SAMPLES, 64, 1))
        original_distances = np.zeros((N_TEST_SAMPLES, 64, 1))
        for i, (bigprobs, smallprobs, bigindices, smallindices, test_small) in enumerate(tqdm.tqdm(testLoader)):
            bigprobs = bigprobs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
            smallprobs = smallprobs.squeeze().to(DEVICE)
            smallindices = smallindices.to(DEVICE)
            bigindices = bigindices.to(DEVICE)
            test_small = test_small.to(DEVICE)

            pred_samples = model.forward(bigprobs)

            overlaps = torch.abs(test_small - pred_samples)
            over_loss = overlaps.sum()

            total_test_loss += over_loss.item()

            bigprobs = bigprobs[:, :, :-1]
            smallprobs = smallprobs[:, :, :-1]
            pred_samples = pred_samples[:, :, :-1]

            filled_up_big_probs = fill_multiple_distributions(bigprobs.cpu().numpy(), bigindices.cpu().numpy())
            filled_up_pred_probs = fill_multiple_distributions(pred_samples.cpu().numpy(), bigindices.cpu().numpy())

            smallprobs = smallprobs.cpu().numpy()
            smallindices = smallindices.cpu().numpy()

            dist_trans_tmp, dist_big_tmp = get_distances(filled_up_pred_probs, filled_up_big_probs,
                                                         smallprobs, smallindices)

            dist_trans_tmp = np.expand_dims(dist_trans_tmp, -1)
            dist_big_tmp = np.expand_dims(dist_big_tmp, -1)

            trans_distances[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = dist_trans_tmp
            original_distances[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = dist_big_tmp

        print(f"Test loss: {(total_test_loss / (len(df_test)*64*VOCAB_AFTER_REDUCTION))}")

        score = get_mean_distances(trans_distances, original_distances)

        print(f"final score: {score}")


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
    sums = np.sum(probs, axis=-1)
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
        self.bigprobs = np.array([text for text in df['source_text']])
        self.bigprobs = torch.FloatTensor(np.array(self.bigprobs))
        self.smallprobs = np.array([text for text in df['target_text']])
        self.smallprobs = torch.FloatTensor(np.array(self.smallprobs))
        self.bigindices = np.array([indx for indx in df['source_index']])
        self.bigindices = torch.FloatTensor(np.array(self.bigindices))
        self.smallindices = [indx for indx in df['target_index']]
        self.smallindices = torch.FloatTensor(np.array(self.smallindices))
        self.testsmall = np.array([text for text in df['test_text']])
        self.testsmall = torch.FloatTensor(np.array(self.testsmall))

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

    def get_testsmall(self, idx):
        # Fetch a batch of raw input probs
        return self.testsmall[idx]

    def __getitem__(self, idx):
        batch_probs_big = self.get_bigprobs(idx)
        batch_probs_small = self.get_smallprobs(idx)
        batch_indices_big = self.get_bigindices(idx)
        batch_indices_small = self.get_smallindices(idx)
        batch_test_small = self.get_testsmall(idx)
        return batch_probs_big, batch_probs_small, batch_indices_big, batch_indices_small, batch_test_small


class MODEL(nn.Module):
    def __init__(self, size):
        super(MODEL, self).__init__()
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.l1 = torch.nn.Linear(size, 5000)
        self.l2 = torch.nn.Linear(5000, 10000)
        self.l3 = torch.nn.Linear(10000, 5000)
        self.l4 = torch.nn.Linear(5000, size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.val_loss = list()
        self.train_loss = list()

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
