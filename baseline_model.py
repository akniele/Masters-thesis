import sys
sys.path.insert(1, '../Non-Residual-GANN')

import pandas as pd
import pickle
import torch
import torch.nn as nn
import tqdm
import numpy as np
from scipy.stats import entropy
from collections import defaultdict
from ClassifierFiles.trainingDataClassifier import prepare_training_data
from Transformation.fill_up_distributions import fill_multiple_distributions
from Transformation.transformation import get_distances, get_mean_distances
import ClassifierFiles.plot_acc_and_loss as plot_acc_and_loss
from Transformation.histogram_of_distances import difference_histogram

"""
This baseline model maps transforms the probability distributions of one model to
more closely resemble the probability distributions of a different model.
"""

DEVICE = "cuda:0"
SEQUENCE_LENGTH = 64
VOCAB_LENGTH = 16384
VOCAB_AFTER_REDUCTION = 257
NUM_TRAIN_SHEETS = 10_000


def baseline(n_test_samples, batch_size, epochs, lr, filename):
    print("  => LOADING MODEL")
    model = MODEL(VOCAB_AFTER_REDUCTION).to(DEVICE)

    early_stopper = EarlyStopper(patience=5)

    print("  => LOADING OPTIMIZER")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("  => LOADING LEARNING RATE SCHEDULER")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

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
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size)
    valDataset = Dataset(df_val)
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batch_size)

    print("  => PREPARING TEST DATA FOR CLASSIFIER")

    with open(f"train_data/big_10000_9.pkl", "rb") as f:  # the big and small probs are sorted separately
        probs1 = pickle.load(f)
        probs1 = probs1[:n_test_samples]
    with open(f"train_data/small_10000_9.pkl", "rb") as g:
        probs0 = pickle.load(g)
        probs0 = probs0[:n_test_samples]
    with open(f"train_data/indices_big_10000_9.pkl", "rb") as h:
        indices = pickle.load(h)
        indices = indices[:n_test_samples]
    with open(f"train_data/indices_small_10000_9.pkl", "rb") as k:
        indices0 = pickle.load(k)
        indices0 = indices0[:n_test_samples]

    with open(f"train_data/small_10000_9sep.pkl", "rb") as g:  # the small probs are sorted by the big probs
        test_small = pickle.load(g)
        test_small = test_small[:n_test_samples]

    tmp_zeros = np.zeros((n_test_samples, 64, 1))
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
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size)

    epoch_last_saved_model = 0

    print("  => TRAINING")
    for epoch in range(epochs):
        model.train()

        total_loss_train = 0
        for bigprobs, smallprobs in tqdm.tqdm(trainLoader):
            optimizer.zero_grad()

            bigprobs = bigprobs.squeeze().to(DEVICE)  # [batch_size, SEQUENCE_LENGTH, VOCAB_LENGTH]
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

        if epoch > 20 and model.val_acc[-1] > model.val_acc[-2]:  # whenever val accuracy has increased, save model
            torch.save(model.state_dict(), f'models/classifier_{filename}.pt')
            epoch_last_saved_model = epoch + 1

        with open(f"logfiles/{filename}.txt", "a") as logfile:
            logfile.write(f'Epochs: {epoch + 1} | Train Loss: '
                          f'{(total_loss_train / (len(df_train)*64*VOCAB_AFTER_REDUCTION)): .4f} | '
                          f'Val Loss: {(total_loss_val / (len(df_val)*64*VOCAB_AFTER_REDUCTION)): .4f}\n\n')

        if early_stopper.early_stop(total_loss_val / (len(df_val)*64*VOCAB_AFTER_REDUCTION)):
            with open(f"logfiles/{filename}_classifier.txt", "a") as logfile:
                logfile.write(f'Model last saved at epoch: {epoch_last_saved_model}.\n')
            break

        scheduler.step((total_loss_val / (len(df_val)*64*VOCAB_AFTER_REDUCTION)))

    loss_dict = defaultdict()
    loss_dict["train_loss"] = model.train_loss
    loss_dict["val_loss"] = model.val_loss

    plot_acc_and_loss.loss_plot(loss_dict, f"loss_{filename}")

    torch.save(model.state_dict(), f'models/baseline_{lr}_{epochs}_{batch_size}.pt')

    print("\n\n\n\n")
    print("--------------------")
    print("----- TESTING ----- ")
    print("--------------------")
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        trans_distances = np.zeros((n_test_samples, 64, 1))
        original_distances = np.zeros((n_test_samples, 64, 1))

        entropy_big = np.zeros((n_test_samples, 64, 1))
        entropy_small = np.zeros((n_test_samples, 64, 1))
        entropy_trans = np.zeros((n_test_samples, 64, 1))

        index_highest_prob_trans = np.zeros((n_test_samples, 64, 1))
        index_highest_prob_big = np.zeros((n_test_samples, 64, 1))
        index_highest_prob_small = np.zeros((n_test_samples, 64, 1))

        for i, (bigprobs, smallprobs, bigindices, smallindices, test_small) in enumerate(tqdm.tqdm(testLoader)):
            bigprobs = bigprobs.squeeze().to(DEVICE)  # [batch_size, SEQUENCE_LENGTH, VOCAB_LENGTH]
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
            filled_up_small_probs = fill_multiple_distributions(smallprobs, smallindices)

            high_idx_big = np.argmax(filled_up_big_probs, axis=-1)
            high_idx_trans = np.argmax(filled_up_pred_probs, axis=-1)
            high_idx_small = np.argmax(filled_up_small_probs, axis=-1)

            index_highest_prob_big[i*batch_size:(i+1)*batch_size] = np.expand_dims(high_idx_big, axis=-1)
            index_highest_prob_trans[i * batch_size:(i + 1) * batch_size] = np.expand_dims(high_idx_trans, axis=-1)
            index_highest_prob_small[i * batch_size:(i + 1) * batch_size] = np.expand_dims(high_idx_small, axis=-1)

            epsilon = 10e-10
            filled_up_pred_probs += epsilon
            filled_up_big_probs += epsilon
            filled_up_small_probs += epsilon

            small_entropy = entropy(filled_up_small_probs, axis=-1)
            transformed_entropy = entropy(filled_up_pred_probs, axis=-1)
            original_entropy = entropy(filled_up_big_probs, axis=-1)

            entropy_small[i*batch_size:(i+1)*batch_size] = np.expand_dims(small_entropy, -1)
            entropy_big[i*batch_size:(i+1)*batch_size] = np.expand_dims(original_entropy, -1)
            entropy_trans[i*batch_size:(i+1)*batch_size] = np.expand_dims(transformed_entropy, -1)

            dist_trans_tmp, dist_big_tmp = get_distances(filled_up_pred_probs, filled_up_big_probs,
                                                         filled_up_small_probs)

            dist_trans_tmp = np.expand_dims(dist_trans_tmp, -1)
            dist_big_tmp = np.expand_dims(dist_big_tmp, -1)

            trans_distances[i*batch_size:(i+1)*batch_size] = dist_trans_tmp
            original_distances[i*batch_size:(i+1)*batch_size] = dist_big_tmp

        with open(f"logfiles/{filename}.txt", "a") as logfile:
            logfile.write(f"Test loss: {(total_test_loss / (len(df_test)*64*VOCAB_AFTER_REDUCTION))}\n")

        accuracy_trans_tmp = index_highest_prob_trans == index_highest_prob_small
        accuracy_orig_tmp = index_highest_prob_big == index_highest_prob_small

        accuracy_trans = np.count_nonzero(accuracy_trans_tmp) / (n_test_samples * 64)
        accuracy_orig = np.count_nonzero(accuracy_orig_tmp) / (n_test_samples * 64)

        small_entropy = np.mean(entropy_small)
        transformed_entropy = np.mean(entropy_trans)
        original_entropy = np.mean(entropy_big)

        with open(f"logfiles/{filename}.txt", "a") as logfile:
            logfile.write(f"mean entropy of transformed distributions: {transformed_entropy}\n"
                          f"mean entropy of original big distributions: {original_entropy}\n"
                          f"mean entropy of original small distributions: {small_entropy}\n"
                          f"Percentage of true tokens that had the highest probability in the original "
                          f"distributions: {accuracy_orig}%\n"
                          f"Percentage of true tokens that had the highest probability in the transformed "
                          f"distributions: {accuracy_trans}%\n")

        difference_histogram(trans_distances, original_distances, filename)

        score_mean, score_std = get_mean_distances(trans_distances, original_distances, filename)

        with open(f"logfiles/{filename}.txt", "a") as f:
            f.write(f"Difference in mean Weighted Manhattan Distance: {score_mean}\n"
                    f"Difference in standard deviation of Weighted Manhattan Distances: {score_std}\n")


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
        self.l3 = torch.nn.Linear(10000, 10000)
        self.l4 = torch.nn.Linear(10000, 10000)
        self.l5 = torch.nn.Linear(10000, 5000)
        self.l6 = torch.nn.Linear(5000, size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.val_loss = list()
        self.train_loss = list()

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.dropout(x)
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        x = self.activation(self.l5(x))
        x = self.dropout(x)
        x = self.l6(x)
        x = self.softmax(x)
        return x


# taken from here: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
