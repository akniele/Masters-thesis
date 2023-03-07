import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from SamplingComparissons.AlterationProbabilityOverlap import generateAlterationProbabilityDistribution
from SamplingComparissons import CompareModels
from Utils import loadGenerator
import pandas as pd
import pickle
from GANN.SamplingStrategies import SamplingTechniques
import torch
import torch.nn as nn
import tqdm
import sys
import os
from functools import partialmethod
import json
from torch.utils.data import Dataset as DatasetPytorch
from sklearn.model_selection import train_test_split

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
DEVICE = "cuda:5"
CHECKPOINT = "Current"
SEQUENCE_LENGTH = 64  # Of tokenized pretrained data.
VOCAB_LENGTH = 16384
VOCAB_AFTER_REDUCTION = 128
TEST_DATA_SIZE = 0.1
NUM_ALTERATIONS = 8
DATA_PATH_NAME = "./sampleNetData.pickle"

#TODO
# MAP WORDS TO LOSS


def main():
    print("  => LOADING SAMPLE NET")
    sampleNet = MODEL(VOCAB_AFTER_REDUCTION).to(DEVICE)

    print("  => LOADING OPTIMIZER")
    optimizer = torch.optim.Adam(sampleNet.parameters(), lr=LEARNING_RATE)

    # print("  => GENERATING ALL DATA")
    # generateData(TOKENIZED_DATA_PATH, SEQUENCE_LENGTH, OUTPUT_NAME="sampleNetData.pickle")

    print("  => LOADING DATA")
    with open(DATA_PATH_NAME, "rb") as f:
        data = pickle.load(f)  # DATA = [{"probabilities" : [SEQUENCE_LENGTH, VOCAB_LENGTH], "samples" : [...]}, ...]

    inputs = []
    outputs = []
    for example in data:
        inputs.append([example["probabilities"], example["probabilities_indices"]])
        outputs.append([example["samples"], example["samples_indices"]])

    print("  => CREATING LOADERS")
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.1)

    trainDataset = DATASET(X_train, Y_train)
    testDataset = DATASET(X_test, Y_test)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

    print("  => TRAINING")
    for epoch in range(EPOCHS):
        print("EPOCH :", (epoch+1), "of", EPOCHS)
        for probs, probs_indices, samples, samples_indices in tqdm.tqdm(trainLoader):
            optimizer.zero_grad()

            probs = probs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
            samples = samples.squeeze().to(DEVICE)
            samples_indices = samples_indices.to(DEVICE)
            probs_indices = probs_indices.to(DEVICE)

            pred_samples = sampleNet.forward(probs)

            unsort_samples = samples.gather(1, samples_indices.argsort(1)).to(DEVICE)
            unsort_pred_samples = pred_samples.gather(1, probs_indices.argsort(1)).to(DEVICE)

            overlaps = torch.abs(unsort_samples - unsort_pred_samples)
            over_loss = overlaps.mean() / BATCH_SIZE

            over_loss.backward()
            optimizer.step()

            print("Overlap loss: ", float(over_loss), "  ")

    print("\n\n\n\n")
    print("--------------------")
    print("----- TESTING ----- ")
    print("--------------------")
    sampleNet.eval()
    for probs, probs_indices, samples, samples_indices in tqdm.tqdm(testLoader):
        probs = probs.squeeze().to(DEVICE)  # [BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_LENGTH]
        samples = samples.squeeze().to(DEVICE)
        samples_indices = samples_indices.to(DEVICE)
        probs_indices = probs_indices.to(DEVICE)

        pred_samples = sampleNet.forward(probs)

        overlaps = torch.abs(samples - pred_samples)
        over_loss = overlaps.sum() / BATCH_SIZE

        unsort_samples = samples.gather(1, samples_indices.argsort(1)).to(DEVICE)
        unsort_pred_samples = pred_samples.gather(1, probs_indices.argsort(1)).to(DEVICE)
        word_differences = (unsort_samples - unsort_pred_samples).to(DEVICE)

        print("Overlap loss: ", float(over_loss), "  ")


class DATASET(DatasetPytorch):
    def __init__(self, x, y):
        super(DATASET, self).__init__()
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index][0], self.x[index][1], self.y[index][0], self.y[index][1]


class MODEL(nn.Module):
    def __init__(self, size):
        super(MODEL, self).__init__()
        self.activation = torch.nn.ReLU()
        self.l1 = torch.nn.Linear(size, 10000)
        self.l2 = torch.nn.Linear(10000, 30000)
        self.l3 = torch.nn.Linear(30000, 10000)
        self.l4 = torch.nn.Linear(10000, size)

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.l4(x)
        return x


def generateData(TOKENIZED_DATA_PATH, SEQUENCE_LENGTH, OUTPUT_NAME="sampleNetData.json"):
    print("  => LOADING BIG MODEL")
    bigModel = loadGenerator(BIG_MODEL_PATH, CHECKPOINT).to(DEVICE)
    bigModel.eval()

    print("  => LOADING SMALL MODEL")
    smallModel = loadGenerator(SMALL_MODEL_PATH, CHECKPOINT).to(DEVICE)
    smallModel.eval()

    print("  => LOADING PRE TOKENIZED DATASET")
    with open(TOKENIZED_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    print("  => FORMATTING DATA")
    new_data = [data[i] for i in range(len(data)) if len(data[i]) == SEQUENCE_LENGTH]
    data = Dataset.from_pandas(pd.DataFrame({"data": new_data}))
    data.set_format("torch")
    loader = torch.utils.data.DataLoader(data["data"], batch_size=1)

    def getData(InputIDs):
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

        bigProbOrdered, bigIndexes = torch.sort(bigProb, descending=True)
        smallProbOrdered, smallIndexes = torch.sort(smallProb, descending=True)

        _, smallSamples = generateAlterationProbabilityDistribution(
            inputIDs=inputIDs,
            genProbs=smallProbOrdered,
            numAlterations=NUM_ALTERATIONS,
            samplingStrategy=SamplingTechniques.SAMPLE_NO_REPLACEMENT,
            samplingParams={},
            iterations=1000
        )

        # ----------- REDUCTIONS -------------
        def compressIndicies(indexes):
            """
            :param indexes: torch tensor of indexes with indexes[i] = k means that the value at i was originally at k.
            :return: If gaps in indexes will compress to range(0, len(indexes)) with the same order: Example (2, 7, 5) --> (0, 2, 1).
            """

            __sortedIndexes, indexes_of_sortedIndexes = torch.sort(indexes)
            indexes = indexes.to(DEVICE)
            indexes_of_sortedIndexes = indexes_of_sortedIndexes.to(DEVICE)

            newIndexes = torch.arange(0, indexes.shape[1]).to(DEVICE)
            waste = torch.zeros(0).to(DEVICE)
            newIndexes = torch.cat((waste, newIndexes.repeat(indexes.shape[0]))).reshape(indexes.shape[0],
                                                                                         indexes.shape[1]).to(
                DEVICE)
            newIndexes = newIndexes.gather(1, indexes_of_sortedIndexes.argsort(1)).to(DEVICE)

            return newIndexes

        # Reduce big probabilities and get new indexes.
        bigProbOrdered = bigProbOrdered.squeeze()
        bigIndexes = bigIndexes.squeeze()
        smallIndexes = smallIndexes.squeeze()
        smallSamples = smallSamples.squeeze()

        reduced_bigProbOrdered = bigProbOrdered[:, :VOCAB_AFTER_REDUCTION]  # Reduce big Prob
        reduced_bigIndexes = bigIndexes[:, :VOCAB_AFTER_REDUCTION]  # Reduce indexes
        compressedBigIndexes = compressIndicies(reduced_bigIndexes)  # Compress indexes

        unsorted_smallSamples = smallSamples.gather(1, smallIndexes.argsort(1)).to(DEVICE)  # Unsort smallSamples
        unsorted_reduced_smallSamples = unsorted_smallSamples.gather(1, reduced_bigIndexes).to(
            DEVICE)  # Pick out the positions that correspond with our new bigProbabilites.

        # Sort back reduced smallSamples
        # Find every element in bigProb in smallSamples, in smallSample order
        list_smallIndexes = smallIndexes.tolist()
        list_reduced_bigIndexes = reduced_bigIndexes.tolist()
        list_big_in_small = []  # Will be in decreasing order, since going from left to right in sorted (biggest to smallest).
        for i in range(len(list_smallIndexes)):
            example = list_smallIndexes[i]
            new_list = []
            for e in example:
                if (e in list_reduced_bigIndexes[i]):
                    new_list.append(e)
            list_big_in_small.append(new_list)
        big_in_small = torch.tensor(list_big_in_small).to(DEVICE)

        # Compress
        compressed_big_in_small = compressIndicies(big_in_small).to(DEVICE)

        reduced_smallSamples = unsorted_reduced_smallSamples.to(DEVICE)

        # WITH REDUCED WILL NOT BE ABLE TO KNOW WHICH POSITION CORRESPOND TO WHICH TOKEN.

        # ------------------------------------

        return reduced_bigProbOrdered, reduced_smallSamples, reduced_bigIndexes, compressed_big_in_small

    storage = []
    index = 0
    for example in tqdm.tqdm(loader):
        inputIDs = (torch.tensor(example).to(DEVICE))  # size [batch_size, SEQUENCE_LENGTH, VOCAB_LENGTH]
        bigProbOrdered, smallSamples, bigIndexes, smallIndexes = getData(inputIDs)
        storage.append({
            "probabilities": bigProbOrdered.detach().cpu().numpy().tolist(),
            "samples": smallSamples.detach().cpu().numpy().tolist(),
            "probabilities_indices": bigIndexes.detach().cpu().numpy().tolist(),
            "samples_indices": smallIndexes.detach().cpu().numpy().tolist()
        })

        if index == 100:
            break
        index += 1

    print("  => LOADING TO FILE")
    with open(OUTPUT_NAME, "wb") as f:
        pickle.dump(storage, f)


if __name__ == '__main__':
    main()
