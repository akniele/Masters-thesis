from datasets import load_dataset, Dataset
from SamplingComparissons import CompareModels
from Utils import loadGenerator
import pandas as pd
import pickle
from GANN.SamplingStrategies import SamplingTechniques
import torch
import tqdm
from torch.utils.data import Dataset as DatasetPytorch
import numpy as np

BIG_MODEL_PATH = "/home/ubuntu/Non-Residual-GANN/Models/MLE+VH8-BIG-BIG-A8-R2-1670318134.3765588/"
SMALL_MODEL_PATH = "/home/ubuntu/Non-Residual-GANN/Models/MLE+VH2-Mini-BIG-A8-R2-1670318134.2979872"
TOKENIZED_DATA_PATH = "../pipeline/train_data/WikitextDataset-16384-64-ls-100-Train-10pct.pkl"
TOKENIZER_PATH = "/home/ubuntu/Non-Residual-GANN/Tokenizers/WikitextTokenizer-16384-64-ls-100"
DEVICE = "cuda:0"
CHECKPOINT = "Epoch-8"
SEQUENCE_LENGTH = 64  # Of tokenized pretrained data.
VOCAB_LENGTH = 16384
NUM_ALTERATIONS = 8
NUM_SAMPLES = 1000  # should be divisible by 1000

SAVE = False
TRUNCATE = False


def generateData(tokenized_data_path=TOKENIZED_DATA_PATH, sequence_length=SEQUENCE_LENGTH,
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
    new_data = [data[i] for i in range(len(data)) if len(data[i]) == sequence_length][:10000]
    data = Dataset.from_pandas(pd.DataFrame({"data": new_data}))
    data.set_format("torch")
    loader = torch.utils.data.DataLoader(data["data"], batch_size=1)

    #big_probabilities = np.zeros((10000, sequence_length, VOCAB_LENGTH))
    #small_probabilities = np.zeros((10000, sequence_length, VOCAB_LENGTH))
    big_probabilities = []
    small_probabilities = []
    big_indices = []
    small_indices = []

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
        inputIDs = (torch.tensor(example).to(DEVICE))  # size [batch_size, SEQUENCE_LENGTH, VOCAB_LENGTH]
        smallProb, bigProb = getData(inputIDs)

        if index == num_samples:
            break
        index += 1

        if save:
            smallProb = smallProb.squeeze(0)
            bigProb = bigProb.squeeze(0)
            big_probabilities[index % 1000] = bigProb.detach().cpu().numpy()
            small_probabilities[index % 1000] = smallProb.detach().cpu().numpy()
            if index % 1000 == 0:
                print(f"  => LOADING BIG PROBS {index//1000} TO FILE")
                with open(f"../pipeline/train_data/train_big_{index//1000}.npy", "wb") as f:
                    np.save(f, big_probabilities)
                big_probabilities = np.zeros((1000, sequence_length, VOCAB_LENGTH))
                print(f"  => LOADING SMALL PROBS {index//1000} TO FILE")
                with open(f"../pipeline/train_data/train_small_{index//1000}.npy", "wb") as g:
                    np.save(g, small_probabilities)
                small_probabilities = np.zeros((1000, sequence_length, VOCAB_LENGTH))

        elif not save and not truncate:

            small_probabilities.append(smallProb.detach().cpu().numpy())
            big_probabilities.append(bigProb.detach().cpu().numpy())
            #small_probabilities[index] = smallProb.detach().cpu().numpy()
            #big_probabilities[index] = bigProb.detach().cpu().numpy()

            prob_dict = dict()
            prob_dict["small_probs"] = small_probabilities
            prob_dict["big_probs"] = big_probabilities

        elif not save and truncate:
            small_probs = smallProb.detach().cpu().numpy()
            big_probs = bigProb.detach().cpu().numpy()
            small_ordered, small_indices = torch.sort(small_probs, descending=True)
            big_ordered, big_indices = torch.sort(big_probs, descending=True)

            small_ordered, small_indx = small_ordered[:, :, :TOPK], small_indices[:, :, :TOPK]
            big_ordered, big_indx = big_ordered[:, :, :TOPK], big_indices[:, :, :TOPK]

            small_probabilities.append(small_ordered)
            big_probabilities.append(big_ordered)
            small_indices.append(small_indx)
            big_indices.append(big_indx)

            prob_dict = dict()
            prob_dict["small_probs"] = small_probabilities
            prob_dict["big_probs"] = big_probabilities
            prob_dict["small_indx"] = small_indices
            prob_dict["big_indx"] = big_indices

    return prob_dict


if __name__ == "__main__":
    print("  => GENERATING ALL DATA")
    if SAVE:
        generateData(TOKENIZED_DATA_PATH, SEQUENCE_LENGTH, NUM_SAMPLES, truncate=TRUNCATE, topk=256, save=SAVE)
    else:
        small_probs, big_probs = generateData(TOKENIZED_DATA_PATH, SEQUENCE_LENGTH, NUM_SAMPLES, truncate=TRUNCATE, save=SAVE)
        print(small_probs.shape)
        print(type(small_probs))
        print(small_probs[0][0])
        print(len(small_probs[0][0]))
