"""Prepare training data for classifier"""
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle


def prepare_training_data(probs1, num_sheets, labels=None):
    formatted_data = []
    for i, sheets in enumerate(tqdm(probs1[:num_sheets])):
        sequence = []
        for j, samples in enumerate(sheets):
            sequence.append(samples)
        formatted_data.append(sequence)

    if labels != None:
        formatted_labels = []
        for i in range(num_sheets):
            formatted_labels.append([])
            for j in range(64):
                formatted_labels[i].append(next(labels))
        df_dict = {
          "text": formatted_data,
          "label": formatted_labels
        }
    else:
        df_dict = {
            "text": formatted_data
        }

    df = pd.DataFrame(df_dict)
    return df


if __name__ == "__main__":
    probs1 = pickle.load(open("../Data/probs_1.p", "rb"))
    probs1 = probs1.detach().numpy()
    labels = list(np.random.randint(0, 2, (32*64)))
    print(len(labels))
    df = prepare_training_data(probs1, num_sheets=32, labels=labels)
    print("Done preparing training data!")
    print(df.head())
    print(df.shape[0])

    labels[3] = 2
    labels[69] = 2
    df_2 = prepare_training_data(probs1, num_sheets=32, labels=labels)
    print(df_2.head())
    print(df_2.shape[0])

