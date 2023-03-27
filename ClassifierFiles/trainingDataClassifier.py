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
            #sorted_probs = sorted(samples, reverse=True)
            #sequence.append(sorted_probs[:256])
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


def downsampled_training_data(probs1, labels, num_sheets):
    formatted_data = []
    for i, sheets in enumerate(tqdm(probs1[:num_sheets])):
        sequence = []
        for j, samples in enumerate(sheets):
            sequence.append(samples)
        formatted_data.append(sequence)

    ignored = 0
    formatted_labels = []
    for i in range(num_sheets):
        if 2 in labels[i*64:(i+1)*64]:  #or 1 not in labels[i*64:(i+1)*64]
            del formatted_data[i-ignored]
            ignored += 1
            continue
        formatted_labels.append([])
        for j in range(64):
            formatted_labels[i-ignored].append(labels[j + ((i-ignored) * num_sheets)])

    df_dict = {
        "text": formatted_data,
        "label": formatted_labels
    }
    df = pd.DataFrame(df_dict)
    return df


if __name__ == "__main__":
    probs1 = pickle.load(open("../Data/probs_1.p", "rb"))
    probs1 = probs1.detach().numpy()
    labels = list(np.random.randint(0, 2, (32*64)))
    print(len(labels))
    df = prepare_training_data(probs1, labels, num_sheets=32)
    print("Done preparing training data!")
    print(df.head())
    print(df.shape[0])

    labels[3] = 2
    labels[69] = 2
    #labels[64:64*2] = [0 for i in range(64)]
    df_2 = prepare_training_data(probs1, labels, num_sheets=32)
    print(df_2.head())
    print(df_2.shape[0])

