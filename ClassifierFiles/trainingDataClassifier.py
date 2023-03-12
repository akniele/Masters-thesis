"""Prepare training data for classifier"""
import pandas as pd
from tqdm import tqdm


def prepare_training_data(probs1, labels, num_sheets):
    formatted_data = []
    for i, sheets in enumerate(tqdm(probs1[:num_sheets])):
        sequence = []
        for j, samples in enumerate(sheets):
            #sorted_probs = sorted(samples, reverse=True)
            #sequence.append(sorted_probs[:256])
            sequence.append(samples)
        formatted_data.append(sequence)

    formatted_labels = []
    for i in range(num_sheets):
        formatted_labels.append([])
        for j in range(64):
            formatted_labels[i].append(labels[j + (i*32)])

    df_dict = {
      "text": formatted_data,
      "label": formatted_labels
    }
    df = pd.DataFrame(df_dict)
    return df
