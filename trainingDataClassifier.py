"""Prepare training data for classifier"""
import pandas as pd


def prepare_training_data(probs1, labels):
    formatted_data = []
    for i, sheets in enumerate(probs1[:10]):
        sequence = []
        for j, samples in enumerate(sheets):
            sorted_probs = sorted(samples, reverse=True)
            sequence.append(sorted_probs[:256])
        formatted_data.append(sequence)

    formatted_labels = []
    for i in range(10):
        formatted_labels.append([])
        for j in range(64):
            formatted_labels[i].append(labels[j + (i*32)])

    df_dict = {
      "text": formatted_data,
      "label": formatted_labels
    }
    df = pd.DataFrame(df_dict)
    return df
