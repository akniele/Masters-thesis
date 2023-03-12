import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = [label for label in df['label']]
        self.labels = torch.FloatTensor(np.array(self.labels))
        self.probs = [text for text in df['text']]
        self.probs = torch.FloatTensor(np.array(self.probs))

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_probs(self, idx):
        # Fetch a batch of raw input probs
        return self.probs[idx]

    def get_ids(self, idx):
        # Fetch a batch of ids
        return self.ids[idx]

    def __getitem__(self, idx):
        batch_probs = self.get_probs(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_probs, batch_y
