import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.probs = [text for text in df['text']]
        self.probs = torch.FloatTensor(np.array(self.probs))

    def __len__(self):
        return len(self.probs)

    def get_probs(self, idx):
        # Fetch a batch of raw input probs
        return self.probs[idx]

    def get_ids(self, idx):
        # Fetch a batch of ids
        return self.ids[idx]

    def __getitem__(self, idx):
        batch_probs = self.get_probs(idx)
        return batch_probs
