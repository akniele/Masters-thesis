from transformers import BertModel
from torch import nn
from transformers import BertConfig
import torch
import numpy as np


class BertClassifier(nn.Module):

    def __init__(self, num_classes=3):  # number of classes is arbitrary right now!

        super(BertClassifier, self).__init__()

        self.embSize = 768  # the embedding size (hidden size) has to be a multiple of the number of attention heads

        self.input_linear = nn.Linear(256, self.embSize)
        conf = BertConfig(num_hidden_layers=12, vocab_size=64, hidden_size=self.embSize, num_attention_heads=12,
                          hidden_dropout_prob=0.2)
        self.bert = BertModel(conf)
        print(f"bert embeddings:\n{self.bert.embeddings}")
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.4, inplace=False)
        self.linear = nn.Linear(self.embSize, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.train_acc = list()
        self.train_loss = list()
        self.val_acc = list()
        self.val_loss = list()

    def forward(self, inputs_embeds):
        right_shape = self.input_linear(inputs_embeds)
        output, _ = self.bert(inputs_embeds=right_shape, return_dict=False)
        relu_output = self.relu(output)
        dropout_output = self.dropout(relu_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.logsoftmax(linear_output)
        return final_layer


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


if __name__ == "__main__":
    input_arr = np.random.rand(1, 64, 256)
    input_arr = torch.Tensor(input_arr)
