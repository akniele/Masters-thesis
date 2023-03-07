from transformers import BertModel
from torch import nn
from transformers import BertConfig


class BertClassifier(nn.Module):

    def __init__(self, num_classes=3):  # number of classes is arbitrary right now!

        super(BertClassifier, self).__init__()

        self.embSize = 256  # the embedding size (hidden size) has to be a multiple of the number of attention heads

        conf = BertConfig(num_hidden_layers=2, vocab_size=64, hidden_size=self.embSize, num_attention_heads=4)
        self.bert = BertModel(conf)
        print(f"bert embeddings:\n{self.bert.embeddings}")
        self.linear = nn.Linear(self.embSize, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.train_acc = list()
        self.train_loss = list()
        self.val_acc = list()
        self.val_loss = list()

    def forward(self, inputs_embeds):
        output, _ = self.bert(inputs_embeds=inputs_embeds, return_dict=False)
        linear_output = self.linear(output)
        final_layer = self.logsoftmax(linear_output)
        return final_layer
