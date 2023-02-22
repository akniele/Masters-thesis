from datasetTrain import Dataset

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict


def train(model, train_data, val_data, learning_rate, epochs, BATCH_SIZE):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            input_embeds = train_input.to(device)

            output = model(inputs_embeds=input_embeds)
            output = torch.permute(output, (0, 2, 1))

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # model.train_acc.append(total_acc_train / len(train_data))
        # model.train_loss.append(total_loss_train / len(train_data))
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                input_embeds = val_input.to(device)

                output = model(inputs_embeds=input_embeds)
                output = torch.permute(output, (0, 2, 1))

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            # model.val_acc.append(total_acc_val / len(val_data))
            # model.val_loss.append(total_loss_val / len(val_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .4f} | Train Accuracy: {total_acc_train / len(train_data): .4f} | Val Loss: {total_loss_val / len(val_data): .4f} | Val Accuracy: {total_acc_val / len(val_data): .4f}')

    # acc_loss = defaultdict()
    # acc_loss["train_accuracy"] = model.train_acc
    # acc_loss["train_loss"] = model.train_loss
    # acc_loss["val_accuracy"] = model.val_acc
    # acc_loss["val_loss"] = model.val_loss

    # with open("acc_loss_pcm_softmax.pkl", "wb") as p:
    #     pickle.dump(acc_loss, p)
