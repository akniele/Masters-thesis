import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict
import ClassifierFiles.datasetTrain as datasetTrain
import ClassifierFiles.plot_acc_and_loss as plot_acc_and_loss


def train(model, train_data, val_data, learning_rate, epochs, BATCH_SIZE, num_classes):
    train, val = datasetTrain.Dataset(train_data), datasetTrain.Dataset(val_data)

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

        with tqdm(train_dataloader, total=(len(train_dataloader)), unit="batch", desc="Train epoch %i" % epoch_num) as batches:
            for train_input, train_label in batches:
                train_label = train_label.to(device)
                # print(f"train label size: {train_label.size()}")
                input_embeds = train_input.to(device)
                # print(f"input embeds size: {input_embeds.size()}")

                output = model(inputs_embeds=input_embeds)
                output = torch.permute(output, (0, 2, 1))
                # print(f"output size: {output.size()}")
                #
                # print(f" length of train_label.long(): {train_label.long().size()}")
                #
                # print(f" train_label.long(): {train_label.long()}")
                # print(f" train_label: {train_label}")

                batch_loss = criterion(output, train_label.long())

                total_loss_train += batch_loss.item()

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

                batches.set_postfix(loss=batch_loss.item() / (BATCH_SIZE*64), accuracy=(acc/(BATCH_SIZE*64)))

        model.train_acc.append(total_acc_train / len(train_data))
        model.train_loss.append(total_loss_train / len(train_data))
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            with tqdm(val_dataloader, total=(len(val_dataloader)), unit="batch",
                      desc="Val epoch %i" % epoch_num) as val_batches:

                for val_input, val_label in val_batches:
                    val_label = val_label.to(device)
                    input_embeds = val_input.to(device)

                    output = model(inputs_embeds=input_embeds)
                    output = torch.permute(output, (0, 2, 1))

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc

                    val_batches.set_postfix(loss=batch_loss.item() / (BATCH_SIZE*64), accuracy=(acc / (BATCH_SIZE * 64)))

            model.val_acc.append(total_acc_val / len(val_data))
            model.val_loss.append(total_loss_val / len(val_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / (len(train_data)*64): .4f} | Train Accuracy: '
            f'{total_acc_train / (len(train_data)*64): .4f} | Val Loss: {total_loss_val / (len(val_data)*64): .4f} '
            f'| Val Accuracy: {total_acc_val / (len(val_data)*64): .4f}')

    acc_loss = defaultdict()
    acc_loss["train_accuracy"] = model.train_acc
    acc_loss["train_loss"] = model.train_loss
    acc_loss["val_accuracy"] = model.val_acc
    acc_loss["val_loss"] = model.val_loss

    plot_acc_and_loss.acc_loss_plot(acc_loss)  # create a plot with train and val accuracy and loss
