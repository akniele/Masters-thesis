from datasetTrain import Dataset
from sklearn.metrics import classification_report
from confusion_matrix import save_confusion_matrix_to_file

import torch


def evaluate(model, test_data, num_classes):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():

        true_labels = []
        pred_labels = []

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            input_embeds = test_input.to(device)

            output = model(inputs_embeds=input_embeds)
            output = torch.permute(output, (0, 2, 1))

            # true and predicted labels to be fed to classification report
            test_label = test_label.squeeze(0)
            pred_label = output.argmax(dim=1).squeeze(0)
            true_labels.extend(test_label.tolist())
            pred_labels.extend(pred_label.tolist())

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    save_confusion_matrix_to_file(true_labels, pred_labels, num_classes)
    #print(f'Classification report:\n{classification_report(true_labels, pred_labels)}')
    