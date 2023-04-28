import ClassifierFiles.datasetTrain as datasetTrain
import ClassifierFiles.confusion_matrix as confusion_matrix

import torch


def evaluate(model, test_data, filename, num_classes):
    test = datasetTrain.Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0

    with torch.no_grad():
        model.eval()

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

    print(f'Test Accuracy: {total_acc_test / (len(test_data)*64): .3f}')

    first_class = len([x for x in true_labels if x == 0])
    second_class = len([x for x in true_labels if x == 1])
    third_class = len([x for x in true_labels if x == 2])
    print(f" num true labels: {first_class, second_class, third_class}")

    first_class = len([x for x in pred_labels if x == 0])
    second_class = len([x for x in pred_labels if x == 1])
    third_class = len([x for x in pred_labels if x == 2])
    print(f" num pred labels: {first_class, second_class, third_class}")

    assert len(pred_labels) == len(true_labels)

    confusion_matrix.save_confusion_matrix_to_file(true_labels, pred_labels, num_classes, filename)

    return pred_labels, true_labels
    