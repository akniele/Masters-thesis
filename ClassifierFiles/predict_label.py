import pandas as pd

import ClassifierFiles.datasetPredict as datasetPredict

import torch
from tqdm import tqdm


def predict_label(model, predict_data):
    predict = datasetPredict.Dataset(predict_data)

    predict_dataloader = torch.utils.data.DataLoader(predict, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        pred_labels = []

        for test_input in tqdm(predict_dataloader):
            input_embeds = test_input.to(device)

            output = model(inputs_embeds=input_embeds)
            output = torch.permute(output, (0, 2, 1))

            pred_label = output.argmax(dim=1).squeeze(0)
            pred_labels.extend(pred_label.tolist())

    first_class = len([x for x in pred_labels if x == 0])
    second_class = len([x for x in pred_labels if x == 1])
    third_class = len([x for x in pred_labels if x == 2])
    print(f" num pred labels: {first_class, second_class, third_class}")

    return pred_labels


if __name__ == "__main__":

    predict_data = [[[4, 6, 3], [9, 66, 2], [0, 86, 3]],
                    [[2, 77, 4], [88, 1, 2], [9, 4, 5]]]

    df_dict = {"text": predict_data}

    df = pd.DataFrame(df_dict)

    pred_labels = predict_label(df)
    print(pred_labels)
