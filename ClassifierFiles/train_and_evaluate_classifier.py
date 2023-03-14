from ClassifierFiles.trainingDataClassifier import prepare_training_data, downsampled_training_data
from ClassifierFiles.classifier import BertClassifier
from ClassifierFiles.trainClassifier import train
from ClassifierFiles.evaluateClassifier import evaluate
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


"""train classifier on big probs and labels from clustering step"""


def train_and_evaluate_classifier(num_classes, batch_size, epochs, lr, labels, num_sheets):
    NUM_CLASSES = num_classes
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr

    print("  => PREPARE TRAINING DATA FOR CLASSIFIER")

    df = False

    labels = list(labels)
    labels = iter(labels)

    for i in tqdm(range(9)):
        with open(f"train_data/train_big_100000_{i}.pkl", "rb") as f:
            probs1 = pickle.load(f)
        probs1 = probs1.numpy()
        #print(f"probs {i}: {probs1[0][0]}")
        df_tmp = prepare_training_data(probs1, labels, num_sheets)
        #print(f"the shape of the temporary df is: {df_tmp.shape[0]}\n")
        if type(df) is bool:
            df = df_tmp
        else:
            df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

    print(f"This is what the training data looks like:\n {df.iloc[[0, 1, 63, 64, 65]]}")

    print(f"length of pandas frame: {df.shape[0]}")

    np.random.seed(112)
    torch.manual_seed(0)
    df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                         [int(.8 * len(df)), int(.95 * len(df))])

    model = BertClassifier(NUM_CLASSES)

    print("  => STARTING TRAINING")
    train(model, df_train, df_val, LR, EPOCHS, BATCH_SIZE, num_classes=NUM_CLASSES)

    torch.save(model.state_dict(), f'first_try.pt')

    print("  => EVALUATING MODEL")
    """test classifier"""
    pred_labels, true_labels = evaluate(model, df_test, num_classes=NUM_CLASSES)

    return pred_labels, true_labels
