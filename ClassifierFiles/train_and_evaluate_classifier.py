from ClassifierFiles.trainingDataClassifier import prepare_training_data
from ClassifierFiles.classifier import BertClassifier, EarlyStopper
from ClassifierFiles.trainClassifier import train
from ClassifierFiles.evaluateClassifier import evaluate
from ClassifierFiles.predict_label import predict_label
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch


"""train classifier on big probs and labels from clustering step"""


def train_and_evaluate_classifier(num_classes, batch_size, epochs, lr, labels, num_sheets, function, filename):
    NUM_CLASSES = num_classes
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LR = lr

    print("  => PREPARE TRAINING DATA FOR CLASSIFIER")

    df = False

    labels = list(labels)
    labels = iter(labels)

    for i in tqdm(range(9)):
        with open(f"train_data/big_10000_{i}.pkl", "rb") as f:
            probs1 = pickle.load(f)
        probs1 = probs1.numpy()
        df_tmp = prepare_training_data(probs1, num_sheets, labels)
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
    early_stopper = EarlyStopper(patience=3)

    print("  => STARTING TRAINING")
    train(model, filename, df_train, df_val, LR, EPOCHS, BATCH_SIZE, early_stopper=early_stopper)

    print("  => EVALUATING MODEL")
    """test classifier"""
    evaluate(model, df_test, filename, num_classes=NUM_CLASSES)


def make_predictions(num_classes, num_sheets, function, epochs, lr):
    NUM_CLASSES = num_classes

    print("  => PREPARE DATA FOR PREDICTIONS")

    with open(f"train_data/big_10000_9.pkl", "rb") as f:
        probs1 = pickle.load(f)
    probs1 = probs1.numpy()
    df = prepare_training_data(probs1, num_sheets)

    print(f"length of pandas frame: {df.shape[0]}")

    model = BertClassifier(NUM_CLASSES)
    model.load_state_dict(torch.load(f'models/{function.__name__ }_{NUM_CLASSES}_{lr}_{epochs}.pt'))
    model.eval()

    print("  => PREDICTING LABELS")

    pred_labels = predict_label(model, df)

    return pred_labels
