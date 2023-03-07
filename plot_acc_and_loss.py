import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

# with open("acc_loss_pcm_pcm.pkl", "rb") as p:
#     records = pickle.load(p)


def acc_loss_plot(records):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(range(len(records["train_loss"])), records["train_loss"], label="training loss", color=color)
    ax1.plot(range(len(records["val_loss"])), records["val_loss"], label="validation loss", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(records["train_accuracy"])), records["train_accuracy"], label="training accuracy", color=color)
    ax2.plot(range(len(records["val_accuracy"])), records["val_accuracy"], label="validation accuracy", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('acc_loss_plot.png')
