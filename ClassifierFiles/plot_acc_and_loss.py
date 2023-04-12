import matplotlib.pyplot as plt


def acc_loss_plot(records):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(range(len(records["train_loss"])), records["train_loss"], label="training loss", color=color)
    ax1.plot(range(len(records["val_loss"])), records["val_loss"], label="validation loss", color=color,
             linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(range(len(records["train_accuracy"])), records["train_accuracy"], label="training accuracy", color=color)
    ax2.plot(range(len(records["val_accuracy"])), records["val_accuracy"], label="validation accuracy", color=color,
             linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc='center right', bbox_to_anchor=(0.85, 0.55))
    plt.savefig('acc_loss_plot.png')


def loss_plot(records, file_name):  # for baseline model (we are not calculating the accuracy for this one)
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(range(len(records["train_loss"])), records["train_loss"], label="training loss", color=color)
    ax1.plot(range(len(records["val_loss"])), records["val_loss"], label="validation loss", color=color,
             linestyle='dashed')
    ax1.tick_params(axis='y', labelcolor=color)

    #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'tab:red'
    # ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
    # ax2.plot(range(len(records["train_accuracy"])), records["train_accuracy"], label="training accuracy", color=color)
    # ax2.plot(range(len(records["val_accuracy"])), records["val_accuracy"], label="validation accuracy", color=color,
    #          linestyle='dashed')
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc='center right', bbox_to_anchor=(0.85, 0.55))
    plt.savefig(file_name)
