import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')


def accuracy_plot(path_cls, title, label, item, xlim=None, ylim=None):
    # epoch
    df = pd.read_csv(path_cls.make_csv_path('epoch_accuracy.csv'))
    max_val = round(df[item].max(), 5)
    max_step = df[item].idxmax() + 1
    epoch = len(df.index)
    label = label + ' ' + str(max_val) + ' (' + str(max_step) + ')'
    x = list(range(1, epoch+1))
    plt.plot(x, df[item], label=label)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.90, 1]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_epoch.png')
    plt.figure()

    # step
    df = pd.read_csv(path_cls.make_csv_path('step_accuracy.csv'))
    max_val = round(df[item].max(), 5)
    max_step = df[item].idxmax() + 1
    epoch = len(df.index)
    label = label + ' ' + str(max_val) + ' (' + str(max_step) + ')'
    x = list(range(1, epoch+1))
    plt.plot(x, df[item], label=label)

    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.90, 1]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_step.png')
    plt.figure()

def multi_accuracy_plot(path_cls, title, label, items, xlim=None, ylim=None):
    # epoch
    for item in items:
        df = pd.read_csv(path_cls.make_csv_path('epoch_accuracy.csv'))
        max_val = round(df[item].max(), 5)
        max_step = df[item].idxmax() + 1
        epoch = len(df.index)
        label_name = label + ' ' + item + ' ' + str(max_val) + ' (' + str(max_step) + ')'
        x = list(range(1, epoch+1))
        plt.plot(x, df[item], label=label_name)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.90, 1]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_epoch.png')
    plt.figure()

    # step
    for item in items:
        df = pd.read_csv(path_cls.make_csv_path('step_accuracy.csv'))
        max_val = round(df[item].max(), 5)
        max_step = df[item].idxmax() + 1
        epoch = len(df.index)
        label_name = label + ' ' + item + ' ' + str(max_val) + ' (' + str(max_step) + ')'
        x = list(range(1, epoch+1))
        plt.plot(x, df[item], label=label_name)

    plt.xlabel("step")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.90, 1]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_step.png')
    plt.figure()


def multi_loss_plot(path_cls, title, label, items, xlim=None, ylim=None):
    # epoch
    for item in items:
        df = pd.read_csv(path_cls.make_csv_path('epoch_accuracy.csv'))
        # print(df[item])
        # print(type(df[item]))
        max_val = round(df[item].max(), 5)
        max_step = df[item].idxmax() + 1
        epoch = len(df.index)
        label_name = label + ' ' + item + ' ' + str(max_val) + ' (' + str(max_step) + ')'
        x = list(range(1, epoch+1))
        plt.plot(x, df[item], label=label_name)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0., 0.008]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_epoch.png')
    plt.figure()

    # step
    for item in items:
        df = pd.read_csv(path_cls.make_csv_path('step_accuracy.csv'))
        max_val = round(df[item].max(), 5)
        max_step = df[item].idxmax() + 1
        epoch = len(df.index)
        label_name = label + ' ' + item + ' ' + str(max_val) + ' (' + str(max_step) + ')'
        x = list(range(1, epoch+1))
        plt.plot(x, df[item], label=label_name)

    plt.xlabel("step")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0., 0.008]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title+'_step.png')
    plt.figure()


def multi_csv_plot(csv_path, title, label, items, xlim=None, ylim=None):
    # epoch
    for item in items:
        df = pd.read_csv(csv_path)
        max_val = round(df[item].max(), 5)
        max_step = df[item].idxmax() + 1
        epoch = len(df.index)
        label_name = label + ' ' + item + ' ' + str(max_val) + ' (' + str(max_step) + ')'
        x = list(range(1, epoch + 1))
        plt.plot(x, df[item], label=label_name)

    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    tmp = ylim if ylim is not None else [0.90, 1]
    plt.ylim(tmp)
    tmp = xlim if xlim is not None else [1, epoch]
    plt.xlim(tmp)
    plt.legend(title="Data Set", loc='lower right')
    plt.title(os.path.basename(title))
    plt.savefig(title + '_epoch.png')
    plt.figure()
    

    # max_id_x = df[item].idxmax()
    # max_step = df.at[max_id_x, 'step']
    # df.plot(x='step', y=item, label=label + ' ' + str(max_val) + ' (' + str(max_step) + ')', title=title)
    # plt.xlabel("step")
    # plt.ylabel("Accuracy")
    # tmp = ylim if ylim is not None else [0.955, 0.985]
    # plt.ylim(tmp)
    # tmp = xlim if xlim is not None else [df['step'].min(), df['step'].max()]
    # plt.xlim(tmp)
    # plt.legend(title="Data Set", loc='lower right')
    # plt.savefig(title+'.png')
