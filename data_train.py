import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('./train-train.csv', header=None, sep=',')
    df.columns = list('abcdefg')

    batch = [float(i) for i in df['a'].to_list()[1:]]
    epoch = [float(i) for i in df['b'].to_list()[1:]]
    train_accuracy = [float(i) for i in df['c'].to_list()[1:]]
    train_f1_score = [float(i) for i in df['d'].to_list()[1:]]
    train_loss = [float(i) for i in df['e'].to_list()[1:]]
    train_precision = [float(i) for i in df['f'].to_list()[1:]]
    train_recall = [float(i) for i in df['g'].to_list()[1:]]

    l1 = plt.plot(batch, train_accuracy, 'r-', label='train_accuracy')
    l2 = plt.plot(batch, train_f1_score, 'g-', label='train_f1_score')
    l3 = plt.plot(batch, train_loss, 'b-', label='train_loss')
    l4 = plt.plot(batch, train_precision, 'y-', label='train_precision')
    l5 = plt.plot(batch, train_recall, 'p-', label='train_recall')

    file_name='train_resNet34'
    plt.title(file_name)
    plt.xlabel('batchs')
    plt.ylabel('column')
    plt.legend()
    plt.savefig('./pic/'+ file_name + '.jpg')
    plt.show()
