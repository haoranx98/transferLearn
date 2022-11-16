import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('./train-test.csv', header=None, sep=',')
    df.columns = list('abcdef')
    epoch = [float(i) for i in df['a'].to_list()[1:]]
    test_accuracy = [float(i) for i in df['b'].to_list()[1:]]
    test_f1_score = [float(i) for i in df['c'].to_list()[1:]]
    test_loss = [float(i) for i in df['d'].to_list()[1:]]
    test_precision = [float(i) for i in df['e'].to_list()[1:]]
    test_recall = [float(i) for i in df['f'].to_list()[1:]]

    l1 = plt.plot(epoch, test_accuracy, 'r-', label='test_accuracy')
    l2 = plt.plot(epoch, test_f1_score, 'g-', label='test_f1_score')
    l3 = plt.plot(epoch, test_loss, 'b-', label='test_loss')
    l4 = plt.plot(epoch, test_precision, 'y-', label='test_precision')
    l5 = plt.plot(epoch, test_recall, 'p-', label='test_recall')

    file_name='test_resNet34'
    plt.title(file_name)
    plt.xlabel('epochs')
    plt.ylabel('column')
    plt.legend()
    plt.savefig('./pic/'+ file_name + '.jpg')
    plt.show()
