import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def plotROC(preds, truths, sample_weight=None, output=None, **kwargs):
    from sklearn.metrics import auc, roc_curve
    num_classes = preds.shape[1]
    if truths.ndim==1:
        truths = to_categorical(truths)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        label = 'class_%d' % i
        fpr[label], tpr[label], _ = roc_curve(truths[:, i], preds[:, i], sample_weight=sample_weight)
        roc_auc[label] = auc(fpr[label], tpr[label], reorder=True)

    plt.figure()
    for label in roc_auc:
        legend = '%s (area = %0.4f)' % (label, roc_auc[label])
        print(legend)
        plt.plot(tpr[label], 1 - fpr[label], label=legend)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0, 1])
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background rejection')
#     plt.title('Receiver operating characteristic example')
    plt.legend(loc='best')
#     plt.yscale('log')
    plt.grid()
    if 'y_min' in kwargs:
        plt.ylim(kwargs['y_min'], 1)
    if output:
        plt.savefig(output)
    return plt
