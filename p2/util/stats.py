import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import csv


def all_stats(predicted, actual):
    p = np.array(predicted)
    a = np.array(actual)
    df = pd.DataFrame({'predicted': p, 'actual': a})
    error = np.abs(df.loc[:, 'predicted'] - df.loc[:, 'actual'])
    df.loc[:, 'error'] = error
    unique, counts = np.unique(error, return_counts=True)
    counts_map = dict(zip(unique, counts))
    counts_map = {'k'+str(k):int(v) for k,v in counts_map.items()}
    count = len(error)
    accurate = (count - np.count_nonzero(error))
    return {"accuracy-perc": accurate / count,
            "error-perc": 1 - accurate / count, # todo maybe this should be just `error`?
            "mean-error": np.mean(error),
            "error-std": np.std(error),
            "total-correct": accurate,
            "total-incorrect": count - accurate,
            "min-error": int(np.min(error)),
            "max-error": int(np.max(error)),
            "error-counts": counts_map}

def accuracy(x, y):
    return accuracy_score(x, y)


def conf_matrix(predicted, actual, filename):
    labels = sorted(set(actual))
    try:
        cm = confusion_matrix(predicted, actual, labels=labels)
    except:
        cm = None

    if cm is not None:
        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(labels)
            for c in cm:
                writer.writerow(c)

    # fields = [attempt, tree_type, seed, accuracy, time, str(options)]



    # f =  open(filename, 'w')
    # f.write(str(confusion_matrix(predicted, actual)))

def performance(classifier, x_test, y_test, confusion_martix_name=None):
    predicted = classifier.predict(x_test)
    actual = y_test.values

    conf_matrix(predicted, actual, confusion_martix_name)

    return all_stats(predicted, actual)