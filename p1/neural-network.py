import datetime
import os
import random

from sklearn.neural_network import MLPClassifier
import numpy as np

import util.charts as charts
from itertools import combinations_with_replacement, permutations
import util.stats as stats
import util.util as util
import csv


attempt = 'x'
all_class_types = {'net': MLPClassifier}



def write_results(seed, type, export_dir, train_error, test_error, time, n_sample, layers, sub=''):
    fields = [attempt, sub+type, seed, train_error, test_error, time, n_sample, layers]
    os.makedirs(export_dir, exist_ok=True)
    with open(export_dir+attempt+'-results-comb.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    with open(export_dir + attempt +'-results-' + type + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def train_and_test(dir, split_data, classifier, layer):
    x_train, x_test, y_train, y_test = split_data
    classifier = all_class_types[classifier](solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layer)
    classifier.fit(x_train, y_train)
    # test classifier

    os.makedirs(dir, exist_ok=True)

    return [stats.performance(classifier, x_train, y_train, dir+'net_confusion-train.csv'), stats.performance(classifier, x_test, y_test, dir+'net_confusion-test.csv')]


def breast_cancer_net(seed, r, options=None):
    dir = 'nn-exports/breast_cancer/'
    seed_dir = 'attempts/'+dir+str(seed)+'/'
    np.random.seed(seed)
    options = util.default(options, {})
    class_types = util.myget(options, 'class_types', ['net'])
    layers = util.myget(options, 'layers', [(5,6)])
    data = util.get_data('../resources/breast-cancer/wdbc.data.csv')
    data = util.split_feature(data, 'diagnosis')
    split_data = util.prep_data(data, 'diagnosis_M', {'exclude-keys': ['id', 'diagnosis_B', 'diagnosis'], 'perc_sample': r})
    net_results = {}
    for t in class_types:
        for l in layers:
            np.random.seed(seed)
            start_time = datetime.datetime.now()
            [train_results, test_results] = train_and_test(seed_dir, split_data, t, l)
            net_results[t+'_train'] = train_results
            net_results[t+'_test'] = test_results
            total_time = datetime.datetime.now() - start_time
            write_results(seed, t, dir, train_results['error-perc'], test_results['error-perc'], total_time, len(split_data[0]), l)
    return net_results



def wine_net(seed, r, nn_type, options=None):
    if nn_type is 'cat':
        filename = '../resources/winequality/winequality-combined-cat.csv'
        y_key = 'Reviews'
        split_options = {'exclude-keys': ['quality']}
    else:
        filename = '../resources/winequality/winequality-combined.csv'
        y_key = 'quality'
        split_options = {'exclude-keys': []}
    split_options['perc_sample'] = r
    dir = 'nn-exports/wine/'
    seed_dir = 'attempts/'+dir+str(seed)+'/'
    np.random.seed(seed)
    options = util.default(options, {})
    class_types = util.myget(options, 'class_types', ['net'])
    layers = util.myget(options, 'layers', [(5,6)])
    data = util.get_data(filename)
    split_data = util.prep_data(data, y_key, split_options)
    net_results = {}
    for t in class_types:
        for l in layers:
            np.random.seed(seed)
            start_time = datetime.datetime.now()
            [train_results, test_results] = train_and_test(seed_dir, split_data, t, l)
            net_results[t+'_train'] = train_results
            net_results[t+'_test'] = test_results
            total_time = datetime.datetime.now() - start_time
            write_results(seed, t, dir, train_results['error-perc'], test_results['error-perc'], total_time, len(split_data[0]), l, nn_type + '-')
    pass


# def hidden_layers():
#     node_high = 15
#     layers_high = 5
#     all_possible = list(combinations_with_replacement(list(range(node_high)), layers_high))
#     new_possible = []
#     for p in all_possible:
#         for perm in permutations(list(filter(lambda x: x != 0, p))):
#             unique_list = list(set(filter(lambda x: len(x) != 0, new_possible)))
#     for i in range(1, node_high):
#         unique_list.append((i))
#     return unique_list

def main():
    # layers = hidden_layers()
    # layers = [(5)]
    layers = [(5), (5,5), (5,5,5), (5,5,5,5), (5,5,5,5,5), (6), (6,6), (6,6,6), (6,6,6,6), (6,6,6,6,6), (7), (7,7), (7,7,7), (7,7,7,7), (7,7,7,7,7), (8), (8,8), (8,8,8), (8,8,8,8), (8,8,8,8,8), (9), (9,9), (9,9,9), (9,9,9,9), (9,9,9,9,9), (10), (10,10), (10,10,10), (10,10,10,10), (10,10,10,10,10), (11), (11,11), (11,11,11), (11,11,11,11), (11,11,11,11,11)]
    options = {'layers': layers}
    print("number of unique hidden layer configurations: ", len(layers))
    # options = {'layers': [(3,3,3,3,3), (2,3), (3,2), (4), (5,6), (2,3,4), (4,3,2), (7, 2), (5,5,5,5,5), (3,4,5,6,5,4,3), (7,7,7,7,7,7,7), (5), (5,5), (5,5,5), (5,5,5,5), (6), (6,6), (6,6,6), (6,6,6,6), (7), (7,7), (7,7,7), (7,7,7,7), (8), (8,8), (8,8,8), (8,8,8,8), (9), (9,9), (9,9,9), (9,9,9,9), (10), (10,10), (10,10,10), (10,10,10,10), (11), (11,11), (11,11,11), (11,11,11,11)]}
    options = {'layers': layers}
    for i in range(100):
        seed = random.randrange(4294967296)
        for r in [.95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]:
            print(str(i) + " weight: " + str(r) + " seed: " + str(seed))
            # seed = 4249808607
            breast_cancer_net(seed, r, options)
            wine_net(seed, r, 'no-cat', options)
            wine_net(seed, r, 'cat', options)



if __name__ == "__main__":
    main()