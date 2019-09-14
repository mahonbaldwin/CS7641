import random
import pandas as pd
import pydotplus
from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split

import numpy as np
import util.charts as charts
import util.stats as stats
import util.util as util
import os
import datetime
import json
import csv

# graph.draw('file.png')

debug_mode = True

attempt = 'd'

all_tree_types = {'decision': tree.DecisionTreeClassifier,
                  'ada': ensemble.AdaBoostClassifier}


def log(v, m=None):
    if debug_mode:
        print("------------------")
        if m is not None:
            print(m)
        print(v)


def decision_classifier_options(options=None):
    max_leaf_nodes = util.myget(options, 'max-leaf-nodes')
    max_depth = util.myget(options, 'max-depth')
    return {'class_weight': None,
            'criterion': 'gini',
            'max_depth': max_depth,
            'max_leaf_nodes': max_leaf_nodes,
            'min_samples_leaf': 5,
            'min_samples_split': 2,
            'min_weight_fraction_leaf': 0.0,
            'presort': False,
            'random_state': None,
            'splitter': 'random'}


def ada_classifier_options(options=None):
    return {'n_estimators': 50}


all_tree_options = {'ada': ada_classifier_options,
                    'decision': decision_classifier_options}


def get_data(filename):
    return pd.read_csv(filename)


def write_results(seed, tree_type, data_dir, data_amount, n_test, error_train, error_test, time, options=None, prepend_attempt=""):
    sub = '-with-opts' if util.myget(options, 'tree-options', None) is not None else '-no-opts'
    fields = [attempt, tree_type+sub, seed, data_amount, n_test, error_train, error_test, time, str(options)]
    with open(data_dir+prepend_attempt+attempt+'-results-comb.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    with open(data_dir+prepend_attempt+attempt+'-results-'+tree_type+sub+'.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def prep_data(data, y_key, options=None, amount_data=0.67):
    options = util.default(options, {})
    exclude_keys = util.myget(options, 'exclude-keys', [])
    exclude_keys.append(y_key)

    x = data.loc[:, data.columns.difference(exclude_keys)]
    y = data.loc[:, y_key]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-amount_data)

    return x_train, x_test, y_train, y_test, len(x_train)


def performance(classifier, x_test, y_test, confusion_martix_name):
    predicted = classifier.predict(x_test)
    actual = y_test.values

    stats.conf_matrix(predicted, actual, confusion_martix_name)

    return stats.all_stats(predicted, actual)


def classify(seed, data, tree_type, amount_data, options=None):
    np.random.seed(seed)
    tree_options = util.myget(options, 'tree-options', {})

    x_train, x_test, y_train, y_test, n_test = prep_data(data, util.myget(options['training-options'], 'y-key'), options['training-options'], amount_data)

    classifier = tree_type(**tree_options)
    classifier.fit(x_train, y_train)

    return classifier, x_train, x_test, y_train, y_test, n_test


def classifier_and_record(seed, data, t, parent_dir, data_amount, options=None, prepend_attempt=""):
    start_time = datetime.datetime.now()
    tree_type = util.myget(all_tree_types, t)
    # Creates and trains a classifier, then saves charts and stats
    tree_options = util.myget(options, 'tree-options')
    fn = 'basic' if tree_options is None else 'option'

    tree_filename = util.myget(util.myget(options, 'training-options'), 'image-name').format('tree', t, fn, 'svg')
    classifier, x_train, x_test, y_train, y_test, n_test = classify(seed, data, tree_type, data_amount, options)
    if t is not 'ada':
        charts.save_tree_chart(tree_filename, classifier)
    results = [performance(classifier, x_train, y_train, util.myget(util.myget(options, 'training-options'), 'image-name').format('confusion-matrix-train', t, fn, 'csv')), performance(classifier, x_test, y_test, util.myget(util.myget(options, 'training-options'), 'image-name').format('confusion-matrix-test', t, fn, 'csv'))]
    [train_results, test_restuls] = results
    total_time = datetime.datetime.now() - start_time
    write_results(seed, t, parent_dir, data_amount, n_test, train_results['error-perc'], test_restuls['error-perc'], total_time, options, prepend_attempt)
    return results


def compare_tree_options(seed, data, tree_types, parent_dir, options, data_amount, prepend_attempt=""):
    training_options_map = {'training-options': util.myget(options, 'training-options')}

    combined_stats = {}
    for t in tree_types:
        options['tree-options'] = util.myget(all_tree_options, t)(util.myget(options, 'tree-options', {}))

        no_opts_stats = classifier_and_record(seed, data, t, parent_dir, data_amount, training_options_map, prepend_attempt)
        with_opts_stats = classifier_and_record(seed, data, t, parent_dir, data_amount, options, prepend_attempt)

        stats = {}

        stats['no-options'] = no_opts_stats
        stats['with-options'] = with_opts_stats

        # stats['with-options']['options'] = util.myget(options, 'tree-options', {})

        combined_stats[t] = stats

    return combined_stats


def check_trees(seed, name, data_file, classifier_key, y_key, exclude_keys, split_features=None, attempt_prepend="", data_amount=0.67):
    np.random.seed(seed)
    parent_dir='dt-exports/'+name+'/'
    dir_name = parent_dir+'attempts/'+str(seed)+'/'
    attempt_name = dir_name+attempt_prepend+attempt+'-'
    os.makedirs(dir_name, exist_ok=True)
    data = get_data('../resources/'+ name +'/'+ data_file)
    if classifier_key is not None:
        data = util.classify_data(data, classifier_key)
    split_features = [] if split_features is None else split_features
    for s in split_features:
        data = util.split_feature(data, s)

    results = {'original': {}}
    leaf_limit = 10
    max_depth = 3
    options = {'tree-options': {'max-leaf-nodes': leaf_limit, 'max-depth': max_depth},
               'training-options': {'y-key': y_key,
                                    'exclude-keys': exclude_keys,
                                    'image-name': attempt_name + '{}-{}-{}-orig-no-opt.{}'}}

    results_info = compare_tree_options(seed, data, ['decision', 'ada'], parent_dir, options, data_amount, attempt_prepend)

    results['original'] = results_info

    with open(attempt_name+'results.json', 'w') as outfile:
        json.dump(results, outfile)

    if classifier_key is not None:
        charts.bar_chart(attempt_name + name + '-histogram.png', data, classifier_key)
    else:
        charts.bar_chart(attempt_name + name + '-histogram.png', data, y_key)


def brest_cancer_trees(seed, data_amount):
    return check_trees(seed, 'breast-cancer', 'wdbc.data.csv', 'diagnosis', 'diagnosis-prime', ['diagnosis'], data_amount=data_amount)


def wine_trees(seed, data_amount):
    check_trees(seed, 'winequality', 'winequality-combined.csv', None, 'quality', [], ['type'], data_amount=data_amount)
    check_trees(seed, 'winequality', 'winequality-combined-cat.csv', None, 'Reviews', [], [], 'cat-', data_amount=data_amount)
    # return check_trees(seed, 'winequality', 'winequality-white.csv', None, 'quality', [], [])


def main():
    # for i in range(1000):
    #     seed = random.randrange(4294967296)
    #     log(str(seed) + " with " + str(r))
    #     brest_cancer_trees(seed, r)
    #     wine_trees(seed, r)
    for i in range(10):
        seed = random.randrange(4294967296)
        for r in [.95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]:
            log(str(i) + ' ' + str(seed) + " with " + str(r))
            brest_cancer_trees(seed, r)
            wine_trees(seed, r)


if __name__ == "__main__":
    main()