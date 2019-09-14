import random
import pandas as pd
import pydotplus
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import util.charts as charts
import util.stats as stats
import util.util as util
import os
import datetime
import json
import csv

attempt = 'b'
dir_name = 'knn'

def write_results(seed, type, export_dir, train_error, test_error, time, sample_size, sub='', k=''):
    microseconds  = time.seconds * 1000000 + time.microseconds
    os.makedirs(export_dir, exist_ok=True)
    fields = [attempt, sub + type, seed, k, train_error, test_error, sample_size, microseconds]
    with open(export_dir + attempt + '-results-' + type + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def train_standard(X, y, k):
    start_time = datetime.datetime.now()
    clf = KNeighborsClassifier(n_neighbors=k)
    classifier = clf.fit(X, y)
    total_time = datetime.datetime.now() - start_time
    return classifier, total_time


def reports(classifier, x_train, y_train, x_test, y_test, export_dir, sub=''):
    # will print results to csv and output confusion matrix to attepts folder
    os.makedirs(export_dir, exist_ok=True)
    return [stats.performance(classifier, x_train, y_train, export_dir + '/'+dir_name+sub+'_confusion-train.csv'), stats.performance(classifier, x_test, y_test, export_dir + '/'+ dir_name + '_' +sub+'_confusion-test.csv')]

def breast_cancer_report(seed, data, export_dir, r, k):
    np.random.seed(seed)
    [x_train, x_test, y_train, y_test] = util.prep_data(data, 'diagnosis_M', {'exclude-keys': ['id', 'diagnosis_B', 'diagnosis'], 'perc_sample': r})
    [classifier, train_time] = train_standard(x_train, y_train, k)
    [is_report, os_report] = reports(classifier, x_train, y_train, x_test, y_test, export_dir+'/attempts/'+str(seed))
    write_results(seed, dir_name, export_dir, is_report['error-perc'], os_report['error-perc'], train_time, len(x_test), k=k)

def breast_cancer_prep():
    bc_data = util.get_data('../resources/breast-cancer/wdbc.data.csv')
    bc_data = util.split_feature(bc_data, 'diagnosis')
    return bc_data

def wine_report(seed, level, data, export_dir, y_key, split_options, r, k):
    np.random.seed(seed)
    split_options['perc_sample'] = r
    [x_train, x_test, y_train, y_test] = util.prep_data(data, y_key, split_options)
    [classifier, train_time] = train_standard(x_train, y_train, k)
    [is_report, os_report] = reports(classifier, x_train, y_train, x_test, y_test, export_dir+'/attempts/'+str(seed), level)
    write_results(seed, dir_name +'-' + level, export_dir, is_report['error-perc'], os_report['error-perc'], train_time, len(x_test), level, k)

def wine_report_prep(level):
    if level is 'cat':
        filename = '../resources/winequality/winequality-combined-cat.csv'
        y_key = 'Reviews'
        split_options = {'exclude-keys': ['quality']}
    else:
        filename = '../resources/winequality/winequality-combined.csv'

        y_key = 'quality'
        split_options = {'exclude-keys': []}
    data = util.get_data(filename)
    return data, y_key, split_options


def main():
    bc_data = breast_cancer_prep()
    reg_data, reg_y_key, reg_split_options = wine_report_prep('reg')
    cat_data, cat_y_key, cat_split_options = wine_report_prep('cat')
    for i in range(1000):
        seed = random.randrange(4294967296)
        for r in [.95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]:
            for k in list(range(2, 10)):
                print(str(i) + " weight: " + str(r) + " k: " + str(k) + " seed: " + str(seed))
                breast_cancer_report(seed, bc_data, dir_name+'-exports/breast-cancer/', r, k)
                wine_report(seed, 'reg', reg_data, dir_name+'-exports/wine/', reg_y_key, reg_split_options, r, k)
                wine_report(seed, 'cat', cat_data, dir_name+'-exports/wine/', cat_y_key, cat_split_options, r, k)


if __name__ == "__main__":
    main()