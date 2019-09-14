import datetime
import os
import random

from sklearn import svm
from itertools import combinations_with_replacement, permutations
import util.stats as stats
import numpy as np
import util.util as util
import csv
from sklearn.model_selection import cross_val_score

# code for support vector machines

attempt = 'c'

def write_results(seed, type, export_dir, train_error, test_error, time, sample_size, sub='', kernel='', ct_scores=None):
    microseconds  = time.seconds * 1000000 + time.microseconds
    os.makedirs(export_dir, exist_ok=True)
    fields = [attempt, sub + type, seed, kernel, train_error, test_error, sample_size, microseconds, ct_scores]
    with open(export_dir + attempt + '-results-' + type + '.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def train_standard(X, y, kernel):
    start_time = datetime.datetime.now()
    clf = svm.SVC(gamma='scale', kernel=kernel)
    classifier = clf.fit(X, y)
    total_time = datetime.datetime.now() - start_time
    return classifier, total_time

def train_cross_validate(X, y, kernel):
    clf = svm.SVC(gamma='scale', kernel=kernel)
    scores = cross_val_score(clf, X, y, cv=4)
    return scores


def reports(classifier, x_train, y_train, x_test, y_test, export_dir, sub=''):
    # will print results to csv and output confusion matrix to attepts folder
    os.makedirs(export_dir, exist_ok=True)
    return [stats.performance(classifier, x_train, y_train, export_dir + '/svm'+sub+'_confusion-train.csv'), stats.performance(classifier, x_test, y_test, export_dir + '/svm_'+sub+'_confusion-test.csv')]

def breast_cancer_report(seed, data, export_dir, r, kernel):
    np.random.seed(seed)
    [x_train, x_test, y_train, y_test] = util.prep_data(data, 'diagnosis_M', {'exclude-keys': ['id', 'diagnosis_B', 'diagnosis'], 'perc_sample': r})
    [classifier, train_time] = train_standard(x_train, y_train, kernel)
    [is_report, os_report] = reports(classifier, x_train, y_train, x_test, y_test, export_dir+'/attempts/'+str(seed))
    ct_scores = train_cross_validate(x_train, y_train, kernel)
    write_results(seed, 'svm', export_dir, is_report['error-perc'], os_report['error-perc'], train_time, len(x_test), kernel=kernel, ct_scores=ct_scores)

def breast_cancer_prep():
    bc_data = util.get_data('../resources/breast-cancer/wdbc.data.csv')
    bc_data = util.split_feature(bc_data, 'diagnosis')
    return bc_data

def wine_report(seed, level, data, export_dir, y_key, split_options, r, kernel):
    np.random.seed(seed)
    split_options['perc_sample'] = r
    [x_train, x_test, y_train, y_test] = util.prep_data(data, y_key, split_options)
    [classifier, train_time] = train_standard(x_train, y_train, kernel)
    [is_report, os_report] = reports(classifier, x_train, y_train, x_test, y_test, export_dir+'/attempts/'+str(seed), level)
    ct_scores = train_cross_validate(x_train, y_train, kernel)
    write_results(seed, 'svm-'+level, export_dir, is_report['error-perc'], os_report['error-perc'], train_time, len(x_test), level, kernel, ct_scores)

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
    for i in range(10):
        seed = random.randrange(4294967296)
        for r in [.95, .9, .85, .8, .75, .7, .65, .6, .55, .5, .45, .4, .35, .3, .25, .2, .15, .1, .05]:
            for k in ['linear', 'poly', 'rbf', 'sigmoid']:
                print(str(i) + " weight: " + str(r) + " kernel: " + k + " seed: " + str(seed))
                breast_cancer_report(seed, bc_data, 'svm-exports/breast-cancer/', r, k)
                wine_report(seed, 'reg', reg_data, 'svm-exports/wine/', reg_y_key, reg_split_options, r, k)
                wine_report(seed, 'cat', cat_data, 'svm-exports/wine/', cat_y_key, cat_split_options, r, k)


if __name__ == "__main__":
    main()