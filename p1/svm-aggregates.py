import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_columns = ['Attempt', 'Type', 'seed', 'Kernel', 'IS Error', 'OS Error', 'Sample Size', 'µs', 'Cross Validation Scrore']
is_agg_columns = ['IS Error Mean', 'IS Error St Dev', 'IS Error Median', 'IS Error Min', 'IS Error Max']
os_agg_columns = ['OS Error Mean', 'OS Error St Dev', 'OS Error Median', 'OS Error Min', 'OS Error Max']
time_agg_columns = ['µs Mean', 'µs St Dev', 'µs Median', 'µs Min', 'µs Max']
all_agg_columns = ['Error Mean', 'Error St Dev', 'Error Median', 'Error Min', 'Error Max']

def get_stats(filename):
    print(filename)
    data = pd.read_csv(filename+'.csv', names=csv_columns)
    kernel_group = data.groupby('Kernel')

    type_agg_train = kernel_group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_train.columns = is_agg_columns

    type_agg_test = kernel_group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_test.columns = os_agg_columns

    time_agg = kernel_group['µs'].agg([np.mean, np.std, np.median, np.min, np.max])
    time_agg.columns = time_agg_columns

    data_agg_os = kernel_group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_agg_os.columns = os_agg_columns
    data_agg_is = kernel_group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_agg_is.columns = is_agg_columns
    data_time_agg = kernel_group['µs'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_time_agg.columns = time_agg_columns

    data_agg_comb = pd.concat([data_agg_os, data_agg_is, data_time_agg], axis=1, sort=False)
    data_agg_comb.to_csv(filename+'.stats.all.csv')


def get_charts(dir, filename):
    data = pd.read_csv(dir + '/' + filename+'.csv', names=csv_columns)
    groups = {}
    for k in ['linear', 'poly', 'rbf', 'sigmoid']:
        kernel_data = data.loc[data['Kernel'] == k]
        sample_size_group = kernel_data.groupby('Sample Size')
        count_agg_train_is = sample_size_group['IS Error'].agg([np.mean])
        count_agg_train_is.columns = ['IS Error Mean']

        count_agg_train_os = sample_size_group['OS Error'].agg([np.mean])
        count_agg_train_os.columns = ['OS Error Mean']

        agg = pd.concat([count_agg_train_is, count_agg_train_os], axis=1, sort=False)
        agg.sort_values(['Sample Size'])

        groups[k] = agg


        # plt.plot(agg['IS Error Mean'], agg['OS Error Mean'])
        plt.plot(agg['IS Error Mean'], 'r', agg['OS Error Mean'], 'b')
        plt.legend(['In Sample', 'Out Of Sample'])
        plt.xlabel(k+' Training Sample Size')
        plt.ylabel(k+' Mean Error Rate')
        plt.savefig(dir+'/'+filename+'_'+k+'_learning_rate.png')
        plt.clf()


def get_charts_agg(dir, filename):
    data = pd.read_csv(dir + '/' + filename+'.csv', names=csv_columns)
    colors = {'linear' : 'b', 'poly' : 'm', 'rbf' : 'r', 'sigmoid' : 'c'}
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    groups = {}
    for k in kernels:
        kernel_data = data.loc[data['Kernel'] == k]
        sample_size_group = kernel_data.groupby('Sample Size')
        count_agg_train_is = sample_size_group['IS Error'].agg([np.mean])
        count_agg_train_is.columns = ['IS Error Mean']

        count_agg_train_os = sample_size_group['OS Error'].agg([np.mean])
        count_agg_train_os.columns = ['OS Error Mean']

        agg = pd.concat([count_agg_train_is, count_agg_train_os], axis=1, sort=False)
        agg.sort_values(['Sample Size'])

        groups[k] = agg


        # plt.plot(agg['IS Error Mean'], agg['OS Error Mean'])
        plt.plot(agg['IS Error Mean'], colors[k], agg['OS Error Mean'], colors[k]+'--')
    plt.legend(['Linear In Sample', 'Linear Out Of Sample', 'Poly In Sample', 'Poly Out Of Sample', 'RBF In Sample', 'RBF Out Of Sample', 'Sigmoid In Sample', 'Sigmoid Out Of Sample'], facecolor='white', framealpha=1)
    plt.xlabel('Training Sample Size')
    plt.ylabel('Mean Error Rate')

    plt.savefig(dir+'/'+filename+'_learning_rate.png')
    plt.clf()


def main():
    get_stats('svm-exports/wine/b-results-svm-reg')
    get_stats('svm-exports/wine/b-results-svm-cat')
    get_stats('svm-exports/breast-cancer/b-results-svm')
    get_charts('svm-exports/wine/','b-results-svm-reg')
    get_charts('svm-exports/wine/','b-results-svm-cat')
    get_charts('svm-exports/breast-cancer/','b-results-svm')
    get_charts_agg('svm-exports/wine/','b-results-svm-reg')
    get_charts_agg('svm-exports/wine/','b-results-svm-cat')
    get_charts_agg('svm-exports/breast-cancer/','b-results-svm')


if __name__ == "__main__":
    main()