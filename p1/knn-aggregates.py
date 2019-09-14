import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_columns = ['Attempt', 'Type', 'seed', 'k', 'IS Error', 'OS Error', 'Sample Size', 'µs']
is_agg_columns = ['IS Error Mean', 'IS Error St Dev', 'IS Error Median', 'IS Error Min', 'IS Error Max']
os_agg_columns = ['OS Error Mean', 'OS Error St Dev', 'OS Error Median', 'OS Error Min', 'OS Error Max']
time_agg_columns = ['µs Mean', 'µs St Dev', 'µs Median', 'µs Min', 'µs Max']
all_agg_columns = ['Error Mean', 'Error St Dev', 'Error Median', 'Error Min', 'Error Max']

def get_stats(filename):
    print(filename)
    data = pd.read_csv(filename+'.csv', names=csv_columns)
    k_group = data.groupby('k')

    type_agg_train = k_group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_train.columns = is_agg_columns

    type_agg_test = k_group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_test.columns = os_agg_columns

    time_agg = k_group['µs'].agg([np.mean, np.std, np.median, np.min, np.max])
    time_agg.columns = time_agg_columns

    data_agg_os = k_group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_agg_os.columns = os_agg_columns
    data_agg_is = k_group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_agg_is.columns = is_agg_columns
    data_time_agg = k_group['µs'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_time_agg.columns = time_agg_columns

    data_agg_comb = pd.concat([data_agg_os, data_agg_is, data_time_agg], axis=1, sort=False)
    data_agg_comb.to_csv(filename+'.stats.all.csv')


def get_charts(dir, filename):
    data = pd.read_csv(dir + '/' + filename+'.csv', names=csv_columns)
    groups = {}
    for k in list(range(2,10)):
        k_data = data.loc[data['k'] == k]
        sample_size_group = k_data.groupby('Sample Size')
        count_agg_train_is = sample_size_group['IS Error'].agg([np.mean])
        count_agg_train_is.columns = ['IS Error Mean']

        count_agg_train_os = sample_size_group['OS Error'].agg([np.mean])
        count_agg_train_os.columns = ['OS Error Mean']

        agg = pd.concat([count_agg_train_is, count_agg_train_os], axis=1, sort=False)
        agg.sort_values(['Sample Size'])

        # plt.plot(agg['IS Error Mean'], agg['OS Error Mean'])
        plt.plot(agg['IS Error Mean'], 'r', agg['OS Error Mean'], 'b')
        plt.legend(['In Sample', 'Out Of Sample'])
        plt.xlabel(str(k)+' Training Sample Size')
        plt.ylabel(str(k)+' Mean Error Rate')
        plt.savefig(dir+'/'+filename+'_'+str(k)+'_learning_rate.png')
        plt.clf()


def to_hex(k):
    if k <= 5:
        x = "#0000{:02x}".format(round(255/3) * (k - 2))
    elif k > 5:
        kp = k - 4
        x = "#00{:02x}ff".format(round(255/4) * (kp - 2))
    return x


def get_charts_agg(dir, filename):
    data = pd.read_csv(dir + '/' + filename+'.csv', names=csv_columns)

    ks = list(range(2,10))
    for k in ks:
        kernel_data = data.loc[data['k'] == k]
        sample_size_group = kernel_data.groupby('Sample Size')
        count_agg_train_is = sample_size_group['IS Error'].agg([np.mean])
        count_agg_train_is.columns = ['IS Error Mean']

        count_agg_train_os = sample_size_group['OS Error'].agg([np.mean])
        count_agg_train_os.columns = ['OS Error Mean']

        agg = pd.concat([count_agg_train_is, count_agg_train_os], axis=1, sort=False)
        agg.sort_values(['Sample Size'])

        # plt.plot(agg['IS Error Mean'], agg['OS Error Mean'])
        plt.plot(agg['IS Error Mean'], to_hex(k))
        plt.plot(agg['OS Error Mean'], to_hex(k), linestyle='--')
    # plt.legend(['k=2 In Sample', 'k=2 Out Of Sample',
    #             'k=3 In Sample', 'k=3 Out Of Sample',
    #             'k=4 In Sample', 'k=4 Out Of Sample',
    #             'k=5 In Sample', 'k=5 Out Of Sample',
    #             'k=6 In Sample', 'k=6 Out Of Sample',
    #             'k=7 In Sample', 'k=7 Out Of Sample',
    #             'k=8 In Sample', 'k=8 Out Of Sample',
    #             'k=9 In Sample', 'k=9 Out Of Sample'], facecolor='white', framealpha=1)
    plt.xlabel('Training Sample Size')
    plt.ylabel('Mean Error Rate')

    plt.savefig(dir+'/'+filename+'_learning_rate.png')
    plt.clf()


def main():
    get_stats('knn-exports/wine/b-results-knn-reg')
    get_stats('knn-exports/wine/b-results-knn-cat')
    get_stats('knn-exports/breast-cancer/b-results-knn')
    get_charts('knn-exports/wine/','b-results-knn-reg')
    get_charts('knn-exports/wine/','b-results-knn-cat')
    get_charts('knn-exports/breast-cancer/','b-results-knn')
    get_charts_agg('knn-exports/wine/','b-results-knn-reg')
    get_charts_agg('knn-exports/wine/','b-results-knn-cat')
    get_charts_agg('knn-exports/breast-cancer/','b-results-knn')


if __name__ == "__main__":
    main()