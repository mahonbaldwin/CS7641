import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stats(filename):
    print(filename)
    data = pd.read_csv(filename+'.csv', names=['Attempt', 'Tree Type', 'Seed', 'Train Data Perc', 'Train Count', 'IS Error', 'OS Error', 'time', 'Options'])
    t = pd.Series(pd.to_timedelta(data['time'], unit="m")).dt
    data['Microseconds'] = t.seconds * 1000000 + t.microseconds
    group = data.groupby('Tree Type')
    type_agg_train = group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_train.columns = ['IS Error Mean', 'IS Error St Dev', 'IS Error Median', 'IS Error Min', 'IS Error Max']

    type_agg_test = group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    type_agg_test.columns = ['OS Error Mean', 'OS Error St Dev', 'OS Error Median', 'OS Error Min', 'OS Error Max']

    time_agg = group['Microseconds'].agg([np.mean, np.std, np.median, np.min, np.max])
    time_agg.columns = ['Microseconds Mean', 'Microseconds St Dev', 'Microseconds Median', 'Microseconds Min', 'Microseconds Max']

    agg = pd.concat([type_agg_train, type_agg_test, time_agg], axis=1, sort=False)
    agg = agg.sort_values(['OS Error Mean', 'OS Error Max', 'OS Error Median', 'OS Error Min'], ascending=[0,0,0,0])
    agg.to_csv(filename+'.stats.csv')

    data_agg_os = data['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    data_agg_is = data['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    # data_agg.columns = ['Accuracy Mean', 'Accuracy St Dev', 'Accuracy Median', 'Accuracy Min', 'Accuracy Max']
    data_time_agg = data['Microseconds'].agg([np.mean, np.std, np.median, np.min, np.max])
    # data_time_agg.columns = ['Time Mean', 'Time St Dev', 'Time Median', 'Time Min', 'Time Max']

    data_agg_comb = pd.concat([data_agg_os, data_agg_is, data_time_agg], axis=1, sort=False)
    data_agg_comb.to_csv(filename+'.stats.all.csv')

def get_charts(dir, filename):
    data = pd.read_csv(dir + '/' + filename+'.csv', names=['Attempt', 'Tree Type', 'Seed', 'Train Data Perc', 'Train Count', 'IS Error', 'OS Error', 'time', 'Options'])
    t = pd.Series(pd.to_timedelta(data['time'], unit="m")).dt
    data['Microseconds'] = t.seconds * 1000000 + t.microseconds
    group = data.groupby('Train Count')
    count_agg_train_is = group['IS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    count_agg_train_is.columns = ['IS Error Mean', 'IS Error St Dev', 'IS Error Median', 'IS Error Min', 'IS Error Max']

    count_agg_train_os = group['OS Error'].agg([np.mean, np.std, np.median, np.min, np.max])
    count_agg_train_os.columns = ['OS Error Mean', 'OS Error St Dev', 'OS Error Median', 'OS Error Min', 'OS Error Max']

    agg = pd.concat([count_agg_train_is, count_agg_train_os], axis=1, sort=False)
    agg.sort_values(['Train Count'])

    # plt.plot(agg['IS Error Mean'], agg['OS Error Mean'])
    plt.plot(agg['IS Error Mean'], 'r', agg['OS Error Mean'], 'b')
    plt.legend(['In Sample', 'Out Of Sample'])
    plt.xlabel('Training Sample Size')
    plt.ylabel('Mean Error Rate')
    plt.savefig(dir+'/'+filename+'_learning_rate.png')
    plt.clf()


def main():
    get_stats('dt-exports/winequality/x-results-comb')
    get_stats('dt-exports/winequality/cat-x-results-comb')
    get_stats('dt-exports/breast-cancer/x-results-comb')
    get_charts('dt-exports/winequality/x-results-comb')
    get_charts('dt-exports/winequality/cat-x-results-comb')
    get_charts('dt-exports/breast-cancer', 'd-results-decision-with-opts')
    get_charts('dt-exports/breast-cancer', 'd-results-ada-no-opts')
    get_charts('dt-exports/winequality', 'd-results-decision-with-opts')
    get_charts('dt-exports/winequality', 'd-results-ada-no-opts')
    get_charts('dt-exports/winequality', 'cat-d-results-decision-with-opts')
    get_charts('dt-exports/winequality', 'cat-d-results-ada-no-opts')


if __name__ == "__main__":
    main()