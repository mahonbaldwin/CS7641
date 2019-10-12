import util.util as util
import pandas as pd

import csv

# will take the wine and find unique rows (but for the Y) and average the Y

def main2():
    filename = '../resources/winequality/winequality-combined.csv'
    data = util.get_data(filename)
    reviews = []
    for i in data['quality']:
        if i >= 1 and i <= 3:
            reviews.append('1')
        elif i >= 4 and i <= 7:
            reviews.append('2')
        elif i >= 8 and i <= 10:
            reviews.append('3')
    data['Reviews'] = reviews
    data.to_csv('../resources/winequality/winequality-combined-cat.csv')


def main():
    filename = '../resources/winequality/winequality-combined.csv'
    # filename = '../resources/winequality/dummy-data.csv'
    exclude_column = 'quality'
    # exclude_column = 'stuff3'
    data = util.get_data(filename)
    duplicated = data.loc[:, data.columns != exclude_column].duplicated()
    print(duplicated.value_counts())
    print(data.duplicated().value_counts())
    data = data.sort_values(by=list(data.columns))
    return duplicated


if __name__ == "__main__":
    main2()