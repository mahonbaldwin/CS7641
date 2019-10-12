import matplotlib.pyplot as plt


def bar_chart(file_name, data, column):
    groups = data.groupby(column).count()
    categories = groups.index.values
    first_key = groups.keys()[0]
    values = groups[first_key]
    plt.bar(categories, values)
    # plt.legend(labels)
    plt.savefig(file_name, bbox_inches='tight')

    plt.clf()
