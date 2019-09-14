import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from sklearn import tree


def bar_chart(file_name, data, column):
    groups = data.groupby(column).count()
    categories = groups.index.values
    first_key = groups.keys()[0]
    values = groups[first_key]
    plt.bar(categories, values)
    # plt.legend(labels)
    plt.savefig(file_name, bbox_inches='tight')

    plt.clf()


def save_tree_chart(filename, classifier):
    if filename is not None:
        print(filename)
        dot_data = StringIO()
        tree.export_graphviz(classifier, out_file=dot_data)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())
        graph.write_svg(filename)