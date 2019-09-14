# Code location
All files are located in the projects repository.

# Code Requirements
The code requires the pyhton packages from `requirements.txt` to be installed.
To install them you'll need to have python3 and pip 3 installed using the Anaconda environment.
Install the `requirements.txt` packages by entering the following:

    pip install -r requirements.txt


# Decision Trees (including Ada Boost)
SciKit Learn `DecisionTreeClassifier` and `AdaBoostClassifier`
The main code is located in `decision-tree.py` and it references the other code as needed.
* To run it, type `python3 decision-tree.py`. (this will use 1000 different seeds to create various trees, to change this update the `main()` function to iterate over the desired number.
* Several export files are created and are put in `dt-exports/winequality` for the wine data and `dt-exports/breast-cancer` for the breast cancer data.
  * within each of these directories there are a few files generated, the ones you probably want are named `x-results-comb.csv` which contains all of the results regardless of options passed to create the tree or the type of tree that was created.
* The code that generates aggregated statistics is located in `tree-aggregates.py` and can be invoked via `python3 tree-aggregates.py`.
  * This generates two files in the same exports directory named `x-results-comb.stats.csv` and `x-results-comb.stats.all.csv` that contains aggregate data for each tree type and all tree types respectively.

Citation: Much of the code was borrowed from https://www.datacamp.com/community/tutorials/decision-tree-classification-python.


# Neural Nets
SciKit Learn `MLPClassifier`
The main code is located in `neural-network.py` and it references the other code as needed.
* To run it, type `python3 neural-network.py`. (this will use 1000 different seeds to create various neural nets, to change this update the `main()` function to iterate over the desired number.
* Several export files are created and are put in `nn-exports/wine` for the wine data and `nn-exports/breast_cancer` for the breast cancer data.
  * within each of these directories there are a few files generated, the ones you want are named `x-results-comb.csv`.
* The code that generates aggregated statistics is located in `nn-aggregates.py` and can be invoked via `python3 nn-aggregates.py`.
  * This generates two files in the same exports directory named `x-results-comb.stats.csv` and `x-results-comb.stats.all.csv` that contains aggregate data for each net shape and all net shapes respectively.


# Support Vector Machines
SciKit learn `svm.SVC`
The main code is located in `svm.py` and it references the other code as needed.
* To run it, type `python3 svm.py`. (This will use 10 different seeds to creat various SVMs, to change this update the `main()` function to iterate over the desired number.
* Several export files are created and are put in `svm-exports/wine` for the wine data and `svm-exports/breast_cancer` for the breast cancer data.
  * within each of these directories there are a few files generated, the ones you want are named `c-results-comb.csv`.
* The code that generates aggregated statistics is located in `svm-aggregates.py` and can be invoked via `python3 svm-aggregates.py`.
  * This generates two files in the same exports directory named `c-results-svm.stats.csv` and `c-results-svm.stats.all.csv` that contains aggregate data for each kernel.

# KNN
SciKit learn `KNeighborsClassifier`.
The main code is locted in `knn.py` and it references the other code as needed.

* To run it, type `python3 knn.py`. (This will use 1000 different seeds to creat various KNNs, to change this update the `main()` function to iterate over the desired number.
* Several export files are created and are put in `knn-exports/wine` for the wine data and `knn-exports/breast_cancer` for the breast cancer data.
  * within each of these directories there are a few files generated, the ones you want are named `b-results-comb.csv`.
* The code that generates aggregated statistics is located in `knn-aggregates.py` and can be invoked via `python3 knn-aggregates.py`.
  * This generates two files in the same exports directory named `b-results-svm.stats.csv` and `b-results-svm.stats.all.csv` that contains aggregate data for each value of `k`.
