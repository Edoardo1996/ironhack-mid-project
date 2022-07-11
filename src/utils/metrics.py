"""
Module for implementing metrics and optimizations
"""
import itertools
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_test, y_pred) -> None:
    """
    Plot a well-formatted confusion matrix for
    a classification prolem results.

    Parameters:
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values as returned by a classifier.
    """
    # confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    # Generate group names
    unique_values = y_test.value_counts().index.tolist()
    id_mat = np.identity(len(unique_values))
    group_names = []
    for i in range(len(id_mat)):
        for j in range(len(id_mat)):
            if id_mat[i, j] == 1:
                group_names.append('True ' + unique_values[i])
            else:
                group_names.append('False ' + unique_values[j])

    # and labels
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(
        value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(
        group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cf_matrix.shape)

    # show plot TODO: export plot
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


def plot_multiclass_roc(clf, X_test, y_test, n_classes: int,
                        figsize=(17, 6)) -> None:
    """
    Plot the Compute Receiver operating characteristic (ROC).
    Note: this implementation is restricted to the binary classification task.

    Parameters:
    ----------
    clf: classifier
        Classifier used to estimate the target values
    X_test: matrix-like
        Features used for testing.
    y_test: array-like
        Correct values for target
    n_classes: int
        Number of classes of target
    figsize: tuple, default = (17, 6)
        Size of the figure as (x, y)
    """
    y_score = clf.predict_proba(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (
            roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


def plot_params_metric(params: OrderedDict, metric, report, figsize, hue) -> None:
    """
    Plot metric variablity along changes of parameters optimization.
    It should not be called directly.

    Parameters:
    ----------
    Params: OrderedDict
    metric: Callable
    report: pandas.DataFrame
        metric report. 
    figsize: tuple
        Size of the figure as (x, y)
    hue: str
        Grouping variable that will produce points with different colors.
    """
    assert len(params.keys()) <= 2, \
        'More than two params handling is not implemented yet'
    plt.figure(figsize=figsize)
    # get x label (param different from set hue)
    x_label = [param for param in params if param != hue][0]
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(metric.__name__, fontsize=16)
    if not hue:
        # if hue has been set, we have to loop on its parameters.
        hue = x_label
        # set colors
    for param in report[hue].unique():  # loop on not-hue param
        if len(params.keys()) == 1:
            plt.plot(report[x_label].astype('str'), report[metric.__name__],
                     color='blue', linestyle='dashed',
                     marker='o', markerfacecolor='red', markersize=10)
        else:
            plt.plot(report.sort_values(metric.__name__)[x_label].loc[report[hue] == param].astype('str'),
                     report.sort_values(metric.__name__)[
                metric.__name__].loc[report[hue] == param],
                linestyle='dashed', marker='o', markerfacecolor='red', markersize=10, label=param)
            plt.legend()
    plt.grid()


def report_metrics(params: OrderedDict, metric, all_true, all_pred, label=None,
                   sort_by_metric=False, show_plot=False, figsize=(10, 6),
                   hue=None) -> pd.DataFrame:
    """
    Report designated metric for model running with different params.
    TODO: Separate this function in sub-functions
    TODO: Error handling
    TODO: Refactoring the function
    TODO: Add time of computations
    Parameters:
    ----------
    params: OrderedDict
    metric: Callable
    all_true: array-like
        Array-like with all the real values for target for each param
    all_pred: array-like
        Array-like with all the predicted values for target for each param
    label: str, default=None
        Label for the recall calculation
    sort_by_metric: bool, default=False
        If True, metric will be sorted with descending order
    show_plot: bool, default=False
        If True, plot of the metrics will be shown
    figsize: tuple, default = (10, 6)
        Size of the figure as (x, y)
    hue: str, default=None
        Grouping variable that will produce points with different colors.

    Returns:
    -------
    report: pandas.DataFrame
        Report with the metric and params.
    """
    # assert consistency for params and test-pred
    if len(all_true) != len(all_pred):
        raise Exception('Predictions and test are not consistent')
    if np.prod([len(values) for values in params.values()]) != len(all_pred):
        raise Exception('Params not consistent with test and predictions')

    # calculate metric values
    metric_values = []
    for true, pred in zip(all_true, all_pred):
        if metric.__name__ == 'recall_score':
            assert label, 'Trying to get recall score without defining a label'
            if isinstance(pred, np.ndarray):
                metric_values.append(metric(true, pred, pos_label=label))
            else:
                metric_values.append(np.nan)
        else:
            if isinstance(pred, np.ndarray):
                metric_values.append(metric(true, pred))
            else:
                metric_values.append(np.nan)

    # Create report
    cols = list(params.keys()) + [metric.__name__]
    # cartesian product of params
    all_labels = np.array(list(itertools.product(*params.values())))
    data = np.concatenate((all_labels,
                           np.array(metric_values).reshape(len(metric_values), 1)),
                          axis=1)
    if sort_by_metric:
        report = pd.DataFrame(columns=cols, data=data).sort_values(
            by=metric.__name__, ascending=False).reset_index(drop=True)
    else:
        report = pd.DataFrame(columns=cols, data=data)

    # Plot if required by user
    if show_plot:
        plot_params_metric(params, metric, report, figsize, hue)

    return report


def calculate_revenue(y_test, y_pred,
                      n_customers, customer_acceptance_gain, action_cost):
    """
    Calculate revenue based on results of a classification model.
    Case scenario is a company that has a certain numbers of customers, sends 
    the offer to the identified "Yes" or "1" customer and it will get a revenue 
    based on the quality on the model.

    Parameters:
    ----------
    
    """
    if pd.Series(y_test).unique().dtype in [int, 'int32', 'int64']:
        y_test = pd.Series(y_test).replace({0: 'No', 1: 'Yes'})
    if pd.Series(predictions).unique().dtype in [int, 'int32', 'int64']:
        predictions = pd.Series(predictions).replace({0: 'No', 1: 'Yes'})
    
    confusion = confusion_matrix(y_test, predictions)
    TP = confusion[1, 1]*n_customers/(len(y_test)) # True positive
    FP = confusion[0, 1]*n_customers/(len(y_test)) # False positive
    FN = confusion[1, 0]*n_customers/(len(y_test)) # False negative
    revenue = - email_cost*(TP+FP) - customer_acceptance_gain*(FN) + customer_acceptance_gain*(TP)
    return revenue
