"""Module for implementing metrics and optimizations"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_test, y_pred):

    """
    Plot a well-formatted confusion matrix for
    a classification prolem
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
                group_names.append('False ' + unique_values[i])
    
    # and labels
    group_counts =  ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(cf_matrix.shape)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    
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
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()