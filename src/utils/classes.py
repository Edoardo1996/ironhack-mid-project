import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
from tqdm import tqdm


class LogisticRegressionWithThreshold(LogisticRegression):
    """
    Logistic Regression classifier where threshold could be change manually
    or calculated with user-defined optimizations.
    If no treshold value is passed, it behaves as the Logistic Regression 
    Classifier.
    Refer to Logistic Regression Classifier from scikit-learn for additional 
    documentation:
     <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
    """
    def predict(self, X, threshold: float=None):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        threshold : float, default = None
            Treshold parameter for the classifier.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        y_pred_with_threshold : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample with modified 
            threshold
        """
        
        if threshold == None:  # If no threshold passed in, simply call the base class predict method
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= threshold).astype(int)

            return y_pred_with_threshold

    def threshold_from_optimal_tpr_minus_fpr(self, X, y):
        """
        Calculate optimal threshold for maximum TPR - FPR

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        y : {array-like}
            True target values (test values)
            
        Returns
        -------
        threshold: float
            Optimised threshold fot TPR - FPR
        tpr-fpr: float
            Maximised tpr-fpr rateo
        """
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        fpr, tpr, thresholds = roc_curve( # replace if target class is categorical
            y.replace({'No': 0, 'Yes': 1}), y_scores)

        optimal_idx = np.argmax(tpr-fpr)

        return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx]

    def threshold_from_optimal_f_score(self, X, y):
        """
        Calculate optimal threshold to maximize f1-score on label 0.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        y : {array-like}
            True target values (test values)
            
        Returns
        -------
        threshold: float
            Optimised threshold for f1-score on label 0 TODO: implement label-1
        f1_score: float
            Maximised f1-score for label 0
        """
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(
            y.replace({'No': 0, 'Yes': 1}), y_scores)

        fscores = (2 * precisions * recalls) / (precisions + recalls)

        optimal_idx = np.argmax(fscores)

        return thresholds[optimal_idx], fscores[optimal_idx]

    def threshold_from_cost_function(self, X, y, cost_function, n_customers, 
                                        customer_acceptance_gain, email_cost):
        """
        Calculate optimal threshold to maximize (or minimize) a custom cost or
        revenue function.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.
        y : {array-like}
            True target values (test values)
        cost_function: callable
            Custom function that maximizes (or minimize) a cost or a revenue
            
        Returns
        -------
        threshold: float
            Optimised threshold for custom function
        max_cost: float
            Maximum revenue achievable
        df_cost: pandas.DataFrame
            DataFrame with metrics for different thresholds
        """
        y = y.replace({'No': 0, 'Yes': 1}) # replace if categorical
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        costs = []

        for threshold in tqdm(thresholds):
            y_pred_with_threshold = (y_scores >= threshold).astype(int)
            costs.append(cost_function(y, y_pred_with_threshold,
                         n_customers, customer_acceptance_gain, email_cost))

        df_cost = pd.DataFrame(
            {'precision': precisions[:-1], 'recall': recalls[:-1], 
             'threshold': thresholds, 'cost': costs})
        max_cost = df_cost['cost'].max()
        threshold = df_cost[df_cost['cost'] == max_cost].iloc[0]['threshold']

        return threshold, max_cost, df_cost
