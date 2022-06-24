from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import numpy as np
import pandas as pd


class LogisticRegressionWithThreshold(LogisticRegression):
    def predict(self, X, threshold=None):
        if threshold == None:  # If no threshold passed in, simply call the base class predict, effectively threshold=0.5
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= threshold).astype(int)

            return y_pred_with_threshold

    def threshold_from_optimal_tpr_minus_fpr(self, X, y, pos_label=1):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y.replace({'No': 0, 'Yes': 1}), y_scores)

        optimal_idx = np.argmax(tpr-fpr)

        return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx]

    def threshold_from_optimal_f_score(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y.replace({'No': 0, 'Yes': 1}), y_scores)

        fscores = (2 * precisions * recalls) / (precisions + recalls)
        
        optimal_idx = np.argmax(fscores)
        
        return thresholds[optimal_idx], fscores[optimal_idx]

    def threshold_from_cost_function(self, X, y, cost_function, n_customers, customer_acceptance_gain, email_cost):
        y = y.replace({'No': 0, 'Yes': 1})
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        costs = []

        for threshold in tqdm(thresholds):
            y_pred_with_threshold = (y_scores >= threshold).astype(int)
            costs.append(cost_function(y, y_pred_with_threshold, n_customers, customer_acceptance_gain, email_cost))

        df_cost = pd.DataFrame({'precision':precisions[:-1], 'recall':recalls[:-1], 'threshold':thresholds, 'cost':costs})
        max_cost = df_cost['cost'].max()
        threshold = df_cost[df_cost['cost']==max_cost].iloc[0]['threshold']

        return threshold, max_cost, df_cost