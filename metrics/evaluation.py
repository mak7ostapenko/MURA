import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score)


def mean_by_value(y_true, y_pred, group_value):
    """Compute mean of predictions grouped by group_value

    Arguments
        y_true : list or array
            True labels of images
        y_pred : list or array
            Predicted labels of images
        group_value : list or array
            Additional vaule about image

    Returns
        y_true :

        y_pred :

    """
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    group_value = np.array(group_value).reshape(-1)

    labels_frame = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group_value': group_value})

    y_true = labels_frame.groupby(['group_value'])['y_true'].mean().round()
    y_pred = labels_frame.groupby(['group_value'])['y_pred'].mean().round()
    return y_true, y_pred


def metrics_by_value(y_true, y_pred, sort_by_value=None):
    """Compute various metrics by value

    Arguments
        y_true : list or array
            True labels of images
        Y_pred : list or array
            Predicted labels of images
        metrics_by_value : list or array
            IDs of patient or study type

    Returns
        metrics_by_value : dict
            All computed metrics for patient

    """
    if  sort_by_value is not None:
        y_true, y_pred = mean_by_value(y_true, y_pred, sort_by_value)

    metrics_by_value = {}
    metrics_by_value['accuracy_score'] = accuracy_score(y_true, y_pred)
    metrics_by_value['f1_score'] = f1_score(y_true, y_pred)
    metrics_by_value['precision_score'] = precision_score(y_true, y_pred)
    metrics_by_value['recall_score'] = recall_score(y_true, y_pred)
    return metrics_by_value



