from typing import Any, List

from util import LANGUAGES


def accuracy_score(y_true: List[Any], y_pred: List[Any]) -> float:
    """
    Compute the accuracy given true and predicted labels

    Args:
        y_true (List[Any]): true labels
        y_pred (List[Any]): predicted labels

    Returns:
        float: accuracy score
    """
    
    # Calculate accuracy
    correct_count = 0

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_count += 1

    total_count = len(y_true)
    accuracy = correct_count / total_count
    return accuracy

    #raise NotImplementedError


def confusion_matrix(y_true: List[Any], y_pred: List[Any], labels: List[Any]) \
    -> List[List[int]]:
    """
    Builds a confusion matrix given predictions
    Uses the labels variable for the row/column order

    Args:
        y_true (List[Any]): true labels
        y_pred (List[Any]): predicted labels
        labels (List[Any]): the column/rows labels for the matrix

    Returns:
        List[List[int]]: the confusion matrix
    """
    # check that all of the labels in y_true and y_pred are in the header list
    
   # Initialize confusion matrix with zeros
    num_labels = len(labels)
    confusion_mat = [[0] * num_labels for _ in range(num_labels)]
    
    # Populate confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = labels.index(true_label)
        pred_index = labels.index(pred_label)
        confusion_mat[pred_index][true_index] += 1
    
    return confusion_mat