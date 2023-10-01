import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_fairness_metrics(dataset, predictions):
    # Extract true labels from the dataset
    true_labels = dataset['labels']

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Calculate disparate impact
    protected_group_indices = dataset['protected_group_indices']
    total_protected_group = len(protected_group_indices)
    total_non_protected_group = len(true_labels) - total_protected_group

    protected_group_predictions = predictions[protected_group_indices]
    non_protected_group_predictions = np.delete(predictions, protected_group_indices)

    protected_group_positive_rate = np.mean(protected_group_predictions)
    non_protected_group_positive_rate = np.mean(non_protected_group_predictions)

    disparate_impact = protected_group_positive_rate / non_protected_group_positive_rate

    # Calculate equal opportunity difference
    protected_group_true_positives = np.sum(np.logical_and(predictions == 1, true_labels == 1)[protected_group_indices])
    non_protected_group_true_positives = np.sum(np.logical_and(predictions == 1, true_labels == 1)) - protected_group_true_positives

    protected_group_positive_rate = protected_group_true_positives / total_protected_group
    non_protected_group_positive_rate = non_protected_group_true_positives / total_non_protected_group

    equal_opportunity_difference = protected_group_positive_rate - non_protected_group_positive_rate

    # Calculate statistical parity difference
    protected_group_positive_rate = np.mean(protected_group_predictions)
    non_protected_group_positive_rate = np.mean(non_protected_group_predictions)

    statistical_parity_difference = protected_group_positive_rate - non_protected_group_positive_rate

    return disparate_impact, equal_opportunity_difference, statistical_parity_difference

# Usage example
dataset = {
    'labels': np.array([0, 1, 1, 0, 1, 0, 1, 0]),
    'protected_group_indices': np.array([1, 2, 4, 6])
}

predictions = np.array([0, 1, 1, 0, 0, 1, 0, 1])

disparate_impact, equal_opportunity_difference, statistical_parity_difference = calculate_fairness_metrics(dataset, predictions)

print('Disparate Impact:', disparate_impact)
print('Equal Opportunity Difference:', equal_opportunity_difference)
print('Statistical Parity Difference:', statistical_parity_difference)
