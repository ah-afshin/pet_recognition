import torch as t



def binary_accuracy_func(
        y_true: t.Tensor,
        y_pred: t.Tensor
    ) -> float:
    """
    Compute binary classification accuracy given logits and true labels.

    Args:
        y_true (Tensor): Tensor of shape [B, 1] with true binary labels (0 or 1).
        y_pred_logit (Tensor): Tensor of shape [B, 1] with predicted labels (model outputs).

    Returns:
        float: Accuracy value between 0 and 1.
    """
    correct = (y_pred == y_true).float()
    accuracy = correct.mean().item()
    
    return accuracy
