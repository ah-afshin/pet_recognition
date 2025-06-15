import torch as t
from torch import nn
from torch.utils.data import DataLoader

from .utils import binary_accuracy_func, multiclass_accuracy_func


def test_species_recog_model(
        model: nn.Module,
        test_dl: DataLoader,
        device: str
    ) -> None:
    """
    Evaluate the trained binary classification model on test dataset.

    Args:
        model (nn.Module): The trained neural network model to test.
        test_dl (DataLoader): PyTorch DataLoader for test dataset.
        device (str): Device to use for evaluation ('cpu' or 'cuda').

    Returns:
        None: Prints average loss and accuracy.
    """
    loss_func = nn.BCEWithLogitsLoss()
    
    batch_num = 0
    accuracy_sum = 0
    loss_sum =0

    model.eval()
    with t.inference_mode():

        for batch in test_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y = y.unsqueeze(1).float() # y shape from [B] to [B, 1]

            y_pred_logit = model(x)
            y_pred_prob = t.sigmoid(y_pred_logit)
            y_pred_label = (y_pred_prob > 0.5).float()

            # debug
            # if batch_num==0: print(y_pred_label)

            batch_num += 1
            loss_sum += loss_func(y_pred_logit, y).item()
            accuracy_sum += binary_accuracy_func(y, y_pred_label)
    
    av_loss = loss_sum/batch_num
    av_acc = accuracy_sum/batch_num
    print("\n\tloss: ", av_loss)
    print("\taccuracy: ", av_acc)


def test_breeds_recog_model(
        model: nn.Module,
        test_dl: DataLoader,
        device: str
    ) -> None:
    """
    Evaluate the trained multiclass classification model on test dataset.

    Args:
        model (nn.Module): The trained neural network model to test.
        test_dl (DataLoader): PyTorch DataLoader for test dataset.
        device (str): Device to use for evaluation ('cpu' or 'cuda').

    Returns:
        None: Prints average loss and accuracy.
    """
    loss_func = nn.CrossEntropyLoss()
    
    batch_num = 0
    accuracy_sum = 0
    loss_sum =0

    model.eval()
    with t.inference_mode():

        for batch in test_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y = y.long() # shape [B]

            y_pred_logit = model(x)
            y_pred_prob = t.softmax(y_pred_logit, dim=1)
            y_pred_label = t.argmax(y_pred_prob, dim=1)

            # debug
            # if batch_num==0: print(y_pred_label)

            batch_num += 1
            loss_sum += loss_func(y_pred_logit, y).item()
            accuracy_sum += multiclass_accuracy_func(y, y_pred_label)
    
    av_loss = loss_sum/batch_num
    av_acc = accuracy_sum/batch_num
    print("\n\tloss: ", av_loss)
    print("\taccuracy: ", av_acc)
