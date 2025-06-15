import torch as t
from torch import nn
from torch.utils.data import DataLoader



def train_species_recog_model(
        model: nn.Module,
        train_dl: DataLoader,
        device: str,
        epochs: int,
        learning_rate: float
    ) -> None:
    """
    Train the given binary classification model using BCEWithLogitsLoss
    and provided training data.

    Args:
        model (nn.Module): The neural network model to train.
        train_dl (DataLoader): PyTorch DataLoader containing training data.
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        None
    """
    loss_func = nn.BCEWithLogitsLoss()
    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    # debug
    # a = True

    model.train()
    for epoch in range(epochs):
        total_loss_of_epoch = 0

        for batch in train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # debug
            # if a: print(y); a=False
            
            y = y.unsqueeze(1).float() # y shape from [B] to [B, 1]

            y_pred_logit = model(x)
            loss = loss_func(y_pred_logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss_of_epoch += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_loss_of_epoch}")


def train_breeds_recog_model(
        model: nn.Module,
        train_dl: DataLoader,
        device: str,
        epochs: int,
        learning_rate: float
    ) -> None:
    """
    Train the given multiclass classification model using CrossEntropyLoss
    and provided training data.

    Args:
        model (nn.Module): The neural network model to train.
        train_dl (DataLoader): PyTorch DataLoader containing training data.
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        
    Returns:
        None
    """
    loss_func = nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=learning_rate)

    # debug
    # a = True

    model.train()
    for epoch in range(epochs):
        total_loss_of_epoch = 0

        for batch in train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            # debug
            # if a: print(y); a=False
            
            y = y.long() # to be LongTensor with shape [B]
            
            y_pred_logit = model(x) # shape [B, 37]
            loss = loss_func(y_pred_logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss_of_epoch += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_loss_of_epoch}")
