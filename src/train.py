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
        learning_rate: float,
        weight_decay: float = 0, # L2 regularization factor
        schedule_lr_step: int = None,
        schedule_lr_gamma: float = None
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
        
        weight_decay (float) default= 0: L2 regularization factor         (Optional)
        schedule_lr_step (int) default= None                              (Optional)
        schedule_lr_gamma (float) default= None                           (Optional)
    Returns:
        None
    """
    loss_func = nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if schedule_lr_step is not None and schedule_lr_gamma is not None:
        t.optim.lr_scheduler.StepLR(optim, schedule_lr_step, schedule_lr_gamma)

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


def train_features_extracted_breeds_recog_model(
        model: nn.Module,
        train_dl: DataLoader,
        device: str,
        epochs: int,
        learning_rate: float
    ) -> None:
    """
    Train the head of the given pre-trained model
    using CrossEntropyLoss and provided training data.

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
    optim = t.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # only train params that require gradient
        lr=learning_rate
    )

    model.train()
    for epoch in range(epochs):
        total_loss_of_epoch = 0

        for batch in train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
                        
            y = y.long() # to be LongTensor with shape [B]
            
            y_pred_logit = model(x) # shape [B, 37]
            loss = loss_func(y_pred_logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss_of_epoch += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_loss_of_epoch}")


def train_fine_tune_breeds_recog_model(
        model: nn.Module,
        train_dl: DataLoader,
        device: str,
        epochs: int,
        learning_rate: float,
        fine_tune_at_epoch: int = 5,
        depth: int = 4
    ) -> None:
    """
    Train the head of the given pre-trained model
    using CrossEntropyLoss and provided training data.

    Args:
        model (nn.Module): The neural network model to train.
        train_dl (DataLoader): PyTorch DataLoader containing training data.
        device (str): Device to use for training ('cpu' or 'cuda').
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        fine_tune_at_epoch (int): at which epoch does the fine tuning actually beggins.
        depth (int): how many layers do we want to tune.
        
    Returns:
        None
    """
    loss_func = nn.CrossEntropyLoss()
    optim = t.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # only train params that require gradient
        lr=learning_rate
    )

    model.train()
    for epoch in range(epochs):
        if epoch==fine_tune_at_epoch:
            model.open_feature_extractor(depth)
            print("finetuning started")

        total_loss_of_epoch = 0
        for batch in train_dl:
            x, y = batch
            x, y = x.to(device), y.to(device)
                        
            y = y.long() # to be LongTensor with shape [B]
            
            y_pred_logit = model(x) # shape [B, 37]
            loss = loss_func(y_pred_logit, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss_of_epoch += loss.item()
        print(f"Epoch {epoch+1} | loss: {total_loss_of_epoch}")
