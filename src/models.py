import torch as t
from torch import nn



class PetSpeciesRecognitionTinyVGG(nn.Module):
    """
    A TinyVGG-style CNN model for binary classification of pet species (dog vs. cat).

    Args:
        feature_maps (int): Number of output channels for conv layers.
        fc_layer_hidden_units (int): Number of hidden units in fully-connected layers.
        input_shape (int, optional): Number of input channels. Defaults to 3 (RGB images).

    Note:
        - The model outputs raw logits. Use nn.BCEWithLogitsLoss for training.
        - Input images must be resized to a fixed size (e.g. 224x224).
    """
    def __init__(self,
                feature_maps: int,
                fc_layer_hidden_units: int,
                input_shape: int = 3,
        ) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                input_shape,
                feature_maps,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                feature_maps,
                feature_maps,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                feature_maps,
                feature_maps,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                feature_maps,
                feature_maps,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2
            )
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps*3136, fc_layer_hidden_units),
            nn.ReLU(),
            nn.Linear(fc_layer_hidden_units, fc_layer_hidden_units),
            nn.ReLU(),
            nn.Linear(fc_layer_hidden_units, 1)
        )


    def forward(self, x:t.Tensor) -> t.Tensor:
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        y = self.fully_connected_layers(z)
        return y # this is the raw logit (before sigmoid activation func)
