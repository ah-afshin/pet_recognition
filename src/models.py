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



class PetBreedsRecognitionTinyVGG(nn.Module):
    """
    A TinyVGG-style CNN model for multiclass classification of pet breeds (e.g 'American Bulldog', 'Basset Hound', 'Beagle', 'Birman', etc.).

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
            nn.Linear(fc_layer_hidden_units, 37) # there are 37 breeds of cats and dogs in database
        )

    def forward(self, x:t.Tensor) -> t.Tensor:
        z = self.conv_block_1(x)
        z = self.conv_block_2(z)
        y = self.fully_connected_layers(z)
        return y # this is the raw logit (before softmax activation func)



class PetBreedsRecognitionAlexNet(nn.Module):
    """
    Implementation of a simplified AlexNet architecture for the Oxford-IIIT Pet Breeds dataset.
    
    Architecture:
    - Input: RGB image (3×224×224)
    - 5 Convolutional layers (with ReLU, LRN, MaxPooling as in original paper)
    - 3 Fully Connected layers (with Dropout and ReLU)
    - Output: 37-class logits (no softmax)

    Differences from original AlexNet:
    - FC layers reduced from 4096→4096→1000 to 512→512→37
    - Preserves core design, suitable for fine-tuning or training from scratch on small datasets.
    """
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=(11, 11),
                stride=4
                # padding: None
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(
                size=5,
                alpha=1e-4,
                beta=0.75,
                k=1
            ),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=2
            )
        ) # [B, 96, 26, 26]

        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=(5, 5),
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.LocalResponseNorm(
                size=5,
                alpha=1e-4,
                beta=0.75,
                k=1
            ),
            nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=2
            )
        ) # [B, 256, 12, 12]

        self.conv_3 = nn.Sequential(
            nn.Conv2d(256, 384, (3,3), 1, 1),
            nn.ReLU()
        ) # [B, 384, 12, 12]
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 384, (3,3), 1, 1),
            nn.ReLU()
        ) # [B, 384, 12, 12]
        self.conv_5 = nn.Sequential(
            nn.Conv2d(384, 256, (3,3), 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((3,3), 2)
        ) # [B, 256, 6, 6]

        self.fc_6 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256*6*6, 512), # c_out=4096 in original model
            nn.ReLU()
        ) # [B, 512]
        self.fc_7 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512), # c_in=4096 and c_out=4096 in original model
            nn.ReLU()
        ) # [B, 512]
        self.fc_8 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 37) # c_in=4096 and c_out=1000 in original model
            # output of our neural net is for 37 classes of Oxford-IIIT Pets dataset
        ) # [B, 37] -> logits

        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        fm = self.conv_1(x)
        fm = self.conv_2(fm)
        fm = self.conv_3(fm)
        fm = self.conv_4(fm)
        fm = self.conv_5(fm)

        z  = self.adaptive_pool(fm)
        z = self.fc_6(z)
        z = self.fc_7(z)
        y = self.fc_8(z)

        return y # this is the raw logit (before softmax activation func)
