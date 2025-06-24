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



class ResidualBlock(nn.Module):
    """
    A basic residual block for ResNet-like architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (feature maps).
        stride (int): Stride used in the first convolution. Default is 1.

    Structure:
        - Two 3x3 convolutional layers with BatchNorm and ReLU.
        - Optional downsampling of identity (skip connection) if shape mismatch.
        - Final addition of F(x) and x followed by ReLU.
    """
    def __init__(
            self,
            in_channels: int, # input shape (last blocks features)
            feature_maps: int, # out channels
            stride: int = 1 # reduce size
        ):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=feature_maps,
                kernel_size=(3,3),
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_maps,
                out_channels=feature_maps,
                kernel_size=(3,3),
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps)
        )

        # prevent F(x) and x mismatch after shortcut
        if (stride != 1) or (in_channels != feature_maps):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, feature_maps, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(feature_maps)
            )
        else:
            self.downsample = None
        
        self.final_activation = nn.ReLU(inplace=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        out = self.block_2(self.block_1(x)) # F(x)
        if self.downsample:
            x = self.downsample(x)
        y = self.final_activation(out + x) # shortcut
        return y # this is a tensor. it contains feature maps.



class PetBreedsRecognitionResNet9_v1(nn.Module):
    """
    Implementation of a simple ResNet architecture for the Oxford-IIIT Pet Breeds dataset.
    
    Architecture:
    - Input: RGB image (3×224×224)
    - 1 Convolutional layer (with BN, ReLU, MaxPooling for preprocess)
    - 3 Residual basic blocks, arch: [( (conv → BN → ReLU → conv → BN) + x ) → ReLU]
    - 1 Classifier (AdaptiveAvgPooling and Linear)
    - Output: 37-class logits (no softmax)
    """
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 37)
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.preprocess(x)
        z = self.residual_blocks(z)
        y = self.classifier(z)
        return y # this is the raw logit (before softmax activation func)



class PetBreedsRecognitionResNet9_v2(nn.Module):
    """
    Implementation of a simple ResNet architecture for the Oxford-IIIT Pet Breeds dataset.
    this one has a dropout layer.

    Architecture:
    - Input: RGB image (3×224×224)
    - 1 Convolutional layer (with BN, ReLU, MaxPooling for preprocess)
    - 3 Residual basic blocks, arch: [( (conv → BN → ReLU → conv → BN) + x ) → ReLU]
    - 1 Classifier (AdaptiveAvgPooling, dropout and Linear)
    - Output: 37-class logits (no softmax)
    """
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 256),
            ResidualBlock(256, 512)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 37)
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.preprocess(x)
        z = self.residual_blocks(z)
        y = self.classifier(z)
        return y # this is the raw logit (before softmax activation func)



class PetBreedsRecognitionResNet9_v3(nn.Module):
    """
    Implementation of a simple ResNet architecture for the Oxford-IIIT Pet Breeds dataset.
    smaller than v1, but the same architecture. less chance for overfitting.
    
    Architecture:
    - Input: RGB image (3×224×224)
    - 1 Convolutional layer (with BN, ReLU, MaxPooling for preprocess)
    - 3 Residual basic blocks, arch: [( (conv → BN → ReLU → conv → BN) + x ) → ReLU]
    - 1 Classifier (AdaptiveAvgPooling and Linear)
    - Output: 37-class logits (no softmax)
    """
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 48),
            ResidualBlock(48, 48),
            ResidualBlock(48, 48)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, 37)
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.preprocess(x)
        z = self.residual_blocks(z)
        y = self.classifier(z)
        return y # this is the raw logit (before softmax activation func)



class DepthwiseSeparableConvBlock(nn.Module):
    """
    Depthwise Separable Convolution Block.

    This block replaces a standard convolution with a depthwise convolution
    followed by a pointwise convolution, significantly reducing the number
    of parameters and computations.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Stride for the depthwise convolution (default: 1).

    Shape
    -----
    Input: (N, in_channels, H, W)
    Output: (N, out_channels, H_out, W_out)
    """
    def __init__(self, in_channels:int, out_channels:int, stride:int):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3,3), stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1,1), stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.pointwise(
            self.depthwise(x)
        )



class PetBreedsRecognitionMobileNet_v1(nn.Module):
    """
    A lightweight CNN for pet breed classification using depthwise separable convolutions.

    This is a MobileNet-style architecture designed for low-resource environments.
    It includes a stem convolution, a sequence of DepthWiseSeparableConvBlock units,
    and a classification head with global average pooling.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 37 for Oxford Pet dataset).

    Shape
    -----
    Input: (N, 3, H, W)
    Output: (N, num_classes)
    """
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.features = nn.Sequential(
            DepthwiseSeparableConvBlock(32, 64, 1),
            DepthwiseSeparableConvBlock(64, 128, 2),
            DepthwiseSeparableConvBlock(128, 128, 1),
            DepthwiseSeparableConvBlock(128, 256, 2),
            DepthwiseSeparableConvBlock(256, 256, 1),
            DepthwiseSeparableConvBlock(256, 512, 2),
            
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            DepthwiseSeparableConvBlock(512, 512, 1),
            
            DepthwiseSeparableConvBlock(512, 1024, 2),
            DepthwiseSeparableConvBlock(1024, 1024, 1)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024, 37)
        )
    
    def forward(self, x:t.Tensor) -> t.Tensor:
        z = self.preprocess(x)
        fm = self.features(z)
        y = self.classifier(fm)
        return y



class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block used in MobileNetV2.

    This block consists of:
    1. Pointwise expansion convolution
    2. Depthwise convolution
    3. Pointwise projection convolution
    A residual connection is added if the input and output shapes match
    and stride is 1.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    expansion_factor : int
        Factor to expand the number of channels before depthwise convolution.
    stride : int
        Stride for depthwise convolution.
    out_channels : int, optional
        Number of output channels. Defaults to in_channels.

    Shape
    -----
    Input: (N, in_channels, H, W)
    Output: (N, out_channels, H_out, W_out)
    """
    def __init__(
            self,
            in_channels: int,
            expansion_factor: int,
            stride: int,
            out_channels: int = None
        ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        expanded_channels = in_channels*expansion_factor
        self.size_match = (in_channels == out_channels) and (stride == 1)

        self.expantion = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, (3,3), stride=stride,
                      padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.expantion(x)
        z = self.depthwise(z)
        y = self.projection(z)

        if self.size_match:
            return x + y
        return y



class PetBreedsRecognitionMobileNet_v2(nn.Module):
    """
    Lightweight CNN for pet breed classification based on MobileNetV2.

    This model uses:
    - A standard initial convolution (stem)
    - A sequence of inverted residual blocks with varying expansions and strides
    - A classifier head with adaptive average pooling and a linear layer

    Designed for efficient training and inference on lightweight devices.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default: 37 for Oxford Pet dataset).

    Shape
    -----
    Input: (N, 3, H, W)
    Output: (N, num_classes)
    """
    def __init__(self, num_classes: int = 37):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6()
        )

        # (in_channels, out_channels, expansion, stride, repeat)
        config = [
            (16, 24, 1, 2, 1),
            (24, 32, 6, 2, 2),
            (32, 64, 6, 2, 2),
            (64, 96, 6, 1, 1),
            (96, 128, 6, 2, 1),
        ]

        layers = []
        for in_c, out_c, exp, stride, reps in config:
            for i in range(reps):
                layers.append(InvertedResidualBlock(
                    in_channels=in_c if i == 0 else out_c,
                    out_channels=out_c,
                    expansion_factor=exp,
                    stride=stride if i == 0 else 1
                ))
        self.blocks = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
