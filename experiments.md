# Experiments and Results
This note contains the experiments, their details, and the results.
---
## Experiment 1: BaseLine Binary Model
### Details
- Model
    - type: tiny VGG (2 conv blocks, 3 fc layer)
    - conv architecture: [conv(k=3) -> ReLU]*2 -> max_pool(2)
    - feature map units: 6 channels
    - fc layer hidden units: 10
- Data
    - labels: species (0,1) cat vs. dog
    - preprocess: resize(224,224), random_hflip, ToTensor
    - train-test ratio: 80-20
    - batch size (B): 32
- Training
    - loss func: BCEWithLogitsLoss
    - optimizer: Adam
    - epochs: 5
    - learning rate (LR): 0.05
### Results
using device: cuda
Epoch 1 | loss: 130.77849334478378
Epoch 2 | loss: 116.23035258054733
Epoch 3 | loss: 116.03492730855942
Epoch 4 | loss: 116.13255050778389
Epoch 5 | loss: 116.17660504579544

        loss:  0.6244820123133452
        accuracy:  0.6850543488626895

        execution time: 849.288964509964s
### note:
The experiment was repeated due to issues in the accuracy function.
---
## Experiment 2: Binary Model deeper and longer training
### Details
- Model
    - type: tiny VGG (2 conv blocks, 3 fc layer)
    - conv architecture: [conv(k=3) -> ReLU]*2 -> max_pool(2)
    - feature map units: 12 channels
    - fc layer hidden units: 32
- Data
    - labels: species (0,1) cat vs. dog
    - preprocess: resize(224,224), random_hflip, ToTensor, Normalization(0.5, 0.5)
    - train-test ratio: 80-20
    - batch size (B): 32
- Training
    - loss func: BCEWithLogitsLoss
    - optimizer: Adam
    - epochs: 7
    - learning rate (LR): 1e-3
### Results
using device: cuda
Epoch 1 | loss: 112.81414380669594
Epoch 2 | loss: 103.01598006486893
Epoch 3 | loss: 94.11443668603897
Epoch 4 | loss: 87.53179979324341
Epoch 5 | loss: 79.0181078016758
Epoch 6 | loss: 71.16721779108047
Epoch 7 | loss: 60.89886475354433

        loss:  0.5400613086379092   
        accuracy:  0.776177537182103

        execution time: 977.5023243427277s
---
...