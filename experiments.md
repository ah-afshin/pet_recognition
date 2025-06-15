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
```
using device: cuda
Epoch 1 | loss: 130.77849334478378
Epoch 2 | loss: 116.23035258054733
Epoch 3 | loss: 116.03492730855942
Epoch 4 | loss: 116.13255050778389
Epoch 5 | loss: 116.17660504579544

        loss:  0.6244820123133452
        accuracy:  0.6850543488626895

        execution time: 849.288964509964s
```

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
```
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
```

### Note
This level of accuracy is acceptable, especially given that the loss has been continuously decreasing, and it seems that this model may achieve even higher accuracy.

---

## Experiment 3: Multiclass tinyVGG Model
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
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 10
    - learning rate (LR): 1e-3

### Results
```
using device: cuda
Epoch 1 | loss: 332.7946288585663
Epoch 2 | loss: 332.5149710178375
Epoch 3 | loss: 332.3634819984436
Epoch 4 | loss: 332.2779014110565
Epoch 5 | loss: 330.7415204048157
Epoch 6 | loss: 317.343389749527
Epoch 7 | loss: 310.9958951473236

        loss:  3.404845901157545      
        accuracy:  0.06114130434782609

        execution time: 712.5488746166229s
```

---


## Experiment 4: Multiclass tinyVGG Model higher lr
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
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 10
    - learning rate (LR): 1e-2

### Results
```
using device: cuda
Epoch 1 | loss: 333.4019317626953
Epoch 2 | loss: 332.53786063194275
Epoch 3 | loss: 332.46077060699463
Epoch 4 | loss: 332.52098178863525
Epoch 5 | loss: 332.48950839042664
Epoch 6 | loss: 332.48462104797363
Epoch 7 | loss: 332.4466698169708
Epoch 8 | loss: 332.4573652744293
Epoch 9 | loss: 332.4914767742157
Epoch 10 | loss: 332.4100832939148

        loss:  3.6214312055836553
        accuracy:  0.020380434782608696

        execution time: 620.9463081359863s
```

### Note
well it clearly not eorking very well, its even worse than a random guess (~2.7%).
it seems tiny VGG is too simple and basic for such complex task.

---

## Experiment 5: AlexNet from scratch
### Details
- Model
    - type: AlexNet (custom implementation)
    - conv architecture:
        - conv(11, stride=4, out=96) → ReLU → LRN → max_pool(3, stride=2)
        - conv(5, padding=2, out=256) → ReLU → LRN → max_pool(3, stride=2)
        - conv(3, out=384) → ReLU
        - conv(3, out=384) → ReLU
        - conv(3, out=256) → ReLU → max_pool(3, stride=2)
    - fully connected (fc) layers:
        - fc1: Flatten → Dropout → Linear(256×6×6 → 1024) → ReLU
        - fc2: Dropout → Linear(1024 → 1024) → ReLU
        - fc3: Dropout → Linear(1024 → 37)
    - total params (approx): ~12–14 million
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: Resize(224, 224), RandomHorizontalFlip, ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 15
    - learning rate (LR): 1e-4
    - dropout rate: default (likely 0.5)

### Results
```
using device: cuda
Epoch 1 | loss: 1329.498753786087
Epoch 2 | loss: 1329.240790605545
Epoch 3 | loss: 1328.935516834259
Epoch 4 | loss: 1328.9312007427216
Epoch 5 | loss: 1328.825945854187
Epoch 6 | loss: 1329.0395317077637
Epoch 7 | loss: 1327.3453888893127
Epoch 8 | loss: 1322.2565281391144
Epoch 9 | loss: 1305.430225610733
Epoch 10 | loss: 1275.6677091121674
Epoch 11 | loss: 1247.8868854045868
Epoch 12 | loss: 1229.5839319229126
Epoch 13 | loss: 1214.8775861263275
Epoch 14 | loss: 1192.2647240161896
Epoch 15 | loss: 1181.5458929538727

        loss:  3.2734026571978694
        accuracy:  0.10326086956521739

        execution time: 1458.1396589279175s
```

### Note

these results are promising. it is learning something, but its slow and inefficient.

---
