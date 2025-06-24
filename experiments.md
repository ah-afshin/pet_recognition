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

## Experiment 6: ResNet9 v1
### Details
### Details
- Model
    - type: ResNet9
    - architecture:
        - conv(7, stride=2, out=64) → BN → ReLU → max_pool(3, stride=2, padding=1)
        - Residual Blocks
            - Residual(64, 128)
            - Residual(128, 256)
            - Residual(256, 512)
        - adaptive_avg_pool((1, 1)) → Flatten → Linear()
    - total params (approx): ~12–14 million
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: Resize(224, 224), RandomHorizontalFlip, ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 16
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 20
    - learning rate (LR): 1e-4

### Result
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 98.00 MiB (GPU 0; 2.00 GiB total capacity; 1.17 GiB already allocated; 0 bytes free; 1.58 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```

### Note
my GPU is kinda old :)
let reduce batch size to 8.

---

## Experiment 7: ResNet9 v1 (Repeated)
### Details
- Model
    - type: ResNet9
    - architecture:
        - conv(7, stride=2, out=64) → BN → ReLU → max_pool(3, stride=2, padding=1)
        - Residual Blocks
            - Residual(64, 128)
            - Residual(128, 256)
            - Residual(256, 512)
        - adaptive_avg_pool((1, 1)) → Flatten → Linear(512 → 37)
    <!-- - total params (approx): ~12–14 million -->
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: Resize(224, 224), RandomHorizontalFlip, ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 20
    - learning rate (LR): 1e-4

### Results
```
using device: cuda
Epoch 1 | loss: 1289.3761713504791
Epoch 2 | loss: 1220.6479277610779
Epoch 3 | loss: 1172.6037809848785
Epoch 4 | loss: 1128.148670911789
Epoch 5 | loss: 1088.570368528366
Epoch 6 | loss: 1061.1119492053986
Epoch 7 | loss: 1021.9093277454376
Epoch 8 | loss: 1004.8771576881409
Epoch 9 | loss: 975.5011868476868
Epoch 10 | loss: 948.7167570590973
Epoch 11 | loss: 919.0879226922989
Epoch 12 | loss: 902.5428096055984
Epoch 13 | loss: 874.5284065008163
Epoch 14 | loss: 855.7540719509125
Epoch 15 | loss: 846.8048573732376
Epoch 16 | loss: 822.7289730906487
Epoch 17 | loss: 808.5026944875717
Epoch 18 | loss: 788.7080084085464
Epoch 19 | loss: 776.1853362321854
Epoch 20 | loss: 768.6493322849274

        loss:  2.6390033429083615
        accuracy:  0.2798913043478261

        execution time: 5629.793249368668s
```

### Note
this is really good and promising, we are going to reload it and train it again.

---

## Experiment 8: ResNet9 v1 (Reloaded)
### Details
- Model: model_07__breedsrecog_ResNet9_v1 (Experiment 7)[#experiment-7] reloaded
    - acc: 0.3641304347826087
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: Resize(224, 224), RandomHorizontalFlip, ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 10
    - learning rate (LR): 1e-4

### Results
```
using device: cuda

model is loaded

        loss:  2.306448975334997
        accuracy:  0.3641304347826087

continue training...

Epoch 1 | loss: 767.6557443141937
Epoch 2 | loss: 749.2265840768814
Epoch 3 | loss: 733.8775580525398
Epoch 4 | loss: 716.6395802497864
Epoch 5 | loss: 697.2467594146729
Epoch 6 | loss: 684.0284001231194
Epoch 7 | loss: 671.3295263648033
Epoch 8 | loss: 657.9248953461647
Epoch 9 | loss: 638.0831761956215
Epoch 10 | loss: 620.8314091563225

        loss:  2.3618192258088486
        accuracy:  0.3111413043478261

        execution time: 2487.7218022346497s
```

### Note
The results are pretty bad and overwhelming, It is definitely overfitting as the accurecy has decreased in the test data.
Notice how the accuracy in the biggining of this training session, was higher than accuracy shown in Experiment 7.
it was probably because it's preforming better on its training data, witch was then shuffled after that experiment and this time
was making up 80% of eval dataset. this is a sign of **overfitting**, and honestly it's not very surprising.

---

at this point dataloaders and datasets were also fixed to prevent overfitting

---

## Experiment 9: ResNet9 v2 and Regularization
### Details
- Model
    - type: ResNet9
    - architecture:
        - conv(7, stride=2, out=64) → BN → ReLU → max_pool(3, stride=2, padding=1)
        - Residual Blocks
            - Residual(64, 128)
            - Residual(128, 256)
            - Residual(256, 512)
        - adaptive_avg_pool((1, 1)) → Flatten → Dropout(0.5) → Linear
    <!-- - total params (approx): ~12–14 million -->
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: 
        - training: RandomResizeCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ToTensor, Normalize(mean=0.5, std=0.5)
        - testing: Resize(256), CenterCrop(224), ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - regulation: L2 (rate=1e-4)
    - epochs: 20
    - learning rate (LR): 1e-4
    - LR schedule
        - step: 10
        - gamma: 0.5

### Results
```
using device: cuda
Epoch 1 | loss: 1307.463386774063
Epoch 2 | loss: 1257.8599770069122
Epoch 3 | loss: 1222.9602036476135
Epoch 4 | loss: 1190.287414073944
Epoch 5 | loss: 1163.1757979393005
Epoch 6 | loss: 1130.9848773479462
Epoch 7 | loss: 1102.7683001756668
Epoch 8 | loss: 1077.6735739707947
Epoch 9 | loss: 1044.3571739196777
Epoch 10 | loss: 1023.6537770032883
Epoch 11 | loss: 1001.2572629451752
Epoch 12 | loss: 981.8684629201889
Epoch 13 | loss: 969.1346012353897
Epoch 14 | loss: 954.9741523265839
Epoch 15 | loss: 934.5768994092941
Epoch 16 | loss: 917.948178768158
Epoch 17 | loss: 909.4419968128204
Epoch 18 | loss: 889.4127442836761
Epoch 19 | loss: 887.626139163971
Epoch 20 | loss: 858.3550046682358

        loss:  2.547024128229722
        accuracy:  0.2717391304347826

        execution time: 4782.906373023987s
```
and after we reload it and shuffle the datasets:
```
model is loaded

        loss:  2.186990054405254     
        accuracy:  0.3532608695652174
```

### Note
it is still really overfit. it happens, especially if you use a model that is so complicated for a simple task.
lets reduce models size, if it doesn't improve, we'll use the *early stopping* method during training which,
since it requires constant evaluation, is slower per epoch.

---

## Experiment 10: ResNet9 v3 smaller
### Details
- Model
    - type: ResNet9
    - architecture:
        - conv(7, stride=2, out=32) → BN → ReLU → max_pool(3, stride=2, padding=1)
        - Residual Blocks
            - Residual(32, 48)
            - Residual(48, 48)
            - Residual(48, 48)
        - adaptive_avg_pool((1, 1)) → Flatten → Dropout(0.5) → Linear
    <!-- - total params (approx): ~12–14 million -->
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: 
        - training: RandomResizeCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ToTensor, Normalize(mean=0.5, std=0.5)
        - testing: Resize(256), CenterCrop(224), ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - regulation: L2 (rate=1e-4)
    - epochs: 20
    - learning rate (LR): 1e-4
    - LR schedule
        - step: 10
        - gamma: 0.5

### Results
```
using device: cuda
Epoch 1 | loss: 1325.9941911697388
Epoch 2 | loss: 1295.174076795578
Epoch 3 | loss: 1276.6105766296387
Epoch 4 | loss: 1262.293910741806
Epoch 10 | loss: 1200.8085913658142
Epoch 11 | loss: 1192.8403906822205
Epoch 12 | loss: 1188.2736237049103
Epoch 13 | loss: 1175.8460626602173
Epoch 14 | loss: 1173.440645456314
Epoch 15 | loss: 1166.471237897873
Epoch 16 | loss: 1153.326904296875
Epoch 17 | loss: 1154.752429485321
Epoch 18 | loss: 1144.6621329784393
Epoch 19 | loss: 1136.7630186080933
Epoch 20 | loss: 1134.104972600937

        loss:  2.9218801506187604
        accuracy:  0.19157608695652173

        execution time: 14310.323883771896s

model is loaded

        loss:  2.8912977187529854
        accuracy:  0.22282608695652173
```

### Note
we are done with ResNets, It's overfit, again.

---

## Experiment 11: MobileNet v1
### Details
- Model
    - type: MobileNet V1
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: 
        - training: RandomResizeCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ToTensor, Normalize(mean=0.5, std=0.5)
        - testing: Resize(256), CenterCrop(224), ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 20
    - learning rate (LR): 7e-4

### Results
```
using device: cuda
Epoch 1 | loss: 1349.303840637207
Epoch 2 | loss: 1279.5130293369293
Epoch 3 | loss: 1229.6131663322449
Epoch 4 | loss: 1206.8201880455017
Epoch 5 | loss: 1157.8438124656677
Epoch 6 | loss: 1129.3577597141266
Epoch 7 | loss: 1089.892460346222
Epoch 8 | loss: 1066.053144812584
Epoch 9 | loss: 1029.4128324985504
Epoch 10 | loss: 1005.1925023794174
Epoch 11 | loss: 966.3287199735641
Epoch 12 | loss: 927.0736595392227
Epoch 13 | loss: 900.3757032155991
Epoch 14 | loss: 865.2207746505737
Epoch 15 | loss: 813.1228718757629
Epoch 16 | loss: 790.3135493993759
Epoch 17 | loss: 748.2036954164505
Epoch 18 | loss: 711.2638738155365
Epoch 19 | loss: 674.0908417701721
Epoch 20 | loss: 630.7947295308113

        loss:  2.225416342849317
        accuracy:  0.36277173913043476
accuracy on traindata:

        loss:  1.361271340480965
        accuracy:  0.5709918478260869

        execution time: 2314.5989351272583s
```

### Note
the good point is, it learns fast :) it's overfit again, but learns fast :)

---

## Experiment 12: MobileNet v2
### Details
- Model
    - type: MobileNet V2 (custom implementation)
    - Block Configs
        - 16 → 24, epxansion-factor: 1, stride: 2
        - 24 → 32, epxansion-factor: 6, stride: 2 (x2)
        - 32 → 64, epxansion-factor: 6, stride: 2 (x2)
        - 64 → 96, epxansion-factor: 6, stride: 1
        - 96 → 128, epxansion-factor: 6, stride: 2
- Data
    - dataset: Oxford-IIIT Pet Dataset
    - labels: 37 pet breeds
    - preprocess: 
        - training: RandomResizeCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(0.5), RandomRotation(15), ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), ToTensor, Normalize(mean=0.5, std=0.5)
        - testing: Resize(256), CenterCrop(224), ToTensor, Normalize(mean=0.5, std=0.5)
    - train-test ratio: 80-20
    - batch size (B): 8 (mini batch)
- Training
    - loss func: CrossEntropyLoss
    - optimizer: Adam
    - epochs: 20
    - learning rate (LR): 5e-4

### Results
```
using device: cuda
Epoch 1 | loss: 1291.7910568714142
Epoch 2 | loss: 1242.5778052806854
Epoch 3 | loss: 1199.41397356987
Epoch 4 | loss: 1161.3064994812012
Epoch 5 | loss: 1123.7230820655823
Epoch 6 | loss: 1080.4931507110596
Epoch 7 | loss: 1051.964612007141
Epoch 8 | loss: 1032.7356933355331
Epoch 9 | loss: 1001.7205847501755
Epoch 10 | loss: 964.6651638746262
Epoch 11 | loss: 951.2211530208588
Epoch 12 | loss: 924.5077345371246
Epoch 13 | loss: 894.1818410158157
Epoch 14 | loss: 874.8258794546127
Epoch 15 | loss: 857.0157412290573
Epoch 15 | loss: 857.0157412290573
Epoch 15 | loss: 857.0157412290573
Epoch 16 | loss: 826.2697563171387
Epoch 17 | loss: 794.3202216625214
Epoch 18 | loss: 793.9646969437599
Epoch 19 | loss: 759.573096871376
Epoch 20 | loss: 737.654336810112

        loss:  2.1869880252558254
        accuracy:  0.37228260869565216
accuracy on traindata:

        loss:  1.6274664704890356
        accuracy:  0.5098505434782609

        execution time: 2401.9491052627563s
```

### Note
overfitted.

---