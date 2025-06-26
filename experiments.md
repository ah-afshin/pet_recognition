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

## Experiment 13: Pre-Trained MobileNet V2 (feature extraction)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 20 epochs (B: 8, lr: 1e-3)

### Results
```
using device: cuda
F:\code\pet_recognition\venv\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
F:\code\pet_recognition\venv\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=MobileNet_V2_Weights.IMAGENET1K_V1`. You can also use `weights=MobileNet_V2_Weights.DEFAULT` to get the most up-to-date weights.      
  warnings.warn(msg)
Epoch 1 | loss: 627.193497210741
Epoch 2 | loss: 296.44069066643715
Epoch 3 | loss: 250.1282778531313
Epoch 4 | loss: 219.66876972466707
Epoch 5 | loss: 208.78418096527457
Epoch 6 | loss: 195.3811812484637
Epoch 7 | loss: 183.3437040783465
Epoch 8 | loss: 169.62748143076897
Epoch 9 | loss: 175.88262029364705
Epoch 10 | loss: 185.72357198176906
Epoch 11 | loss: 175.8746707260143
Epoch 12 | loss: 157.851230006665
Epoch 13 | loss: 158.62252208963037
Epoch 14 | loss: 168.83395702997223
Epoch 15 | loss: 154.1298102173023
Epoch 16 | loss: 158.85925243142992
Epoch 17 | loss: 145.53149625076912
Epoch 18 | loss: 162.39680461841635
Epoch 19 | loss: 158.24795548710972
Epoch 20 | loss: 152.62473891256377

        loss:  0.4515860762204165
        accuracy:  0.8532608695652174
accuracy on traindata:

        loss:  0.11823268756230075
        accuracy:  0.9599184782608695

        execution time: 2025.8601686954498s
```

### Note
I got a warning because I loaded the pre-trained model the old-fashioned way 
which it appeared not to be very standard. well, this experiment is actually
really **successful**. accuracy over 85% on test data is good and acceptable
and even though it is 10% lower than the accuracy on the train data, this
much overfitting isn't concerning.

---

## Experiment 14: Pre-Trained MobileNet V2 (fine tuned)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 5 epochs (B: 8, lr: 1e-4)
- then fine tuned all layers for 15 epochs (B: 8, lr: 1e-4)

### Result
```
using device: cuda
Epoch 1 | loss: 1144.4444634914398
Epoch 2 | loss: 832.1231826543808
Epoch 3 | loss: 644.1364095211029
Epoch 4 | loss: 526.922719180584
Epoch 5 | loss: 458.66025799512863
finetuning started
Epoch 6 | loss: 404.52193224430084
Epoch 7 | loss: 369.41794365644455
Epoch 8 | loss: 334.28317525982857
Epoch 9 | loss: 313.6946943998337
Epoch 10 | loss: 297.7504475712776
Epoch 11 | loss: 275.3107729703188
Epoch 12 | loss: 268.87535017728806
Epoch 13 | loss: 251.4853989034891
Epoch 14 | loss: 240.90420266985893
Epoch 16 | loss: 234.14072681963444
Epoch 17 | loss: 224.76877158135176
Epoch 18 | loss: 218.61519815027714
Epoch 19 | loss: 211.10922230780125
Epoch 20 | loss: 208.95960349962115

        loss:  0.3827713985886911
        accuracy:  0.8858695652173914
accuracy on traindata:

        loss:  0.30242004249062715
        accuracy:  0.9296875

        execution time: 2421.153921365738s
```

### Note
well i accidentally fine-tuned all the layers due to a coding mistake.
generally this isn't a good approach because it increases the risk of
overfitting. however since accuracy and overfitting levels in this experiment
were very good and acceptable, I decided not to change anything and not to
repeat the experiment.

---

## Experiment 15: Pre-Trained ResNet18 (fine tuned)
### Details
- blocks and first two layers of network were freezed. the model was previously trained on ImageNet data set.
- layer 3 and 4 were not freezed. they are fine tuned from the beggining of the train loop.
- classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 20 epochs (B: 8, lr: 1e-3)

### Results
```
using device: cuda
Epoch 1 | loss: 898.4128633141518
Epoch 2 | loss: 569.4255219399929
Epoch 3 | loss: 475.93361650407314
Epoch 4 | loss: 383.99166238307953
Epoch 5 | loss: 339.4772923439741
Epoch 6 | loss: 319.32122905924916
Epoch 7 | loss: 255.49813161417842
Epoch 8 | loss: 234.55711001344025
Epoch 9 | loss: 224.54181181639433
Epoch 10 | loss: 199.87826238945127
Epoch 11 | loss: 182.32954698614776
Epoch 12 | loss: 160.09410419873893
Epoch 13 | loss: 155.05987930949777
Epoch 14 | loss: 135.25611167866737
Epoch 15 | loss: 132.1545193658676
Epoch 16 | loss: 143.02430854877457
Epoch 17 | loss: 111.55557688942645
Epoch 18 | loss: 98.33810705982614
Epoch 19 | loss: 114.80089500872418
Epoch 20 | loss: 101.23259816970676

        loss:  1.2180900018986152
        accuracy:  0.7309782608695652
accuracy on traindata:

        loss:  0.15947526884679064
        accuracy:  0.9548233695652174

        execution time: 2260.1676919460297s
```
### Note
this model, was agian a bit overfit, thats becaese we didn't train the head first, befor finetuning.
anyway, it was worth a try.

---

## Experiment 16: Pre-Trained EfficientNet b0 (feature extraction)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 20 epochs (B: 8, lr: 1e-4)

### Results
```
using device: cuda
Epoch 1 | loss: 1217.9891784191132
Epoch 2 | loss: 1016.3267788887024
Epoch 3 | loss: 862.9189001321793
Epoch 4 | loss: 754.7283246517181
Epoch 5 | loss: 662.4334119558334
Epoch 6 | loss: 598.999404668808
Epoch 7 | loss: 539.3183450698853
Epoch 8 | loss: 502.4701461791992
Epoch 9 | loss: 463.5139807462692
Epoch 10 | loss: 440.66223526000977
Epoch 11 | loss: 418.90176421403885
Epoch 12 | loss: 393.08573311567307
Epoch 13 | loss: 381.1789300441742
Epoch 14 | loss: 365.3804070651531
Epoch 15 | loss: 346.22139117121696
Epoch 16 | loss: 340.44976300001144
Epoch 17 | loss: 329.89475670456886
Epoch 18 | loss: 318.23450353741646
Epoch 19 | loss: 306.9516511261463
Epoch 20 | loss: 294.85318760573864

        loss:  0.6976782818851264
        accuracy:  0.8641304347826086
accuracy on traindata:

        loss:  0.5370603730173215
        accuracy:  0.9038722826086957

        execution time: 2299.3265442848206s
```

### Note
very successful.

---

## Experiment 17: Pre-Trained EfficientNet b0 (fine tuned)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 5 epochs (B: 8, lr: 1e-4)
- then fine tuned last 4 layers for 15 epochs (B: 8, lr: 1e-4)

### Result
```
using device: cuda
Epoch 1 | loss: 1217.6927936077118
Epoch 2 | loss: 1017.619083404541
Epoch 3 | loss: 872.7235282659531
Epoch 4 | loss: 759.6861160993576
Epoch 5 | loss: 665.8681279420853
finetuning started
Epoch 6 | loss: 603.1910520792007
Epoch 7 | loss: 547.2118337154388
Epoch 8 | loss: 507.7261085510254
Epoch 9 | loss: 472.21606266498566
Epoch 10 | loss: 441.0216094851494
Epoch 11 | loss: 421.02173775434494
Epoch 12 | loss: 396.834845662117
Epoch 13 | loss: 384.0993186533451
Epoch 14 | loss: 363.9455943107605
Epoch 15 | loss: 361.5533020198345
Epoch 16 | loss: 339.20350420475006
Epoch 17 | loss: 331.6569611430168
Epoch 18 | loss: 322.7000236660242
Epoch 19 | loss: 312.9666806459427
Epoch 20 | loss: 300.1557802259922

        loss:  0.6990421586062597
        accuracy:  0.8478260869565217
accuracy on traindata:

        loss:  0.5409904095222768
        accuracy:  0.8967391304347826

        execution time: 2301.211370229721s
```

### Note
successful.

---

## Experiment 18: Pre-Trained MobileNet V3 (feature extraction)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- a part of classification layer was replaced with a linear layer (with Dropout).
- the classification layer (block) was trained for 20 epochs (B: 8, lr: 1e-3)

### Results
```
using device: cuda
Epoch 1 | loss: 750.0477274060249
Epoch 2 | loss: 472.1994064003229
Epoch 3 | loss: 415.4004752486944
Epoch 4 | loss: 378.5403856188059
Epoch 5 | loss: 340.4806291908026
Epoch 6 | loss: 335.4834839850664
Epoch 7 | loss: 315.61730998009443
Epoch 8 | loss: 285.00090880692005
Epoch 9 | loss: 285.2142367400229
Epoch 10 | loss: 280.08437560498714
Epoch 11 | loss: 264.8213483430445
Epoch 12 | loss: 264.496435623616
Epoch 13 | loss: 262.46735134813935
Epoch 14 | loss: 244.01285494491458
Epoch 15 | loss: 253.1249573091045
Epoch 16 | loss: 237.69082155637443
Epoch 17 | loss: 228.72976906690747
Epoch 18 | loss: 223.46348185744137
Epoch 19 | loss: 227.87884776387364
Epoch 20 | loss: 213.15924902632833

        loss:  0.7490428084021677
        accuracy:  0.7934782608695652
accuracy on traindata:

        loss:  0.13061818935373437
        accuracy:  0.9561820652173914

        execution time: 2435.996080636978s
```

### Note
more complex models are more likely to overfit. as you can see mobile net v3
is more overfitted than mobile net v2.

---

## Experiment 19: Pre-Trained MobileNet V3 (fine tuned)
### Details
- blocks and network were freezed. the model was previously trained on ImageNet data set
- classification layer was replaced with a linear layer (with Dropout).
- a part of the classification layer (block) was trained for 5 epochs (B: 8, lr: 1e-4)
- then fine tuned last 4 layers for 15 epochs (B: 8, lr: 1e-4)

### Result
```
using device: cuda
Epoch 1 | loss: 1161.5775635242462
Epoch 2 | loss: 790.3239207267761
Epoch 3 | loss: 591.6434963941574
Epoch 4 | loss: 494.7524176090956
Epoch 5 | loss: 445.9883885383606
finetuning started
Epoch 6 | loss: 416.11276364326477
Epoch 7 | loss: 396.15985372662544
Epoch 8 | loss: 370.09851460158825
Epoch 9 | loss: 366.626371845603
Epoch 10 | loss: 339.1929488927126
Epoch 11 | loss: 338.61429522931576
Epoch 12 | loss: 323.1123569533229
Epoch 13 | loss: 316.5488931685686
Epoch 14 | loss: 301.2735814861953
Epoch 15 | loss: 300.08365738391876
Epoch 16 | loss: 286.67273046821356
Epoch 17 | loss: 282.997614108026
Epoch 18 | loss: 277.2514775916934
Epoch 19 | loss: 271.1470742672682
Epoch 20 | loss: 267.7297163680196

        loss:  0.48165148919772194
        accuracy:  0.8396739130434783
accuracy on traindata:

        loss:  0.3395599427825326
        accuracy:  0.9110054347826086

        execution time: 1901.2987399101257s
```

### Note
learned more this time.

---

# Conclusion
pre-trained models preformed much better and became less overfit on training data.
you can see a full analysis and conclusion in [analysis notebook](./notebooks/analysis.ipynb).

I think model 16 (feature-extracted Efficient Net b0) is the best model I have trained.
