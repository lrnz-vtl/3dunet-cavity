# 3dunet-cavity

Predict protein binding cavities

Forked from the PyTorch implementation of 3D U-Net  https://travis-ci.com/wolny/pytorch-3dunet


## Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

### Running on Windows
The package has not been tested on Windows, however some reported using it on Windows. One thing to keep in mind:
when training with `CrossEntropyLoss`: the label type in the config file should be change from `long` to `int64`,
otherwise there will be an error: `RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target'`.

## Supported Loss Functions

### Semantic Segmentation
- _BCEWithLogitsLoss_ (binary cross-entropy)
- _DiceLoss_ (standard `DiceLoss` defined as `1 - DiceCoefficient` used for binary semantic segmentation; when more than 2 classes are present in the ground truth, it computes the `DiceLoss` per channel and averages the values).
- _BCEDiceLoss_ (Linear combination of BCE and Dice losses, i.e. `alpha * BCE + beta * Dice`, `alpha, beta` can be specified in the `loss` section of the config)
- _CrossEntropyLoss_ (one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _PixelWiseCrossEntropyLoss_ (one can specify not only class weights but also per pixel weights in order to give more gradient to important (or under-represented) regions in the ground truth)
- _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config)
- _GeneralizedDiceLoss_ (see 'Generalized Dice Loss (GDL)' in the below paper for a detailed explanation; one can specify class weights via `weight: [w_1, ..., w_k]` in the `loss` section of the config). 
Note: use this loss function only if the labels in the training dataset are very imbalanced e.g. one class having at least 3 orders of magnitude more voxels than the others. Otherwise use standard _DiceLoss_.

For a detailed explanation of some of the supported loss functions see:
[Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/pdf/1707.03237.pdf)
Carole H. Sudre, Wenqi Li, Tom Vercauteren, Sebastien Ourselin, M. Jorge Cardoso

**IMPORTANT**: if one wants to use their own loss function, bear in mind that the current model implementation always
output logits and it's up to the implementation of the loss to normalize it correctly, e.g. by applying Sigmoid or Softmax.


## Supported Evaluation Metrics

### Semantic Segmentation
- _MeanIoU_ - Mean intersection over union
- _DiceCoefficient_ - Dice Coefficient (computes per channel Dice Coefficient and returns the average)
If a 3D U-Net was trained to predict cell boundaries, one can use the following semantic instance segmentation metrics
(the metrics below are computed by running connected components on thresholded boundary map and comparing the resulted instances to the ground truth instance segmentation): 
- _BoundaryAveragePrecision_ - Average Precision applied to the boundary probability maps: thresholds the boundary maps given by the network, runs connected components to get the segmentation and computes AP between the resulting segmentation and the ground truth
- _AdaptedRandError_ - Adapted Rand Error (see http://brainiac2.mit.edu/SNEMI3D/evaluation for a detailed explanation)
- _AveragePrecision_ - see https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric


If not specified `MeanIoU` will be used by default.



## Installation
- TODO


## Train

TODO

## Prediction

TODO

## Data Parallelism
By default, if multiple GPUs are available training/prediction will be run on all the GPUs using [DataParallel](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html).
If training/prediction on all available GPUs is not desirable, restrict the number of GPUs using `CUDA_VISIBLE_DEVICES`, e.g.
```bash
CUDA_VISIBLE_DEVICES=0,1 train3dunet --config <CONFIG>
``` 
or
```bash
CUDA_VISIBLE_DEVICES=0,1 predict3dunet --config <CONFIG>
```
