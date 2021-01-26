### Simplified ResNets Pytorch

Simplified code for resnets backbone. Use 'python3 train.py' to train and test. Accuracy achieved is as follows

|Model     |Accuracy |
|----------|---------|
|`resnet18`|95.43%   |
|`resnet50`|xx.xx%   |

Training takes roughly 50 mins on a Tesla V100S. Trained for 200 epochs. Augmentation and hyperparameters adopted from [here](https://github.com/kuangliu/pytorch-cifar).

Detailed training and validation plots are available [here](https://wandb.ai/afzal/resnet-test/reports/ResNets-CIFAR10-Tests--Vmlldzo0Mjk0NTg)
