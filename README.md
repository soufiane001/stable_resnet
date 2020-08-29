# stable_resnet
Code for stable resnets.

Datesets included : Cifar10, Cifar100 and TinyImagenet.

Architectures included : ResNet32, ResNet50, ResNet104

To get set up:

1) (optional) make a virtual environment venv and activate it
2) install requirements: ```pip install -r requirements.txt```
3) install the (editable version of the) package: ```pip install --editable .```

Examples :

- Run ResNet32 on Cifar10 with BatchNorm, activation Relu, 

```python main.py --config configs/cifar10/resnet32/config.json --act relu --bn```

- Run ResNet32 on Cifar10 without BatchNorm and with decreasing scaling, activation Relu,

```python main.py --config configs/cifar10/resnet32/config.json --act relu --scaling Decrease```
