# stable_resnet
Code for stable resnets

Examples :

- Run ResNet32 on Cifar10 with BatchNorm, initialization on the EOC, activation Relu, 
python main.py --config configs/cifar10/resnet32/config_bn.json --init EOC --act relu --bn

- Run scaled ResNet32 on Cifar10 without BatchNorm, initialization on the EOC, activation Relu, 
python main.py --config configs/cifar10/resnet32/config_no_bn.json --init EOC --act relu --scaled
