import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from stable_resnet.utils import try_cuda

__all__ = [
    "resnet"
]  # , 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
_AFFINE = True


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        scaled=False,
        act="relu",
        scaling_fac=1.0,
        bias=False,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        # self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias
        )
        # self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.scaled = scaled
        self.downsample = None
        # self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                )
            )
            # self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

        if act == "relu":
            self.activation = F.relu
        elif act == "elu":
            self.activation = F.elu
        elif act == "tanh":
            self.activation = F.tanh

        self.scaling_fac = scaling_fac
        if self.scaled:
            print("The scaling factor is {}".format(self.scaling_fac))

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = self.activation(self.conv1(x))
        if self.scaled:
            # TODO change to make it more modular 15 is only for resnet32
            out = self.conv2(out) / math.sqrt(self.scaling_fac)
        else:
            out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


class BasicBlock_BN(nn.Module):  # BasicBlock with batchnorm
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        scaled=False,
        act="relu",
        scaling_fac=1.0,
        bias=False,
    ):
        super(BasicBlock_BN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.scaled = scaled
        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                )
            )
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

        if act == "relu":
            self.activation = F.relu
        elif act == "elu":
            self.activation = F.elu
        elif act == "tanh":
            self.activation = F.tanh

        self.scaling_fac = scaling_fac
        if self.scaled:
            print("The scaling factor is {}".format(self.scaling_fac))

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        if self.scaled:
            # TODO change to make it more modular 15 is only for resnet32
            out = self.bn2(self.conv2(out)) / math.sqrt(self.scaling_fac)
        else:
            out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        num_classes=10,
        scaling="none",
        act="relu",
        bias=False,
        use_batch_norm=False,
    ):
        super(ResNet, self).__init__()
        _outputs = [32, 64, 128]

        self.in_planes = _outputs[0]
        self.act = act
        self.conv1 = nn.Conv2d(
            3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            print("We are using BatchNorm")
            self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
            block = BasicBlock_BN
        else:
            print("We are not using BatchNorm")
            block = BasicBlock

        self.layer1 = self._make_layer(
            block,
            1,
            _outputs[0],
            num_blocks[0],
            stride=1,
            scaling=scaling,
            act=act,
            bias=bias,
        )
        self.layer2 = self._make_layer(
            block,
            2,
            _outputs[1],
            num_blocks[1],
            stride=2,
            scaling=scaling,
            act=act,
            bias=bias,
        )
        self.layer3 = self._make_layer(
            block,
            3,
            _outputs[2],
            num_blocks[2],
            stride=2,
            scaling=scaling,
            act=act,
            bias=bias,
        )
        self.linear = nn.Linear(_outputs[2], num_classes)

        if act == "relu":
            print("We are using RELU")
            self.activation = F.relu
        elif act == "elu":
            print("We are using ElU")
            self.activation = F.elu
        elif act == "tanh":
            print("We are using TANH")
            self.activation = F.tanh

        if bias:
            print("We are using bias")
        # self.apply(weights_init)

    def _make_layer(
        self, block, section_num, planes, num_blocks, stride, scaling, act, bias,
    ):
        strides = [stride] + [1] * (num_blocks - 1)

        if scaling == "None":
            scaled = False
            scaling_factors = [1] * num_blocks
        elif scaling == "Decrease":
            scaled = True
            scaling_factors = 1 + np.arange(
                (section_num - 1) * num_blocks, section_num * num_blocks
            )
            scaling_factors = scaling_factors * np.log(scaling_factors + 1) ** 2
        elif scaling == "Uniform":
            scaled = True
            scaling_factor = 3 * num_blocks
            scaling_factors = [scaling_factor] * num_blocks
        else:
            raise ValueError(f"scaling {scaling} not found.")

        layers = []
        for i in range(num_blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    strides[i],
                    scaled,
                    act,
                    scaling_fac=scaling_factors[i],
                    bias=bias,
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_batch_norm:
            out = self.activation(self.bn(self.conv1(x)))
        else:
            out = self.activation(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet(
    depth=32, dataset="cifar10", scaling="none", BatchNorm=False, act="relu", bias=False
):
    if scaling != "none":
        print("#" * 40)
        print("We are scaling the blocks!!!!")
        print("#" * 40)
    assert (depth - 2) % 6 == 0, "Depth must be = 6n + 2, got %d" % depth
    n = (depth - 2) // 6
    if dataset == "cifar10":
        num_classes = 10
    elif dataset == "cifar100":
        num_classes = 100
    elif dataset == "tiny_imagenet":
        num_classes = 200
    else:
        raise NotImplementedError("Dataset [%s] is not supported." % dataset)

    return ResNet(
        [n] * 3, num_classes, scaling, act, bias=bias, use_batch_norm=BatchNorm
    )


def test(net):
    import numpy as np

    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print(
        "Total layers",
        len(
            list(
                filter(
                    lambda p: p.requires_grad and len(p.data.size()) > 1,
                    net.parameters(),
                )
            )
        ),
    )


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)
            test(globals()[net_name]())
            print()
