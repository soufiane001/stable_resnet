import torch.nn as nn
import torch.nn.functional as F

defaultcfg = {
    11: [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    13: [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    16: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
    19: [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ],
    34: [
        64,
        64,
        64,
        64,
        64,
        "M",
        128,
        128,
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        512,
        512,
        512,
    ],
    49: [
        64,
        64,
        64,
        64,
        64,
        "M",
        64,
        64,
        64,
        "M",
        128,
        128,
        128,
        128,
        128,
        "M",
        128,
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
    ],
}


class padding_cir(nn.Module):

    """
	Creating circular padding
	"""

    def __init__(self):
        super(padding_cir, self).__init__()

    def forward(self, x):
        # creating the circular padding
        return F.pad(x, (1, 1, 1, 1), mode="circular")


class VGG(nn.Module):
    def __init__(
        self,
        dataset="cifar10",
        depth=19,
        init_weights=True,
        cfg=None,
        affine=True,
        batchnorm=True,
        act="relu",
    ):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine

        if act == "relu":
            print("We are using RELU")
            self.activation = nn.ReLU()
        elif act == "elu":
            print("We are using ElU")
            self.activation = nn.ELU()
        elif act == "tanh":
            print("We are using TANH")
            self.activation = nn.Tanh()

        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == "cifar10" or dataset == "cinic-10":
            num_classes = 10
        elif dataset == "cifar100":
            num_classes = 100
        elif dataset == "tiny_imagenet":
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    # if init_weights:
    #     self.apply(weights_init)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3

        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, bias=False)
                if batch_norm:
                    layers += [
                        padding_cir(),
                        conv2d,
                        nn.BatchNorm2d(v, affine=self._AFFINE),
                        self.activation,
                    ]
                else:
                    layers += [padding_cir(), conv2d, self.activation]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == "tiny_imagenet":
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


# def _initialize_weights(self):
# 	for m in self.modules():
# 		if isinstance(m, nn.Conv2d):
# 			n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
# 			m.weight.data.normal_(0, math.sqrt(2. / n))
# 			if m.bias is not None:
# 				m.bias.data.zero_()
# 		elif isinstance(m, nn.BatchNorm2d):
# 			if m.weight is not None:
# 				m.weight.data.fill_(1.0)
# 				m.bias.data.zero_()
# 		elif isinstance(m, nn.Linear):
# 			m.weight.data.normal_(0, 0.01)
# 			m.bias.data.zero_()
