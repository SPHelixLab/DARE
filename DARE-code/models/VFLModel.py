"""
@author: tzh666
@context: VFL Model Structure
"""
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


# ---------------------------------------------------------------#
# ----------------------RESNET INIT START------------------------#
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


# -------------------------------------------------------------#
# ----------------------FCNN INIT START------------------------#
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --------------------------------------------------------------#
# -----------------------FOR VFL TRAINING-----------------------#
class VFLBottomModel(nn.Module):
    def __init__(self, args):
        super(VFLBottomModel, self).__init__()
        self.net, _ = model_init(args)

    def forward(self, x):
        x = self.net(x)
        return x


class VFLTopModel(nn.Module):
    def __init__(self, args):
        super(VFLTopModel, self).__init__()
        _, model_class = model_init(args)
        ## If test cutting layer here, you would modify it accordingly. At the same time, the last layer of the bottom model should have one less block.
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.fc0top = nn.Linear(16, model_class*2)

        # The input size of the top model is twice that of the bottom model.
        self.fc1top = nn.Linear(model_class*2, model_class*2)
        self.fc2top = nn.Linear(model_class*2, model_class)
        self.fc3top = nn.Linear(model_class, model_class)
        self.fc4top = nn.Linear(model_class, model_class)
        self.bn0top = nn.BatchNorm1d(model_class*2)
        self.bn1top = nn.BatchNorm1d(model_class*2)
        self.bn2top = nn.BatchNorm1d(model_class)
        self.bn3top = nn.BatchNorm1d(model_class)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models

        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))

        return F.log_softmax(x, dim=1)


def model_init(args):
    if args.dataset == "CIFAR10":
        model_class = 10
    elif args.dataset == "CIFAR100":
        model_class = 100
    elif args.dataset == "TinyImageNet":
        model_class = 200
    else:
        model_class = 2

    if args.vfl_model == "resnet8":
        bottom_model = resnet8(num_classes=model_class)
    elif args.vfl_model == "resnet14":
        bottom_model = resnet14(num_classes=model_class)
    elif args.vfl_model == "resnet20":
        bottom_model = resnet20(num_classes=model_class)
    elif args.vfl_model == "resnet50":
        bottom_model = resnet50(num_classes=model_class)
    else:
        bottom_model = FCNN()

    return bottom_model, model_class


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def resnet8(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[1, 1, 1], kernel_size=kernel_size, num_classes=num_classes)


def resnet14(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2], kernel_size=kernel_size, num_classes=num_classes)


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet50(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[8, 8, 8], kernel_size=kernel_size, num_classes=num_classes)

