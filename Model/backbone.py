import torch
from torch import nn
from torchvision.models.resnet import Bottleneck, BasicBlock

from argSettings import arg


class ResNet(nn.Module):

    def __init__(self, num_classes, layers=[1,1,1,1], zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        block = Bottleneck

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None ")

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EmbeddingImagenet(nn.Module):
    def __init__(self, emb_size):
        super(EmbeddingImagenet, self).__init__()

        # cifar or not
        self.hidden = 64
        if arg.dataset == 'cifar':
            self.last_hidden = self.hidden * 1
        else:
            self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=self.hidden,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=self.hidden),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                             out_channels=int(self.hidden * 1.5),
                                             kernel_size=3,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                             out_channels=self.hidden * 2,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=self.hidden * 2),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.4))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 2,
                                             out_channels=self.hidden * 4,
                                             kernel_size=3,
                                             padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(num_features=self.hidden * 4),
                                   nn.MaxPool2d(kernel_size=2),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Dropout2d(0.5))
        self.last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                            out_features=self.emb_size, bias=True),
                                  nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        output_data = self.conv4(self.conv3(self.conv2(self.conv1(input_data))))
        return self.last(output_data.view(output_data.size(0), -1))

