from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, use_upconv=True):
        super().__init__()
        if use_upconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.SELU(inplace=True))
        else:
            self.block = nn.Sequential(
                ConvRelu(in_channels, out_channels), nn.Upsample(scale_factor=(2, 2)), nn.SELU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ResUNet34(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        #self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class ResUNet101(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        #self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(2048, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(2048 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(1024 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(512 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(256 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class Resnet34Discriminator(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        #self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.encoder.conv1 = nn.Conv2d(
            num_classes + num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.encoder(x)


class ReResUNet34(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """

    def __init__(self, num_classes=1, num_inputs=3, num_mem_chan=1, num_filters=32, pretrained=True, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_inputs = num_inputs
        self.num_mem_chan = num_mem_chan

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        #self.encoder.bn1 = GroupNorm(self.encoder.bn1.num_features, num_groups=32)
        #self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_inputs + num_classes, self.encoder.conv1.out_channels, kernel_size=7, padding=3, stride=2),
            self.encoder.bn1,
            #nn.SELU(inplace=True),
            self.encoder.relu,
            self.pool)

        #for layer in [self.encoder.layer1, self.encoder.layer2, self.encoder.layer3, self.encoder.layer4]:
        #for block in layer:
        #block.bn1 = nn.SELU(inplace=True)#GroupNorm(block.bn1.num_features, num_groups=32)
        #block.bn2 = nn.SELU(inplace=True)#GroupNorm(block.bn2.num_features, num_groups=32)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes + 3*num_mem_chan, kernel_size=1)

    def forward(self, inp, itr=1, return_mem=False):
        N, C, H, W = inp.shape
        assert C == self.num_inputs
        x = torch.zeros((N, self.num_classes + self.num_inputs, H, W), dtype=inp.dtype, device=inp.device)
        x[:, :C, :, :] = inp
        with torch.no_grad():
            for i in range(itr):
               self.iterate(x)
        if return_mem:
            return x[:, :self.num_inputs, :, :], x[:, self.num_inputs:, :, :]
        return x[:, self.num_inputs:, :, :]

    def residual(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        return self.final(dec0)

    def iterate(self, x):
        x_mod = self.residual(x)
        forget = torch.sigmoid(x_mod[:, :self.num_mem_chan, :, :])
        attend = torch.sigmoid(x_mod[:, self.num_mem_chan:2*self.num_mem_chan, :, :])
        update = torch.tanh(x_mod[:, 2*self.num_mem_chan:3*self.num_mem_chan, :, :])
        x = x * (1-forget) + attend*update
        return x

    def anti_sig(self, x):
        x = torch.clamp(x, 1e-5, 1 - 1e-5)
        return torch.log(x) - torch.log(1 - x)

    def execute(self, inp, itr=0):
        N, C, H, W = inp.shape
        assert C == self.num_inputs
        x = torch.zeros((N, self.num_classes + self.num_inputs, H, W), dtype=inp.dtype, device=inp.device)
        x[:, :C, :, :] = inp
        with torch.no_grad():
            for i in range(itr):
                x_mod = self.residual(x)
                forget = torch.sigmoid(x_mod[:, :self.num_inputs + self.num_classes, :, :])
                remember = x_mod[:, self.num_inputs + self.num_classes:, :, :]
                x = x * forget + remember * (1 - forget)

        x_mod = self.residual(x)
        forget = torch.sigmoid(x_mod[:, :self.num_inputs + self.num_classes, :, :])
        remember = x_mod[:, self.num_inputs + self.num_classes:, :, :]
        x = x * forget + remember * (1 - forget)
        x_mod = self.residual(x)
        forget = torch.sigmoid(x_mod[:, :self.num_inputs + self.num_classes, :, :])
        remember = x_mod[:, self.num_inputs + self.num_classes:, :, :]
        x = x * forget + remember * (1 - forget)
        return x[:, self.num_inputs:, :, :]


class GroupNorm(nn.Module):
    """
    GroupNorm impl. KuangLiu
    https://github.com/kuangliu/pytorch-groupnorm/blob/master/groupnorm.py
    """

    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
