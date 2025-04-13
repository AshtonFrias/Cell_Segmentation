import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# UNet Architecture was taken from this link, and in this code I modifed the skip connection layers
# https://www.kaggle.com/code/whats2000/u-net-bcss-segmentation

class DoubleConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvolutionBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, skip_type="default"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvolutionBlock(in_channels, out_channels)

        self.skip_type = skip_type  # "default", "fusion", or "convolutional"

        if self.skip_type == "convolutional":
            self.skip_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x1, x2, feature_maps=[]):
        #a = nn.Parameter(torch.tensor(0.5))
        x1 = self.up(x1)

        max_height = max(feature_maps.size(2) for feature_maps in feature_maps)
        max_width = max(feature_maps.size(3) for feature_maps in feature_maps)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        ''' Skip conneciton layer Modifications '''
        if self.skip_type == "default":
            # Standard Skip connection layer
            x = torch.cat([x2, x1], dim=1)

        elif self.skip_type == "fusion":
            # Fusion Skip Conneciton Layer
            fused_feature_maps = x1
            target_size = x1.size()[2:]
            for feature_maps in feature_maps:
                feature_maps_conv = nn.Conv2d(feature_maps.size(1), x1.size(1), kernel_size=1, bias=False).to(feature_maps.device)
                feature_maps_resized = F.interpolate(feature_maps, size=target_size, mode='bilinear', align_corners=False)
                feature_maps_resized = feature_maps_conv(feature_maps_resized)

                fused_feature_maps = fused_feature_maps + feature_maps_resized

                #Messed around with adding alpha but the results did not increase performance'''
                #fused_feature_maps = fused_feature_maps + (1 - a) * feature_maps_resized 

            x = torch.cat([x2, fused_feature_maps], dim=1) 

        elif self.skip_type == "convolutional":
            # Convolutional Skip Connection Layer
            x2_con = self.skip_conv(x2)
            x = torch.cat([x1, x2_con], dim=1)
        else:
            raise ValueError("Invalid skip_type. Choose from ['default', 'fusion', 'residual'].")

        return self.conv(x)

class OutputConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes, skip_type="default"):
        super().__init__()
        self.skip_type = skip_type
        self.inc = DoubleConvolutionBlock(num_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 1024)

        self.up1 = UpSample(1024, 512, skip_type=skip_type)
        self.up2 = UpSample(512, 256, skip_type=skip_type)
        self.up3 = UpSample(256, 128, skip_type=skip_type)
        self.up4 = UpSample(128, 64, skip_type=skip_type)

        self.outc = OutputConvolution(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        feature_maps = [x1, x2, x3, x4] 

        x = self.up1(x5, x4, feature_maps)
        x = self.up2(x,  x3, feature_maps)
        x = self.up3(x,  x2, feature_maps)
        x = self.up4(x,  x1, feature_maps)
        logits = self.outc(x)

        return logits