import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel
from swin_transformer_pytorch import SwinTransformer
import torch.nn.functional as F

class SwinUNet(nn.Module):
    def __init__(self, n_classes, device="cpu", skip_type="default"):
        super(SwinUNet, self).__init__()
        self.device = torch.device(device)
        self.skip_type = skip_type

        # Swin Transformer Encoder (224x224 input)
        # Pretained encoder from https://arxiv.org/abs/2103.14030
        config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        config.image_size = 224
        self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224", config=config)

        # Decoder, Upsampling to 224x224 to match mask output
        self.up_sample_conv4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample_conv3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample_conv2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.up_sample_conv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder Convolutional Blocks
        self.decode4 = self.conv_block(768, 384)
        self.decode3 = self.conv_block(384 + 768, 192)
        self.decode2 = self.conv_block(192 + 384, 96)
        self.decode1 = self.conv_block(96 + 192, 64)

        # Final Convolution (Output Mask)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Convolutional Skip Connection layer
        self.skip_conv3 = nn.Conv2d(768, 768, kernel_size=1, device=self.device)
        self.skip_conv2 = nn.Conv2d(384, 384, kernel_size=1, device=self.device)
        self.skip_conv1 = nn.Conv2d(192, 192, kernel_size=1, device=self.device)

    # Convolutional Similar to UNets
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=self.device),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=self.device),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        outputs = self.encoder(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Extract Encoder Feature Maps from Swin Encoder
        enc1 = hidden_states[1]
        enc2 = hidden_states[2]
        enc3 = hidden_states[3]
        enc4 = hidden_states[4]
        batch_size, num_patches, embedding_dim = enc1.shape

        # Dynamically Calculate Spatial Dimensions
        h1 = int((num_patches)**0.5)
        h2 = h1 // 2
        h3 = h2 // 2
        h4 = h3 

        # Reshape Feature Maps
        enc1 = enc1.transpose(1, 2).view(batch_size, 192, h1, h1)
        enc2 = enc2.transpose(1, 2).view(batch_size, 384, h2, h2)
        enc3 = enc3.transpose(1, 2).view(batch_size, 768, h3, h3)
        enc4 = enc4.transpose(1, 2).view(batch_size, 768, h4, h4)

        feature_maps = [enc1, enc2, enc3, enc4]

        # Decoder Path with Skip Connections
        up4 = self.up_sample_conv4(enc4)
        up4 = self.decode4(up4)
        enc3_up = F.interpolate(enc3, size=up4.shape[2:], mode='bilinear', align_corners=False)

        ''' Skip conneciton layer Modifications '''
        if self.skip_type == "convolutional":
            enc3_conv = self.skip_conv3(enc3_up)
            up3 = torch.cat([up4, enc3_conv], dim=1)

        elif self.skip_type == "fusion":
            fused_fm = enc3_up
            target_size = enc3_up.size()[2:]
            feature_maps = [enc1, enc2]
            for fm in feature_maps:
                fm_conv = nn.Conv2d(fm.size(1), enc3_up.size(1), kernel_size=1, bias=False).to(fm.device)
                fm_resized = F.interpolate(fm, size=target_size, mode='bilinear', align_corners=False)
                fm_resized = fm_conv(fm_resized)
                fused_fm = fused_fm + fm_resized
            up3 = torch.cat([up4, fused_fm], dim=1)
        else:
            up3 = torch.cat([up4, enc3_up], dim=1)

        # Standard upconvolution
        up3 = self.decode3(up3)
        up3 = self.up_sample_conv3(up3)
        enc2_up = F.interpolate(enc2, size=up3.shape[2:], mode='bilinear', align_corners=False)

        ''' Skip conneciton layer Modifications '''
        if self.skip_type == "convolutional":
            enc2_conv = self.skip_conv2(enc2_up)
            up2 = torch.cat([up3, enc2_conv], dim=1)
        elif self.skip_type == "fusion":
            fused_fm = enc2_up
            target_size = enc2_up.size()[2:]
            feature_maps = [enc1, enc3]
            for fm in feature_maps:
                fm_conv = nn.Conv2d(fm.size(1), enc2_up.size(1), kernel_size=1, bias=False).to(fm.device)
                fm_resized = F.interpolate(fm, size=target_size, mode='bilinear', align_corners=False)
                fm_resized = fm_conv(fm_resized)
                fused_fm = fused_fm + fm_resized
            up2 = torch.cat([up3, fused_fm], dim=1)
        else:
            up2 = torch.cat([up3, enc2_up], dim=1)

        # Standard upconvolution
        up2 = self.decode2(up2)
        up2 = self.up_sample_conv2(up2)
        enc1_up = F.interpolate(enc1, size=up2.shape[2:], mode='bilinear', align_corners=False)

        ''' Skip conneciton layer Modifications '''
        if self.skip_type == "convolutional":
            enc1_conv = self.skip_conv1(enc1_up)
            up1 = torch.cat([up2, enc1_conv], dim=1)
        elif self.skip_type == "fusion":
            fused_fm = enc1_up
            target_size = enc1_up.size()[2:]
            feature_maps = [enc2, enc3]
            for fm in feature_maps:
                fm_conv = nn.Conv2d(fm.size(1), enc1_up.size(1), kernel_size=1, bias=False).to(fm.device)
                fm_resized = F.interpolate(fm, size=target_size, mode='bilinear', align_corners=False)
                fm_resized = fm_conv(fm_resized)
                fused_fm = fused_fm + fm_resized
            up1 = torch.cat([up2, fused_fm], dim=1)
        else:
            up1 = torch.cat([up2, enc1_up], dim=1)

        # Standard upconvolution
        up1 = self.decode1(up1)
        up1 = self.up_sample_conv1(up1)

        up1 = self.final_up(up1) 
        output = self.final(up1)

        return output
