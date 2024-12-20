import torch
import torch.nn as nn

class ResNetGenerator(nn.Module):
    # This class implements ResNetGenerator architecture
    def __init__(self):
        super(ResNetGenerator, self).__init__()

        # Definition function for convolutional block with optional normalization and dropout
        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Definition function for residual block 
        def residual_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False))
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            return nn.Sequential(*layers)

        # Definition function for residual convolutional block
        def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=output_padding, bias=False)]
            layers.append(nn.InstanceNorm2d(out_channels))
            if dropout>0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Encoder layers
        self.enc1 = conv_block(3, 64, kernel_size=7, stride=1, padding=3)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        # Residual connections
        self.res_blocks = nn.ModuleList([residual_block(256, 256) for _ in range(9)])

        # Decoder layers
        self.dec1 = deconv_block(256, 128)
        self.dec2 = deconv_block(128, 64)

        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    # Forward path
    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # Residual block
        for res_block in self.res_blocks:
            x = x + res_block(x)

        # Decoder
        x = self.dec1(x)
        x = self.dec2(x)
        output = self.final_layer(x)
        return output
        
