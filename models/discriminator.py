import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    # This class implements PatchGANDiscriminator architecture
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        # Definition function for convolutional block
        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)


        self.enc1 = conv_block(3, 64, normalize=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256, dropout=0.5)
        self.enc4 = conv_block(256, 512, stride=1, dropout=0.5)
        self.enc5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)


    def forward(self, x):
        # Forward path through the network
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

if __name__ == "__main__":
    discriminator = PatchGANDiscriminator()