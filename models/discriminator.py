import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        # self.model = nn.Sequential(
        #     *conv_block(3, 64, normalize=False), # (512 * 512) -> (256 * 256)  
        #     *conv_block(64, 128),  # (256 * 256) -> (128 * 128)
        #     *conv_block(128, 256),  # (128 * 128) -> (64 * 64)
        #     *conv_block(256, 512),  # (64 * 64) -> (32 * 32)
        #     *conv_block(512, 512),  # (32 * 32) -> (16 * 16)
        #     *conv_block(512, 512),  # (16 * 16) -> (8 * 8)
        #     nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (8 * 8) -> (7 * 7)
        #     nn.Sigmoid()
        # )
        self.enc1 = conv_block(6, 64, normalize=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256, dropout=0.5)
        self.enc4 = conv_block(256, 512, dropout=0.5)
        #self.enc5 = conv_block(512, 512)
        #self.enc6 = conv_block(512, 512, stride=1)
        self.enc7 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.activation = nn.Sigmoid()


    def forward(self, x, y):
        #return self.model(x)
        u = torch.cat([x,y], dim=1)
        enc1 = self.enc1(u)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        #enc5 = self.enc5(enc4)
        #enc6 = self.enc6(enc4)
        enc7 = self.enc7(enc4)
        return self.activation(enc7)

if __name__ == "__main__":
    discriminator = PatchGANDiscriminator()