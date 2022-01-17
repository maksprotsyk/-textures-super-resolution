""" Full assembly of the parts to form the complete network """
import torch

from src.models.pytorch_models.UNet.unet_parts import *


class UNet(nn.Module):
    """
    https://github.com/milesial/Pytorch-UNet/tree/master/unet
    x2 super resolution
    TODO: handle x2-x4 switch
    """

    def __init__(self, bilinear=True):
        """
        """
        super(UNet, self).__init__()
        # self.n_channels = n_channels
        print("hi from Unet")

        self.bilinear = bilinear

        self.inc = DoubleConv(4, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, 4)

    def forward(self, x):
        initial_x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x1 = self.inc(initial_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return initial_x + x


if __name__ == "__main__":
    inp = torch.randn((16, 4, 128, 128))

    model = UNet()

    print(inp.shape)
    out = model(inp)
    print(out.shape)
