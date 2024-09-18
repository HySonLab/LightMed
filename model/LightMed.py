import torch
import torch
import torch.nn as nn
from model.antialias import Sequential_complex, ComplexLowpass
from model.block_complex import Complex_AdaptiveAvgPool2d, ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv2d, ComplexConvTranspose2d, ComplexDropBlock2D, ComplexDropout, ComplexDropout2d, ComplexELU, ComplexLinear, ComplexMaxPool2d, \
                                ComplexReLU, ComplexSequential, Conv2d, ConvTranspose2d, complex_AdaptiveAvgPool2d, complex_cat, complex_dropout, complex_dropout2d, complex_elu, complex_max_pool2d, complex_relu, complex_up, complex_up_16, \
                                complex_up_4, complex_up_8, _ComplexBatchNorm, NaiveComplexBatchNorm1d, NaiveComplexBatchNorm2d
from model.ccbam import ComplexCAM, ComplexSAM, ComplexCBAM
from utils.filter import low_pass_filter, low_pass_filter2, low_pass_filter3, low_pass_filter4, LowPassFilter, ZeroPadding
from torch.nn import Module
import torch.nn.functional as F
from model.CCV import ComplexConvBlock
image_size = 256
r = 64
in_channels = 3
out_channels = 1
class LightMed(Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(LightMed, self).__init__()
        factor = 4 
        filters = [64//factor, 128//factor, 256//factor, 512//factor, 1024//factor]
        self.lowpass = LowPassFilter(r=r)
        self.pad = ZeroPadding(size=image_size, r = r)
        
        self.encoder1 = ComplexConvBlock(in_channels, 64//factor)
        self.pool1 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti1 = ComplexLowpass(filt_size=3, stride=2, channels=64//factor)

        self.encoder2 = ComplexConvBlock(64//factor, 128//factor)
        self.pool2 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti2 = ComplexLowpass(filt_size=3, stride=2, channels=128//factor)

        self.encoder3 = ComplexConvBlock(128//factor, 256//factor, drop = True)
        self.pool3 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti3 = ComplexLowpass(filt_size=3, stride=2, channels=256//factor)

        self.encoder4 = ComplexConvBlock(256//factor, 512//factor, drop = True)
        self.pool4 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti4 = ComplexLowpass(filt_size=3, stride=2, channels=512//factor)

        self.center = ComplexConvBlock(512//factor, 1024//factor)
        self.centercbam = ComplexCBAM(1024//factor,1024//factor)
        
        self.up4 = ComplexConvTranspose2d(1024//factor, 512//factor, kernel_size=2, stride=2)
        self.decoder4 = ComplexConvBlock(1024//factor, 512//factor)

        self.up3 = ComplexConvTranspose2d(512//factor, 256//factor, kernel_size=2, stride=2)
        self.decoder3 = ComplexConvBlock(512//factor, 256//factor)

        self.up2 = ComplexConvTranspose2d(256//factor, 128//factor, kernel_size=2, stride=2)
        self.decoder2 = ComplexConvBlock(256//factor, 128//factor)

        self.up1 = ComplexConvTranspose2d(128//factor, 64//factor, kernel_size=2, stride=2)
        self.decoder1 = ComplexConvBlock(128//factor, 64//factor)

        self.final_conv = ComplexConv2d(64//factor, out_channels, kernel_size=1)
        self.s = nn.Sigmoid()
        self.cb1 = ComplexCBAM(filters[0],r = 2)
        self.cb2 = ComplexCBAM(filters[1],r = 2)
        self.cb3 = ComplexCBAM(filters[2],r = 2)
        self.cb4 = ComplexCBAM(filters[3],r = 2)
        self.cb5 = ComplexCBAM(filters[4],r = 2)

    def forward(self, x):
        # print(x)
        x = self.lowpass(x)
        input_r = x.real
        input_i = x.imag

        enc1_r, enc1_i = self.encoder1(input_r, input_i)
        pool1_r, pool1_i = self.anti1(enc1_r, enc1_i)
       # pool1_r, pool1_i = self.cb1(pool1_r, pool1_i)

        enc2_r, enc2_i = self.encoder2(pool1_r, pool1_i)
        pool2_r, pool2_i = self.anti2(enc2_r, enc2_i)
       # pool2_r, pool2_i = self.cb2(pool2_r, pool2_i)

        enc3_r, enc3_i = self.encoder3(pool2_r, pool2_i)
        pool3_r, pool3_i = self.anti3(enc3_r, enc3_i)
        #pool3_r, pool3_i = self.cb3(pool3_r, pool3_i)

        enc4_r, enc4_i = self.encoder4(pool3_r, pool3_i)
        pool4_r, pool4_i = self.anti4(enc4_r, enc4_i)
        #pool4_r, pool4_i = self.cb4(pool4_r, pool4_i)

        center_r, center_i = self.center(pool4_r, pool4_i)
        center_r, center_i = self.centercbam(center_r, center_i)
        
        up4_r, up4_i = self.up4(center_r, center_i)
        dec4_r = torch.cat((up4_r, enc4_r), dim=1)
        dec4_i = torch.cat((up4_i, enc4_i), dim=1)
        dec4_r, dec4_i = self.decoder4(dec4_r, dec4_i)

        up3_r, up3_i = self.up3(dec4_r, dec4_i)
        dec3_r = torch.cat((up3_r, enc3_r), dim=1)
        dec3_i = torch.cat((up3_i, enc3_i), dim=1)
        dec3_r, dec3_i = self.decoder3(dec3_r, dec3_i)

        up2_r, up2_i = self.up2(dec3_r, dec3_i)
        dec2_r = torch.cat((up2_r, enc2_r), dim=1)
        dec2_i = torch.cat((up2_i, enc2_i), dim=1)
        dec2_r, dec2_i = self.decoder2(dec2_r, dec2_i)

        up1_r, up1_i = self.up1(dec2_r, dec2_i)
        dec1_r = torch.cat((up1_r, enc1_r), dim=1)
        dec1_i = torch.cat((up1_i, enc1_i), dim=1)
        dec1_r, dec1_i = self.decoder1(dec1_r, dec1_i)

        final_r, final_i = self.final_conv(dec1_r, dec1_i)
        final_r = final_r.float()
        final_i = final_i.float()
        x  = torch.complex(final_r, final_i)
        x = self.pad(x)
        x = torch.real(torch.fft.ifft2(x, norm="backward"))
        x = F.sigmoid(x)
        return x
