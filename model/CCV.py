import torch
import torch.nn as nn
from model.antialias import Sequential_complex, ComplexLowpass
from model.block_complex import Complex_AdaptiveAvgPool2d, ComplexBatchNorm1d, ComplexBatchNorm2d, ComplexConv2d, ComplexConvTranspose2d, ComplexDropBlock2D, ComplexDropout, ComplexDropout2d, ComplexELU, ComplexLinear, ComplexMaxPool2d, \
                                ComplexReLU, ComplexSequential, Conv2d, ConvTranspose2d, complex_AdaptiveAvgPool2d, complex_cat, complex_dropout, complex_dropout2d, complex_elu, complex_max_pool2d, complex_relu, complex_up, complex_up_16, \
                                complex_up_4, complex_up_8, _ComplexBatchNorm, NaiveComplexBatchNorm1d, NaiveComplexBatchNorm2d
from model.ccbam import ComplexCAM, ComplexSAM, ComplexCBAM
from utils.filter import low_pass_filter, low_pass_filter2, low_pass_filter3, low_pass_filter4
from torch.nn import Module
image_size = 256
in_channels = 3
out_channels = 1
r = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
class ComplexConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,filter_size=3, block_size = 3 , drop = False):
        super().__init__()

        #residual function
        self.residual_function = Sequential_complex(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True), 
            ComplexConv2d(out_channels, out_channels * ComplexConvBlock.expansion, kernel_size=3, padding=1, bias=False),
            
        )
        self.dropblock = Sequential_complex(
            ComplexDropBlock2D(block_size = block_size),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True), 
        )
    def forward(self, x1, x2, drop = False):
        x3, x4 = self.residual_function(x1, x2) 
        if drop:
            x3, x4 = self.dropblock(x3,x4)
        return x3, x4
    

    
