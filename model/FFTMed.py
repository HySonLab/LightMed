class ComplexDropBlock2D(nn.Module):
    def __init__(self, drop_prob = 0.2, block_size = 3):
        super(ComplexDropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, input_r, input_i):
        # shape: (bsize, channels, height, width)
        assert input_r.size() == input_i.size(), "Real and imaginary parts must have the same size"
        assert input_r.dim() == 4, "Expected input with 4 dimensions (bsize, channels, height, width)"
        
        if not self.training or self.drop_prob == 0.:
            return input_r, input_i
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size ** 2)
            # sample mask
            mask = (torch.rand(input_r.shape[0], *input_r.shape[2:], device=input_r.device) < gamma).float()
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask to both real and imaginary parts
            out_r = input_r * block_mask[:, None, :, :]
            out_i = input_i * block_mask[:, None, :, :]
            # scale output
            scale = block_mask.numel() / block_mask.sum()
            out_r = out_r * scale
            out_i = out_i * scale
            return out_r, out_i

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask
    

class ComplexResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,filter_size=3, block_size = 3 , drop = False, fre = False):
        super().__init__()

        #residual function
        self.residual_function = Sequential_complex(
            ComplexConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True), 
            ComplexConv2d(out_channels, out_channels * ComplexResidualBlock.expansion, kernel_size=3, padding=1, bias=False),
            
        )
        self.dropblock = Sequential_complex(
            ComplexDropBlock2D(block_size = block_size),
            ComplexBatchNorm2d(out_channels),
            ComplexReLU(inplace=True), 
        )
        if fre:
            self.fre = ComplexCBAM(in_channels, 2)
            self.conv = ComplexConv2d(in_channels, out_channels * ComplexResidualBlock.expansion, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x1, x2, fre = False, drop = False):
        x3, x4 = self.residual_function(x1, x2) 
        if drop:
            x3, x4 = self.dropblock(x3,x4)
        if fre:
            x5, x6 = self.fre(x1,x2)
            x5, x6 = self.conv(x5,x6)
            return x3 + x5, x4 + x6 
        return x3,x4 






class RefUnet(nn.Module):
    def __init__(self,in_ch,inc_ch):
        super(RefUnet, self).__init__()
        self.conv0 = nn.Conv2d(in_ch,inc_ch,3,padding=1)
        
        self.conv1 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn1 = nn.BatchNorm2d(inc_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn2 = nn.BatchNorm2d(inc_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn3 = nn.BatchNorm2d(inc_ch)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(inc_ch,inc_ch,3,padding=1)
        self.bn5 = nn.BatchNorm2d(inc_ch)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d3 = nn.Conv2d(inc_ch*2,inc_ch,3,padding=1)
        self.bn_d3 = nn.BatchNorm2d(inc_ch)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(inc_ch*2,inc_ch,3,padding=1)
        self.bn_d2 = nn.BatchNorm2d(inc_ch)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(inc_ch*2,inc_ch,3,padding=1)
        self.bn_d1 = nn.BatchNorm2d(inc_ch)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(inc_ch,1,3,padding=1)

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bicubic')

    def forward(self,x):

        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        hx = self.upscore2(hx5)

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx,hx3),1))))
        hx = self.upscore2(d3)

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx,hx2),1))))
        hx = self.upscore2(d2)

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx,hx1),1))))

        residual = self.conv_d0(d1)

        return x + residual

class FFTMed(Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FFTMed, self).__init__()
        factor = 4 
        filters = [64//factor, 128//factor, 256//factor, 512//factor, 1024//factor]
        self.encoder1 = ComplexResidualBlock(in_channels, 64//factor, fre = True)
        # self.pool1 = ComplexMaxPool2d(kernel_size=2, stride=2)  
        self.anti1 = ComplexLowpass(filt_size=3, stride=2, channels=64//factor)

        self.encoder2 = ComplexResidualBlock(64//factor, 128//factor, fre = True)
        # self.pool2 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti2 = ComplexLowpass(filt_size=3, stride=2, channels=128//factor)

        self.encoder3 = ComplexResidualBlock(128//factor, 256//factor, drop = True, fre = True)
        # self.pool3 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti3 = ComplexLowpass(filt_size=3, stride=2, channels=256//factor)

        self.encoder4 = ComplexResidualBlock(256//factor, 512//factor, drop = True, fre = True)
        # self.pool4 = ComplexMaxPool2d(kernel_size=2, stride=2)
        self.anti4 = ComplexLowpass(filt_size=3, stride=2, channels=512//factor)

        self.center = ComplexResidualBlock(512//factor, 1024//factor)
        # self.centercbam = ComplexCBAM(1024//factor,1024//factor)
        
        self.up4 = ComplexConvTranspose2d(1024//factor, 512//factor, kernel_size=2, stride=2)
        self.decoder4 = ComplexResidualBlock(1024//factor, 512//factor)

        self.up3 = ComplexConvTranspose2d(512//factor, 256//factor, kernel_size=2, stride=2)
        self.decoder3 = ComplexResidualBlock(512//factor, 256//factor)

        self.up2 = ComplexConvTranspose2d(256//factor, 128//factor, kernel_size=2, stride=2)
        self.decoder2 = ComplexResidualBlock(256//factor, 128//factor)

        self.up1 = ComplexConvTranspose2d(128//factor, 64//factor, kernel_size=2, stride=2)
        self.decoder1 = ComplexResidualBlock(128//factor, 64//factor)

        self.final_conv = ComplexConv2d(64//factor, out_channels, kernel_size=1)
        self.refunet = RefUnet(out_channels,8) 
        self.s = nn.Sigmoid()
        # self.cb1 = ComplexCBAM(filters[0],r = 2)
        # self.cb2 = ComplexCBAM(filters[1],r = 2)
        # self.cb3 = ComplexCBAM(filters[2],r = 2)
        # self.cb4 = ComplexCBAM(filters[3],r = 2)
        # self.cb5 = ComplexCBAM(filters[4],r = 2)

    

    def forward(self, x):
        # print(x)
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
        # pool3_r, pool3_i = self.cb3(pool3_r, pool3_i)

        enc4_r, enc4_i = self.encoder4(pool3_r, pool3_i)
        pool4_r, pool4_i = self.anti4(enc4_r, enc4_i)
        # pool4_r, pool4_i = self.cb4(pool4_r, pool4_i)

        center_r, center_i = self.center(pool4_r, pool4_i)
        # center_r, center_i = self.centercbam(center_r, center_i) 
        
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
        freq_x  = torch.complex(final_r, final_i)

        x = low_pass_filter4(freq_x, r=r)
        x = torch.fft.irfft2(x, norm="backward", s = (image_size, image_size))
        x = torch.real(x)
        x = self.refunet(x)
        x = F.sigmoid(x)
        return freq_x, x

# Example usage:
model = FFTMed(in_channels, out_channels).to(device)
x = torch.complex(torch.randn(1, in_channels, 2*r, r),torch.randn(1, in_channels, 2*r, r)).to(device) 
freq, output = model(x) 
print(freq.shape) 
    
sum(p.numel() for p in model.parameters()) 
