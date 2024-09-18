import torch
import torch.nn as nn
import torch.nn.functional as F
class ComplexSAM(nn.Module):
    def __init__(self, bias=False):
        super(ComplexSAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x_r, x_i):
        max_r = torch.max(x_r, 1)[0].unsqueeze(1)
        max_i = torch.max(x_i, 1)[0].unsqueeze(1)
        avg_r = torch.mean(x_r, 1).unsqueeze(1)
        avg_i = torch.mean(x_i, 1).unsqueeze(1)
        concat_r = torch.cat((max_r, avg_r), dim=1)
        concat_i = torch.cat((max_i, avg_i), dim=1)
        output_r = self.conv(concat_r) - self.conv(concat_i)
        output_i = self.conv(concat_r) + self.conv(concat_i)
        output_r = torch.sigmoid(output_r) * x_r
        output_i = torch.sigmoid(output_i) * x_i
        return output_r, output_i

class ComplexCAM(nn.Module):
    def __init__(self, channels, r):
        super(ComplexCAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x_r, x_i):
        max_r = F.adaptive_max_pool2d(x_r, output_size=1)
        max_i = F.adaptive_max_pool2d(x_i, output_size=1)
        avg_r = F.adaptive_avg_pool2d(x_r, output_size=1)
        avg_i = F.adaptive_avg_pool2d(x_i, output_size=1)
        b, c, _, _ = x_r.size()
        linear_max_r = self.linear(max_r.view(b, c)).view(b, c, 1, 1)
        linear_max_i = self.linear(max_i.view(b, c)).view(b, c, 1, 1)
        linear_avg_r = self.linear(avg_r.view(b, c)).view(b, c, 1, 1)
        linear_avg_i = self.linear(avg_i.view(b, c)).view(b, c, 1, 1)
        output_r = (linear_max_r + linear_avg_r) * x_r
        output_i = (linear_max_i + linear_avg_i) * x_i
        return output_r, output_i

class ComplexCBAM(nn.Module):
    def __init__(self, channels, r):
        super(ComplexCBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = ComplexSAM(bias=False)
        self.cam = ComplexCAM(channels=self.channels, r=self.r)

    def forward(self, x_r, x_i):
        output_r, output_i = self.cam(x_r, x_i)
        output_r, output_i = self.sam(output_r, output_i)
        return output_r + x_r, output_i + x_i
    