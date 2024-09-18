
import torch
import numpy as np
from torch.nn import Module
in_channels = 3
image_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def low_pass_filter(image, r = 8):
    f = torch.tensor(image)
    f = torch.fft.fft2(f)
    f_compress = torch.complex(torch.zeros(in_channels,2*r,r), torch.zeros(in_channels,2*r,r))
    for i in range(in_channels):
        f_compress[i, :r, :r] = f[i, :r, :r]
        f_compress[i, -r:, :r] = f[i, -r:, :r]
    return f_compress


def low_pass_filter2(image, r = 8):
    f = image
    f_compress = torch.complex(torch.zeros(in_channels, image_size, image_size//2), torch.zeros(in_channels, image_size, image_size//2)).to(device)
    for i in range(in_channels):
        f_compress[i, :r, :r] = f[i, :r, :r]
        f_compress[i, -r:, :r] = f[i, -r:, :r]
    return f_compress


def low_pass_filter3(image, r = 8):
    f = image
    f_compress = torch.complex(torch.zeros(f.size(0),1,image_size, image_size//2,2), torch.zeros(f.size(0),1,image_size, image_size//2,2)).to(device)
    f_compress[:,:, :r, :r,:] = f[:,:, :r, :r,:]
    f_compress[:,:, -r:, :r,:] = f[:,:, -r:, :r,:]
    return f_compress

def low_pass_filter4(image, r = 8):
    f = image
    f_compress = torch.complex(torch.zeros(f.size(0),1,image_size, image_size//2), torch.zeros(f.size(0),1,image_size, image_size//2)).to(device)
    f_compress[:,:, :r, :r] = f[:,:, :r, :r]
    f_compress[:,:, -r:, :r] = f[:,:, -r:, :r]
    return f_compress


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    normalized_img = (img - img_min) / (img_max - img_min)
    return normalized_img

class LowPassFilter(Module):
    def __init__(self, r):
        super(LowPassFilter,self).__init__()
        self.r = r
        self.fft = torch.fft.fft2
    def forward(self, x):
        r = self.r

        f = self.fft(x)
        f_compress = torch.complex(torch.zeros(x.shape[0], 3,2*r,r), torch.zeros(x.shape[0],3,2*r,r))
        for i in range(3):
            f_compress[:,i, :r, :r] = f[:, i, :r, :r]
            f_compress[:, i, -r:, :r] = f[:, i, -r:, :r]
        return f_compress.to('cuda')

class ZeroPadding(Module):
    def __init__(self, size, r):
        super(ZeroPadding,self).__init__()
        self.r = r
        self.size = size
    def forward(self, f):
        r = self.r

        f_compress = torch.complex(torch.zeros(f.size(0),1,self.size,self.size), torch.zeros(f.size(0),1,self.size, self.size))
        f_compress[:, :, :r, :r] = f[:,:, :r, :r]
        f_compress[:,:, -r:, :r] = f[:,:, -r:, :r]
        return f_compress.to('cuda')