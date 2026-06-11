import torch
import torch.nn as nn
import torch.nn.functional as F


backwarp_tenGrid = {}
def warp(tenInput, tenFlow):
    ## tuple key is instantly hashed compared to string casting
    k = (tenFlow.device, tenFlow.shape[2], tenFlow.shape[3], tenFlow.dtype)
    
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=k[0], dtype=k[3]).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=k[0], dtype=k[3]).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)

    ## OPTM: reduced intermediate tensor creation
    tenFlow_mapped = torch.cat([
        tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
        tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)
    ], 1)

    g = (backwarp_tenGrid[k] + tenFlow_mapped).permute(0, 2, 3, 1)
    
    return F.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.LeakyReLU(0.2, inplace=True) ## OPTM: inplace
    )

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True) ## OPTM: inplace

    def forward(self, x):
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)

class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        ## OPTM: in-place multiplication and addition
        out = self.conv(x)
        out.mul_(self.beta).add_(x)
        return self.relu(out)

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
            ResConv(c), ResConv(c), ResConv(c), ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4*13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1.0):
        ## OPTM: bypass interpolation if scale is exactly 1.0 (saves heavy compute)
        if scale != 1.0:
            x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
            if flow is not None:
                flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False).mul_(1. / scale)
        
        if flow is not None:
            x = torch.cat((x, flow), 1)
            
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        
        if scale != 1.0:
            tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
            
        flow = tmp[:, :4].mul_(scale) ## OPTM: inplace
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+8, c=192)
        self.block1 = IFBlock(8+4+8+8, c=128)
        self.block2 = IFBlock(8+4+8+8, c=96)
        self.block3 = IFBlock(8+4+8+8, c=64)
        self.block4 = IFBlock(8+4+8+8, c=32)
        self.encode = Head()
        ## stripped all dead training parameters (teacher, caltime, ...)

    def forward(self, x, timestep=0.5, scale_list=[8, 4, 2, 1, 1]):
        ## stripped dynamic branches. Optimized for static inference
        img0 = x[:, :3]
        img1 = x[:, 3:6]

        ## OPTM: generate timestep tensor directly in memory instead of heavy cloning
        if not torch.is_tensor(timestep):
            timestep_tensor = torch.full((x.shape[0], 1, img0.shape[2], img0.shape[3]), timestep, dtype=x.dtype, device=x.device)
        else:
            timestep_tensor = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0)
        f1 = self.encode(img1)
        
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        feat = None
        
        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        
        ## OPTM: variables are overwritten in place rather than appended to lists..
        ## this allows PyTorch to immediately garbage-collect the VRAM from previous blocks
        for i in range(5):
            if flow is None:
                flow, mask, feat = block[i](torch.cat((img0, img1, f0, f1, timestep_tensor), 1), None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = block[i](torch.cat((warped_img0, warped_img1, wf0, wf1, timestep_tensor, mask, feat), 1), flow, scale=scale_list[i])
                flow = flow + fd
                
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            
        mask = torch.sigmoid(mask)
        final_merged = warped_img0 * mask + warped_img1 * (1 - mask)
        
        ## we mock the return structure `_, _, merged` so it doesn't break our wrapper logic `[2][-1]`
        ## by only passing the final image in the list, we don't return gigabytes of dead tensors..
        return None, None, [None, None, None, None, final_merged]