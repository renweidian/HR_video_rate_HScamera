import time

import torch
import torch.nn as nn

import numpy as np

import torch.nn.functional as F
from einops import rearrange

from utils import create_F

def create_gaussian_kernel(sigma, k_size=(7, 7)):
    l = []
    for i in range(k_size[0]):
        for j in range(k_size[1]):
            temp_x = i - k_size[0] // 2
            temp_y = j - k_size[1] // 2
            temp = 1 / (2 * np.pi * sigma ** 2) * torch.exp(-(temp_x ** 2 + temp_y ** 2) / (2 * sigma ** 2))
            l.append(temp)
    kernel = torch.stack(l).view(k_size)
    return kernel




# 定义 CustomModel
class SSF(nn.Module):
    def __init__(self, hsi_c, pan_c,up_scale):
        super(SSF, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0),requires_grad=True)
        self.half_k = nn.Parameter(torch.tensor(1.5),requires_grad=True)
        self.spek = nn.Parameter(torch.rand(hsi_c, pan_c))
        self.up_scale=up_scale

    def forward(self):
        sigma = self.sigma
        half_k = self.half_k

        all_spak = create_gaussian_kernel(sigma, k_size=(15, 15))

        f_hk = torch.floor(half_k).int()
        c_hk = torch.ceil(half_k).int()

        weight = half_k - f_hk.float()  # 权重

        f_k = all_spak[7 - f_hk:7 + f_hk + 1, 7 - f_hk:7 + f_hk + 1]
        c_k = all_spak[7 - c_hk:7 + c_hk + 1, 7 - c_hk:7 + c_hk + 1]
        f_k = F.pad(f_k, (1, 1, 1, 1))     # 确保两个核一样的尺寸
        spak = f_k * (1 - weight) + weight * c_k     # 模糊核的插值

        spak = spak / spak.sum()             # 归一化
        spak1 = spak.unsqueeze(0).unsqueeze(0)   #增加维度
        spak1 = spak1.expand(self.spek.shape[0],-1, -1, -1)   # 空间模糊核

        spek1=torch.abs(self.spek)    # 确保为正数
        spek1=spek1/torch.sum(spek1,dim=0)    # 归一化 ,且和为1
        return spak1,spek1,half_k,sigma



class ChannelPool(nn.Module):
    def forward(self, x):
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        # return torch.mean(x,1)
        return torch.mean(x, 1).unsqueeze(1)
class Light_Conv(nn.Module):
    def __init__(self, msfa_size,channel):
        super(Light_Conv, self).__init__()
        self.compress = ChannelPool()
        self.shuffledown = nn.PixelUnshuffle(msfa_size)
        self.shuffleup=nn.PixelShuffle(msfa_size)
        self.share_conv=nn.Sequential(
            nn.Conv2d(msfa_size**2,msfa_size**2,3,1,1),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(1, msfa_size**1, 3, 1, 1),
            # nn.LeakyReLU(0.1)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(channel,channel,3,1,1),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(channel, channel, 3, 1, 1),
            # nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        res = x
        N, C, H, W = x.size()
        x = x.view(N * C, 1, H, W)  # N, C, H, W to N*C, 1, H, W
        sq_x = self.shuffledown(x)  # N*C, 1, H, W to N*C, 16, H/4, W/4
        x=self.share_conv(sq_x)
        x=self.shuffleup(x)
        x=res+x.view(N , C, H, W)
        out=self.conv(x)
        return out


class AWSCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AWSCA, self).__init__()
        self.pan_conv = nn.Conv2d(3, channel, 1, bias=False)
        self.lr_conv = nn.Conv2d(32, channel, 1, bias=False)

        self.spa_conv=nn.AdaptiveAvgPool2d(1)
        self.spe_conv=nn.Conv2d(channel, 1, 1, bias=False)

        self.conv_down = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=6, stride=4, padding=2),
        )

        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=2)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        self.conv=nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x,lr,pan):
        b, c, h, w = x.size()
        input_x = x

        q_s=self.pan_conv(pan)    # 16*64*64
        k_s=self.spa_conv(x)      # 16*1*1

        q_c=self.lr_conv(lr)      # 16*8*8
        d_x=self.conv_down(x)       # 16*8*8
        k_c=self.spe_conv(d_x)      # 1*8*8

        q_s=q_s.reshape(b, c, h*w).transpose(-2, -1)
        k_s=k_s.reshape(b, c, 1)
        sa=torch.matmul(q_s,k_s).reshape(b,1, h, w)   # b,wh*1

        b2, c2, h2, w2 = d_x.size()
        q_c=q_c.reshape(b, c, h2*w2)
        k_c=k_c.reshape(b, 1, h2*w2).transpose(-2, -1)
        ca=torch.matmul(q_c,k_c).reshape(b,c)   # b,c*1

        ca=self.fc(ca).reshape(b,c,1,1)

        sa=self.conv(sa)

        x=sa*x
        x=self.final_conv(x)
        out=ca*x
        return out



class Att_Conv(nn.Module):
    def __init__(self,channel,msfa_size):
        super(Att_Conv, self).__init__()
        self.att=AWSCA(channel)

        self.compress = ChannelPool()
        self.shuffledown = nn.PixelUnshuffle(msfa_size)
        self.shuffleup = nn.PixelShuffle(msfa_size)
        self.share_conv = nn.Sequential(
            nn.Conv2d(msfa_size ** 2, msfa_size ** 2, 3, 1, 1),
            nn.LeakyReLU(0.2),

        )
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2),

        )

    def forward(self,  x,lr,pan):
        res = x
        N, C, H, W = x.size()
        x = x.view(N * C, 1, H, W)  # N, C, H, W to N*C, 1, H, W
        sq_x = self.shuffledown(x)  # N*C, 1, H, W to N*C, 16, H/4, W/4
        x=self.share_conv(sq_x)
        x=self.shuffleup(x)
        x=x.view(N , C, H, W)
        out=self.conv(x)
        out=self.att(out,lr,pan)+res
        return out


class Fusion(nn.Module):
    def __init__(self, hsi_chnnel=16, pan_channel=1, num_res=6, num_fm=64,up_scale=2):
        super(Fusion, self).__init__()
        self.up_scale=up_scale
        self.msfa_size=4

        self.pan_ups = nn.PixelUnshuffle(2)
        self.lr_ps = nn.PixelShuffle(2)
        self.conv_32=nn.Conv2d(31, 32, kernel_size=1, stride=1)
        self.ps=nn.PixelShuffle(2)
        self.psconv=nn.Conv2d(hsi_chnnel*2, hsi_chnnel*4, kernel_size=1, stride=1)


        self.cat1=nn.Sequential(
            nn.Conv2d(4*pan_channel+hsi_chnnel//4, hsi_chnnel*2, kernel_size=1, stride=1),
        )
        self.ligth_conv1=Light_Conv(self.msfa_size,hsi_chnnel*2)

        self.cat2 = nn.Sequential(
            nn.Conv2d(hsi_chnnel+pan_channel, hsi_chnnel*2, kernel_size=1, stride=1),
        )

        self.ligth_conv2=Light_Conv(self.msfa_size,hsi_chnnel*2)

        self.attconv1=Att_Conv(hsi_chnnel*2,4)
        self.attconv2 = Att_Conv(hsi_chnnel*2,4)
        # self.attconv3 = Att_Conv(hsi_chnnel*2,4)
        # self.attconv4 = Att_Conv(hsi_chnnel*2,4)

        self.final_conv = nn.Conv2d(hsi_chnnel*2, 31, kernel_size=3, stride=1, padding=1)

    def forward(self,lr,pan):
        lr_up=torch.nn.functional.interpolate(lr,scale_factor=self.up_scale,mode='bicubic')

        #
        lr=self.conv_32(lr)
        pan_f1=self.pan_ups(pan)     # 32*32*4
        lr_f1=self.lr_ps(lr)     # 32*32*1


        # 32*32
        fea_1=torch.cat([pan_f1,lr_f1],dim=1)
        fea_1=self.cat1(fea_1)
        fea_1=self.ligth_conv1(fea_1)

        # 64*64
        fea_1=self.psconv(fea_1)


        fea2=self.ps(fea_1)
        fea_2=torch.cat([fea2,pan],dim=1)
        fea_2=self.cat2(fea_2)
        fea_2=self.ligth_conv2(fea_2)

        # 融合模块
        fu=self.attconv1(fea_2,lr,pan)
        fu = self.attconv2(fu,lr,pan)
        # fu = self.attconv3(fu,lr,pan)
        # fu = self.attconv4(fu,lr,pan)
        out=self.final_conv(fu)+lr_up
        return out


class D_F_net(nn.Module):
    def __init__(self, hsi_spectral=16, pan_chenenl=1,num_res=6, num_fm=64,up_scale=2):
        super(D_F_net, self).__init__()

        self.fusion_net=Fusion(hsi_spectral, pan_chenenl,num_res, num_fm,up_scale)
        self.ssf_net=SSF(31, 3,up_scale)
        self.up_scale=up_scale

    def spa_and_spe_d(self, lrhsi,pan):
        for param in self.ssf_net.parameters():
            param.requires_grad = True
        spak1, spek1, half_k, sigma = self.ssf_net()
        pan_d1 = torch.matmul(lrhsi.permute(0, 2, 3, 1), spek1).permute(0, 3, 1, 2)  # 融合的高分辨率高光谱经过光谱响应，生成伪全色

        # 融合的高分辨率高光谱经过空间响应，生成伪低分辨率高光谱
        pan_d2 = F.conv2d(pan, spak1[0:3,:,:,:], bias=None, stride=(1, 1), padding='same', dilation=1,
                           groups=3)  # 空间模糊
        pan_d2 = pan_d2[:, :, 0:: self.up_scale, 0:: self.up_scale]  # 空间下采样
        return pan_d1, pan_d2, half_k, sigma

    def fusion(self,hsi, pan):
        fusion_hsi=self.fusion_net(hsi, pan)
        return fusion_hsi


    def fusion_d(self, fusion_hsi):
        for param in self.ssf_net.parameters():
            param.requires_grad = False

        spak1, spek1, half_k, sigma = self.ssf_net()
        pan_p1 = torch.matmul(fusion_hsi.permute(0, 2, 3, 1), spek1).permute(0, 3, 1, 2)  # 融合的高分辨率高光谱经过光谱响应，生成伪全色

        # 融合的高分辨率高光谱经过空间响应，生成伪低分辨率高光谱
        lrhsi_p2 = F.conv2d(fusion_hsi, spak1, bias=None, stride=(1, 1), padding=spak1.shape[2] // 2, dilation=1,
                           groups=31)  # 空间模糊
        lrhsi_p2 = lrhsi_p2[:, :, 0:: self.up_scale, 0:: self.up_scale]  # 空间下采样

        return pan_p1, lrhsi_p2, half_k, sigma


    def forward(self,hsi,pan):
        fusion_hsi=self.fusion(hsi,pan)
        pan_p1, lrhsi_p2,_,_=self.fusion_d(fusion_hsi)
        pan_d1, pan_d2, half_k, sigma=self.spa_and_spe_d(hsi,pan)
        return fusion_hsi,pan_d1, pan_d2, pan_p1, lrhsi_p2, half_k, sigma


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("*" * 10 + "运行的是 HSRnet.py文件" + "*" * 10)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    R = create_F()
    print("测试 Cross_Guide_Fusion")
    hsi = torch.randn((2, 31,16, 16), device=device)
    msi = torch.randn((2,3, 64, 64), device=device)

    model2 = D_F_net(32,3,4,32,4).to(device)
    model2.train()



    total = sum([param.nelement() for param in model2.fusion_net.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1e6))
    # for i in range(10):
    #     model2.eval()
    #     with torch.no_grad():
    #         time1 = time.time()
    #         fusion_hsi,pan_d1, pan_d2, pan_p1, pan_p2, half_k, sigma= model2(hsi, msi)
    #         print(fusion_hsi.shape, pan_d1.shape, pan_d2.shape, pan_p1.shape, pan_p2.shape, half_k, sigma)
    #         time2 = time.time()
    #         print(time2 - time1)

    # for i in range(50):
    #     model2.eval()
    #     with torch.no_grad():
    #         time1 = time.time()
    #         fusion_hsi= model2(hsi, msi)
    #         time2 = time.time()
    #         # print(fusion_hsi.shape)
    #         print(time2 - time1)