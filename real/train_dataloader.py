import glob
import os

import hdf5storage as h5
from torch.utils import data

from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset


def to_position_order(ori):
    color_images2 = torch.zeros_like(ori)
    #
    #
    color_images2[:, 0, :, :] = ori[:, 10, :, :]
    color_images2[:, 1, :, :] = ori[:, 6, :, :]
    color_images2[:, 2, :, :] = ori[:, 2, :, :]
    color_images2[:, 3, :, :] = ori[:, 14, :, :]
    color_images2[:, 4, :, :] = ori[:, 9, :, :]
    color_images2[:, 5, :, :] = ori[:, 5, :, :]
    color_images2[:, 6, :, :] = ori[:, 1, :, :]
    color_images2[:, 7, :, :] = ori[:, 13, :, :]
    color_images2[:, 8, :, :] = ori[:, 8, :, :]

    color_images2[:, 9, :, :] = ori[:, 4, :, :]
    color_images2[:, 10, :, :] = ori[:, 0, :, :]
    color_images2[:, 11, :, :] = ori[:, 12, :, :]
    color_images2[:, 12, :, :] = ori[:, 11, :, :]
    color_images2[:, 13, :, :] = ori[:, 7, :, :]
    color_images2[:, 14, :, :] = ori[:, 3, :, :]
    color_images2[:, 15, :, :] = ori[:, 15, :, :]
    return color_images2


unps = torch.nn.PixelUnshuffle(4)
class RealdatasetF(Dataset):
    def __init__(self, path,training_size, stride, downsample_factor):

        imglist = os.listdir(path)
        train_hrms = []
        train_lrhs = []

        for i in range(len(imglist)):
            img = h5.loadmat(path + imglist[i])
            raw0 = img['mosaic'].astype(np.float32)  #
            pan = img['pan'].astype(np.float32)  #
            # raw0 = raw0[:2008 // 1, :2008 // 1, :]
            # pan = pan[:2008, :2008, :]
            raw0 = raw0 / 255.0
            pan = pan / 255.0

            raw0 = torch.from_numpy(raw0).float().permute(2, 0, 1).unsqueeze(0)
            lrhsi = unps(raw0)
            HSI_LR = to_position_order(lrhsi).squeeze(0).numpy()
            MSI=pan.astype(np.float32).transpose(2,0,1)    # c,w,h



            for j in range(0, MSI.shape[1] - training_size + 1, stride):
                for k in range(0, MSI.shape[2] - training_size + 1, stride):
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)

        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_lrhs, train_hrms

    def __len__(self):
        return self.train_hrms_all.shape[0]






class RealdatasetF_Z(Dataset):
    def __init__(self, path,training_size, stride, downsample_factor):
        imglist = os.listdir(path)
        train_hrms = []
        train_lrhs = []
        train_hrhs=[]
        PSF2 = np.array([[0.0020, 0.0021, 0.0023, 0.0024, 0.0025, 0.0025, 0.0025, 0.0024, 0.0023,
                 0.0021, 0.0020],
                [0.0021, 0.0093, 0.0100, 0.0105, 0.0108, 0.0109, 0.0108, 0.0105, 0.0100,
                 0.0093, 0.0021],
                [0.0023, 0.0100, 0.0107, 0.0112, 0.0115, 0.0117, 0.0115, 0.0112, 0.0107,
                 0.0100, 0.0023],
                [0.0024, 0.0105, 0.0112, 0.0118, 0.0121, 0.0122, 0.0121, 0.0118, 0.0112,
                 0.0105, 0.0024],
                [0.0025, 0.0108, 0.0115, 0.0121, 0.0125, 0.0126, 0.0125, 0.0121, 0.0115,
                 0.0108, 0.0025],
                [0.0025, 0.0109, 0.0117, 0.0122, 0.0126, 0.0127, 0.0126, 0.0122, 0.0117,
                 0.0109, 0.0025],
                [0.0025, 0.0108, 0.0115, 0.0121, 0.0125, 0.0126, 0.0125, 0.0121, 0.0115,
                 0.0108, 0.0025],
                [0.0024, 0.0105, 0.0112, 0.0118, 0.0121, 0.0122, 0.0121, 0.0118, 0.0112,
                 0.0105, 0.0024],
                [0.0023, 0.0100, 0.0107, 0.0112, 0.0115, 0.0117, 0.0115, 0.0112, 0.0107,
                 0.0100, 0.0023],
                [0.0021, 0.0093, 0.0100, 0.0105, 0.0108, 0.0109, 0.0108, 0.0105, 0.0100,
                 0.0093, 0.0021],
                [0.0020, 0.0021, 0.0023, 0.0024, 0.0025, 0.0025, 0.0025, 0.0024, 0.0023,
                 0.0021, 0.0020]])
        # the PSF2 is get from the D_F_net.ssf_net.

        for i in range(len(imglist)):
            img = h5.loadmat(path + imglist[i])
            raw0 = img['mosaic'].astype(np.float32)  # 原始的一个通道raw
            pan = img['pan'].astype(np.float32)  # 对应的全色图像
            # raw0 = raw0[:2008 // 1, :2008 // 1, :]
            # pan = pan[:2008, :2008, :]
            raw0 = raw0 / 255.0
            pan = pan / 255.0

            raw0 = torch.from_numpy(raw0).float().permute(2, 0, 1).unsqueeze(0)
            lrhsi = unps(raw0)
            HSI_LR = to_position_order(lrhsi).squeeze(0).numpy()
            MSI=pan.astype(np.float32).transpose(2,0,1)    # c,w,h

            c,w,h=HSI_LR.shape
            nw1,nh1=w//8,h//8
            c,w,h=MSI.shape
            nw2,nh2=w//8,h//8
            nw=min(nw1,nw2)
            nh=min(nh1,nh2)
            HSI_LR=HSI_LR[:,:nw*8,:nh*8]
            MSI=MSI[:,:nw*8*8,:nh*8*8]

            HSId = Gaussian_downsample(HSI_LR, PSF2, downsample_factor)
            MSId = Gaussian_downsample(MSI, PSF2, downsample_factor)

            for j in range(0, HSI_LR.shape[1] - training_size + 1, stride):
                for k in range(0, HSI_LR.shape[2] - training_size + 1, stride):
                    temp_hrhs = HSI_LR[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSId[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSId[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float32)
                    temp_hrms=temp_hrms.astype(np.float32)
                    temp_lrhs=temp_lrhs.astype(np.float32)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))
        train_lrhs = torch.Tensor(np.array(train_lrhs))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_hrms_all = train_hrms
        self.train_lrhs_all = train_lrhs

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return train_lrhs, train_hrms, train_hrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]