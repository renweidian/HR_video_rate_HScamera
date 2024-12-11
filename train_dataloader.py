

import hdf5storage as h5

from utils import *
import numpy as np
import torch
from torch.utils.data import Dataset





class CAVEHSIDATAprocess(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)

        # PSF2=[[0.0289, 0.0453, 0.0289],
        #      [0.0453, 0.7031, 0.0453],
        #      [0.0289, 0.0453, 0.0289]]
        # PSF2=np.load('psf_cave2.npy')
        PSF2=np.load(r'E:\code\demosaic_and_fusion\simulation_fusion\psf\psf_icvl_30_2.npy')
        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(len(imglist)):
            img = h5.loadmat(path + imglist[i])
            img1 = img["rad"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))

            b,w,h=HRHSI.shape
            nw,nh=w//16,h//16
            HRHSI=HRHSI[:,:nw*16,:nh*16]
            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            SNRh = 30
            sigma = np.sqrt(np.sum(HSI_LR ** 2) / (10 ** (SNRh / 10)) / (HSI_LR.shape[0] * HSI_LR.shape[1] * HSI_LR.shape[2]));
            HSI_LR=HSI_LR+sigma*np.random.randn(HSI_LR.shape[0],HSI_LR.shape[1],HSI_LR.shape[2])



            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            SNRh = 35
            sigma = np.sqrt(np.sum(MSI ** 2) / (10 ** (SNRh / 10)) / (MSI.shape[0] * MSI.shape[1] * MSI.shape[2]));
            MSI = MSI + sigma * np.random.randn(MSI.shape[0], MSI.shape[1], MSI.shape[2])


            HSId= Gaussian_downsample(HSI_LR, PSF2, downsample_factor)
            SNRh = 30
            sigma = np.sqrt(np.sum(HSId ** 2) / (10 ** (SNRh / 10)) / (HSId.shape[0] * HSId.shape[1] * HSId.shape[2]));
            HSId=HSId+sigma*np.random.randn(HSId.shape[0],HSId.shape[1],HSId.shape[2])


            MSId=Gaussian_downsample(MSI, PSF2, downsample_factor)
            SNRh = 35
            sigma = np.sqrt(np.sum(MSId ** 2) / (10 ** (SNRh / 10)) / (MSId.shape[0] * MSId.shape[1] * MSId.shape[2]));
            MSId = MSId + sigma * np.random.randn(MSId.shape[0], MSId.shape[1], MSId.shape[2])




            for j in range(0, HSI_LR.shape[1] - training_size + 1, stride):
                for k in range(0, HSI_LR.shape[2] - training_size + 1, stride):
                    temp_hrhs = HSI_LR[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSId[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSId[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float16)
                    temp_hrms=temp_hrms.astype(np.float16)
                    temp_lrhs=temp_lrhs.astype(np.float16)
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



class CAVEHSIDATAprocess2(Dataset):
    def __init__(self, path, R, training_size, stride, downsample_factor, PSF):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        imglist = os.listdir(path)

        train_hrhs = []
        train_hrms = []
        train_lrhs = []

        for i in range(len(imglist)):
            img = h5.loadmat(path + imglist[i])
            img1 = img["rad"]
            img1 = img1 / img1.max()

            HRHSI = np.transpose(img1, (2, 0, 1))

            b, w, h = HRHSI.shape
            nw, nh = w // 16, h // 16
            HRHSI = HRHSI[:, :nw * 16, :nh * 16]

            # hwc-chw
            HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
            SNRh = 30
            sigma = np.sqrt(np.sum(HSI_LR ** 2) / (10 ** (SNRh / 10)) / (HSI_LR.shape[0] * HSI_LR.shape[1] * HSI_LR.shape[2]));
            HSI_LR=HSI_LR+sigma*np.random.randn(HSI_LR.shape[0],HSI_LR.shape[1],HSI_LR.shape[2])



            MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
            SNRh = 35
            sigma = np.sqrt(np.sum(MSI ** 2) / (10 ** (SNRh / 10)) / (MSI.shape[0] * MSI.shape[1] * MSI.shape[2]));
            MSI = MSI + sigma * np.random.randn(MSI.shape[0], MSI.shape[1], MSI.shape[2])

            for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
                for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                    temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                    temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
                    temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                                int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                    # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                    temp_hrhs=temp_hrhs.astype(np.float16)
                    temp_hrms=temp_hrms.astype(np.float16)
                    temp_lrhs=temp_lrhs.astype(np.float16)
                    train_hrhs.append(temp_hrhs)
                    train_hrms.append(temp_hrms)
                    train_lrhs.append(temp_lrhs)
            # train_hrhs.append(HRHSI)
            # train_hrms.append(MSI)
            # train_lrhs.append(HSI_LR)
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


#
#
# class RemoteAprocess(Dataset):
#     def __init__(self, path, R, training_size, stride, downsample_factor, PSF):
#         """
#         :param path:
#         :param R: 光谱响应矩阵
#         :param training_size:
#         :param stride:
#         :param downsample_factor:
#         :param PSF: 高斯模糊核
#         :param num: cave=32个场景
#         """
#         imglist = os.listdir(path)
#
#         train_hrhs = []
#         train_hrms = []
#         train_lrhs = []
#
#         for i in range(len(imglist)):
#             img = h5.loadmat(path + imglist[i])
#             img1 = img["rad"]
#             img1 = img1 / img1.max()
#
#             HRHSI = np.transpose(img1, (2, 0, 1))
#
#             b, w, h = HRHSI.shape
#             nw, nh = w // 16, h // 16
#             HRHSI = HRHSI[:, :nw * 16, :nh * 16]
#
#             # hwc-chw
#             HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)
#             SNRh = 15
#             sigma = np.sqrt(np.sum(HSI_LR ** 2) / (10 ** (SNRh / 10)) / (HSI_LR.shape[0] * HSI_LR.shape[1] * HSI_LR.shape[2]));
#             HSI_LR=HSI_LR+sigma*np.random.randn(HSI_LR.shape[0],HSI_LR.shape[1],HSI_LR.shape[2])
#
#
#
#             MSI = np.tensordot(R, HRHSI, axes=([1], [0]))
#             SNRh = 20
#             sigma = np.sqrt(np.sum(MSI ** 2) / (10 ** (SNRh / 10)) / (MSI.shape[0] * MSI.shape[1] * MSI.shape[2]));
#             MSI = MSI + sigma * np.random.randn(MSI.shape[0], MSI.shape[1], MSI.shape[2])
#
#             for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
#                 for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
#                     temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
#                     temp_hrms = MSI[:, j:j + training_size, k:k + training_size]
#                     temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
#                                 int(k / downsample_factor):int((k + training_size) / downsample_factor)]
#                     # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
#                     temp_hrhs=temp_hrhs.astype(np.float16)
#                     temp_hrms=temp_hrms.astype(np.float16)
#                     temp_lrhs=temp_lrhs.astype(np.float16)
#                     train_hrhs.append(temp_hrhs)
#                     train_hrms.append(temp_hrms)
#                     train_lrhs.append(temp_lrhs)
#             # train_hrhs.append(HRHSI)
#             # train_hrms.append(MSI)
#             # train_lrhs.append(HSI_LR)
#         train_hrhs = torch.Tensor(np.array(train_hrhs))
#         train_hrms = torch.Tensor(np.array(train_hrms))
#         train_lrhs = torch.Tensor(np.array(train_lrhs))
#
#         # print(train_hrhs.shape, train_hrms.shape)
#         self.train_hrhs_all = train_hrhs
#         self.train_hrms_all = train_hrms
#         self.train_lrhs_all = train_lrhs
#
#     def __getitem__(self, index):
#         train_hrhs = self.train_hrhs_all[index, :, :, :]
#         train_hrms = self.train_hrms_all[index, :, :, :]
#         train_lrhs = self.train_lrhs_all[index, :, :, :]
#         # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
#         return train_lrhs, train_hrms, train_hrhs
#
#     def __len__(self):
#         return self.train_hrhs_all.shape[0]