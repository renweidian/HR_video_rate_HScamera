import os
import time

import hdf5storage

from fusion_net import D_F_net

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import numpy as np
import torch

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



if __name__ == '__main__':
    # 路径参数


    # data_name = 'cave'
    path1 = r'F:\realdata\new_data\reg\mat\test/'

    #

    model= D_F_net(16, 1, 6, 64, 8).cuda()
    # model_path = r"E:\code\demosaic_and_fusion\only_fusion\train_save\fusion\1\real_pkl\100EPOCH.pkl"
    model_path=r'E:\code\demosaic_and_fusion\only_fusion\train_save\fusion_xytidu\5\real_pkl\5EPOCH.pkl'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    # net2.load_state_dict(checkpoint2['model_state_dict'])  # 加载模型可学习参数
    # model.load_state_dict(checkpoint2)
    #  E:\code\Unsupervised-Spectral-Demosaicing-main-gai\demosaic_save
    save_path = r"E:\code\demosaic_and_fusion\only_fusion\xy_tidu_result/"

    imglist1 = os.listdir(path1)


    for i in range(len(imglist1)):
        img = hdf5storage.loadmat(path1 + imglist1[i])
        raw0 = img['mosaic'].astype(np.float32)  # 原始的一个通道raw
        pan = img['pan'].astype(np.float32)  # 对应的全色图像
        # raw0 = raw0[:2008 // 1, :2008 // 1, :]
        # pan = pan[:2008, :2008, :]
        raw0 = raw0 / 255.0
        pan = pan / 255.0

        raw0 = torch.from_numpy(raw0).float().permute(2, 0, 1).unsqueeze(0)
        lrhsi = unps(raw0)
        HSI_LR = to_position_order(lrhsi).squeeze(0).numpy()
        MSI = pan.astype(np.float32).transpose(2, 0, 1)  # c,w,h

        HSI_LR=torch.Tensor(HSI_LR).unsqueeze(0).cuda()
        MSI=torch.Tensor(MSI).unsqueeze(0).cuda()
        time1=time.time()
        fusion_hsi= model.fusion(HSI_LR, MSI)
        time2=time.time()
        print(time2-time1)

        fusion_hsi=fusion_hsi.clamp(0,1)
        fusion_hsi = fusion_hsi.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
        HSI_LR = HSI_LR.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
        MSI = MSI.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()

        # MSI = MSI.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()
        # HSI_LR = HSI_LR.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()
        # pan_d2= pan_d2.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()
        # pan_d1= pan_d1.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()
        #
        # pan_p1= pan_p1.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()
        # lrhsi_p2= lrhsi_p2.squeeze(0).cpu().permute(1, 1, 0).detach().numpy()


        test_data_path = os.path.join(save_path + imglist1[i])
        hdf5storage.savemat(test_data_path, {'hr': fusion_hsi}, format='7.3')
        hdf5storage.savemat(test_data_path, {'lr': HSI_LR}, format='7.3')
        hdf5storage.savemat(test_data_path, {'pan': MSI}, format='7.3')

        # hdf5storage.savemat(test_data_path, {'pan': MSI}, format='7.3')
        # hdf5storage.savemat(test_data_path, {'lr': HSI_LR}, format='7.3')
        # hdf5storage.savemat(test_data_path, {'pan_d2': pan_d2}, format='7.3')
        # hdf5storage.savemat(test_data_path, {'pan_d1': pan_d1}, format='7.3')
        # hdf5storage.savemat(test_data_path, {'lrhsi_p': lrhsi_p2}, format='7.3')
        # hdf5storage.savemat(test_data_path, {'pan_p1': pan_p1}, format='7.3')

        print(i)