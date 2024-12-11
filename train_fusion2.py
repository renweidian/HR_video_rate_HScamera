import os

import hdf5storage
import torch
from scipy.io import loadmat
from thop import profile, clever_format

from fusion_net import D_F_net
from loss import SSIM
from calculate_metrics import Loss_SAM, Loss_PSNR, Loss_RMSE

from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


def create_F():
    F = np.array(
        [[2.0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 11, 16, 19, 21, 20, 18, 16, 14, 11, 7, 5, 3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16, 9, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i] / div
    return F

# class MyarcLoss(torch.nn.Module):
#     def __init__(self):
#         super(MyarcLoss, self).__init__()
#
#     def forward(self, output, target):
#         sum1 = output * target
#         sum2 = torch.sum(sum1, dim=0) + 1e-10
#         norm_abs1 = torch.sqrt(torch.sum(output * output, dim=0)) + 1e-10
#         norm_abs2 = torch.sqrt(torch.sum(target * target, dim=0)) + 1e-10
#         aa = sum2 / norm_abs1 / norm_abs2
#         aa[aa < -1] = -1
#         aa[aa > 1] = 1
#         spectralmap = torch.acos(aa)
#         return torch.mean(spectralmap)

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:,:, :, :-1] - x[:,:, :, 1:])) + torch.mean(torch.abs(x[:,:, :-1, :] - x[:,:, 1:, :]))

if __name__ == '__main__':
    # 路径参数
    root = os.getcwd() + "/train_save"
    model_name = 'ICVL_30db'
    mkdir(os.path.join(root, model_name))
    current_list = os.listdir(os.path.join(root, model_name))
    for i in current_list:
        if len(i) > 1:
            current_list.remove(i)

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list) + 1
    model_name = os.path.join(model_name, str(train_value))


    # data_name='cave'
    # path1 = 'D:\data\cave\cave_train/'
    # path2 = 'D:\data\cave\cave_test/'

    data_name='icvl'
    path1 = 'D:\data\ICVL/'
    path2 = 'D:\data\icvl_test/'

    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    ssim_func = SSIM(window_size=8).cuda()
    # loss_func = nn.MSELoss(reduction='mean')

    downsample_factor = 4
    training_size = 128
    stride = 128
    LR1 = 4e-4
    LR2 = 5e-3
    EPOCH = 1000
    weight_decay = 1e-8  # 我的模型是1e-8
    BATCH_SIZE = 16

    test_epoch = 10
    val_interval = 10  # 每隔val_interval epoch测试一次
    checkpoint_interval = 100

    # print("maxiteration：", maxiteration)

    a, b = 64, 64
    hsi = torch.randn(2, 31, a //4, b // 4).cuda()
    msi = torch.randn(2, 3, a, b).cuda()

    model2 = D_F_net(32, 3, 6, 64, 4).cuda()
    # model2=_3DT_Net(31,8,64).cuda()
    flops, params, DIC = profile(model2, inputs=(hsi, msi), ret_layer_info=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    # 创建方法名字的文件
    path=os.path.join(root,model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name=data_name+'_pkl'
    pkl_path=os.path.join(path,pkl_name)      # 模型保存路径
    os.makedirs(pkl_path)      # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'lr', 'train_loss','val_loss','val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name=data_name+'_record.csv'
    excel_path=os.path.join(path,excel_name)
    df.to_csv(excel_path, index=False)


    R=create_F()
    # PSF = fspecial('gaussian', 5, 0.5)
    PSF=np.load(r'E:\code\demosaic_and_fusion\simulation_fusion\psf\psf_icvl_30.npy')


    # test_data = CAVEHSIDATAprocess2(path2, R, 512, 512, downsample_factor, PSF)
    # test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    train_data = CAVEHSIDATAprocess(path1, R, training_size, stride, downsample_factor, PSF)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)





    cnn = D_F_net(32, 3, 6, 64, 4).cuda()


    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer1 = torch.optim.Adam(cnn.fusion_net.parameters(), lr=LR1, betas=(0.9, 0.999), weight_decay=weight_decay)      # 融合的


    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [30], 0.5)
    # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, EPOCH*12, 0)

    # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, EPOCH*len(train_data)//BATCH_SIZE, 0)
    # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.9)

    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [20, 40, 60,80], 0.1)

    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径

    # start_step=0
    if resume:
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        cnn.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # start_step=start_epoch -1
        # scheduler.last_epoch = start_epoch +1 # 设置学习率的last_epoch
        psnr_optimal = checkpoint['psnr_optimal']
        rmse_optimal = checkpoint['rmse_optimal']

    # step = start_step
    step = 0  # warm_lr_scheduler要用
    for epoch in range(start_epoch + 1, EPOCH + 1):
        cnn.train()
        all_losses1 = AverageMeter()
        all_losses2 = AverageMeter()

        pan_losses = AverageMeter()
        lr_losses = AverageMeter()
        down_losses = AverageMeter()
        tv_losses = AverageMeter()
        g_ssim_losses1=AverageMeter()
        g_ssim_losses2=AverageMeter()
        hr_losses=AverageMeter()

        loop = tqdm(train_loader, total=len(train_loader))
        for  train_lrhs,train_hrpan,train_hrhs in loop:
            train_hrpan, train_lrhs ,train_hrhs=  train_hrpan.cuda(), train_lrhs.cuda(),train_hrhs.cuda()
            step = step + 1
            lr1 = optimizer1.param_groups[0]['lr']
            # lr2 = optimizer2.param_groups[0]['lr']

            fusion_hsi = cnn.fusion(train_lrhs, train_hrpan)

            loss = loss_func(fusion_hsi, train_hrhs)



            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()


            all_losses1.update(loss.detach().cpu().numpy())


            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({
                'loss_all1': f'{all_losses1.avg.item():.6f}',
                'lr1': f'{lr1:.6f}'
            })
        scheduler1.step()
        # scheduler2.step()



        if ((epoch % 10 == 0) and (epoch >= 0)) or epoch == 1 or epoch == EPOCH:
            cnn.eval()

            test_hr_losses = AverageMeter()

            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()


            # test_bar = tqdm(test_loader)
            # with (torch.no_grad()):
            #     for train_lrhs, train_hrpan,hr in test_bar:
            #         train_hrpan, train_lrhs,hr = train_hrpan.cuda(), train_lrhs.cuda(),hr.cuda()

            imglist = os.listdir(path2)
            for i in range(0, len(imglist)):
                img = hdf5storage.loadmat(path2 + imglist[i])
                img1 = img["rad"]
                img1 = img1 / img1.max()
                HRHSI = torch.Tensor(np.transpose(img1, (2, 0, 1)))
                b, w, h = HRHSI.shape
                nw, nh = w // 16, h // 16
                HRHSI = HRHSI[:, :nw * 16, :nh * 16]
                MSI = torch.tensordot(torch.Tensor(R), HRHSI, dims=([1], [0]))
                MSI=MSI.cpu().numpy()
                HSI_LR = Gaussian_downsample(HRHSI, PSF, downsample_factor)

                SNRh = 30
                sigma = np.sqrt(
                    np.sum(HSI_LR ** 2) / (10 ** (SNRh / 10)) / (HSI_LR.shape[0] * HSI_LR.shape[1] * HSI_LR.shape[2]))
                HSI_LR = HSI_LR + sigma * np.random.randn(HSI_LR.shape[0], HSI_LR.shape[1], HSI_LR.shape[2])

                SNRh = 35
                sigma = np.sqrt(np.sum(MSI ** 2) / (10 ** (SNRh / 10)) / (MSI.shape[0] * MSI.shape[1] * MSI.shape[2]))
                MSI = MSI + sigma * np.random.randn(MSI.shape[0], MSI.shape[1], MSI.shape[2])

                train_hrpan = torch.unsqueeze(torch.Tensor(MSI), 0).cuda()
                train_lrhs = torch.unsqueeze(torch.Tensor(HSI_LR), 0).cuda() # 加维度 (b,c,h,w)
                hr=torch.unsqueeze(torch.Tensor(HRHSI), 0).cuda()
                with (torch.no_grad()):
                    fusion_hsi= cnn.fusion(train_lrhs,train_hrpan)
                    test_hr_loss = loss_func(fusion_hsi, hr)  # 只是计算，不更新参数
                    test_hr_losses.update(test_hr_loss.detach().cpu().numpy())
                    hrhsi = torch.clamp(fusion_hsi, 0, 1)
                    # sam.update(SAM(np.transpose(hr.squeeze().cpu().detach().numpy(), (1, 1, 0)),
                    #                np.transpose(hrhsi.squeeze().cpu().detach().numpy(), (1, 1, 0))))
                    rmse.update(RMSE(hr.squeeze().cpu().permute(1, 2, 0), hrhsi.squeeze().cpu().permute(1, 2, 0)))
                    psnr.update(PSNR(hr.squeeze().cpu().permute(1, 2, 0), hrhsi.squeeze().cpu().permute(1, 2, 0)))
            print("PSNR:", psnr.avg.cpu().detach().numpy(), "RMSE:", rmse.avg.cpu().detach().numpy())
            print('train: {:.6f}'.format(all_losses1.avg.item()),
                  'test: {:.6f}'.format(test_hr_losses.avg.item()))

            val_list = [epoch, lr1, all_losses1.avg.item(), test_hr_losses.avg.item(), rmse.avg.cpu().detach().numpy(),
                        psnr.avg.cpu().detach().numpy()]

            val_data = pd.DataFrame([val_list])
            val_data.to_csv(excel_path, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
            time.sleep(0.1)

            torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '.pkl')