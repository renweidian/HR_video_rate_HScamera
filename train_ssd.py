import os

import torch
from thop import profile, clever_format

from fusion_net import D_F_net
from loss import SSIM

from train_dataloader import *
from torch import nn
from tqdm import tqdm
import pandas as pd
import torch.utils.data as data
from utils import  fspecial

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

def total_variation_loss(x):
    return torch.mean(torch.abs(x[:,:, :, :-1] - x[:,:, :, 1:])) + torch.mean(torch.abs(x[:,:, :-1, :] - x[:,:, 1:, :]))

if __name__ == '__main__':
    # 路径参数
    root = os.getcwd() + "/train_save"
    model_name = 'ssd_icvl'
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

    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    ssim_func = SSIM().cuda()
    # loss_func = nn.MSELoss(reduction='mean')

    downsample_factor = 4
    training_size = 128
    stride = 256
    LR1 = 1e-4
    LR2 = 1e-2
    EPOCH = 20
    weight_decay = 1e-8  # 我的模型是1e-8
    BATCH_SIZE = 16

    test_epoch = 10
    val_interval = 10  # 每隔val_interval epoch测试一次
    checkpoint_interval = 100

    # print("maxiteration：", maxiteration)

    a, b = 64, 64
    hsi = torch.randn(2, 31, a //4, b //4).cuda()
    msi = torch.randn(2, 3, a, b).cuda()

    model2 = D_F_net(32, 3, 6, 64, 4).cuda()
    # model2=_3DT_Net(31,8,64).cuda()
    flops, params, DIC = profile(model2, inputs=(hsi, msi), ret_layer_info=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

    # 创建方法名字的文件
    path = os.path.join(root, model_name)
    mkdir(path)  # 创建文件夹
    # 创建训练记录文件
    pkl_name = data_name + '_pkl'
    pkl_path = os.path.join(path, pkl_name)  # 模型保存路径
    os.makedirs(pkl_path)  # 创建文件夹
    # 创建excel
    df = pd.DataFrame(columns=['epoch', 'fusion_lr', 'd_lr','train_loss', 'val_loss', 'val_rmse', 'val_psnr', 'val_sam'])  # 列名
    excel_name = data_name + '_record.csv'
    excel_path = os.path.join(path, excel_name)
    df.to_csv(excel_path, index=False)

    R=create_F()
    PSF = fspecial('gaussian', 5, 0.5)
    np.save('psf/psf_icvl.npy', PSF)

    train_data = CAVEHSIDATAprocess2(path1, R, training_size, stride, downsample_factor, PSF)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


    # test_data = CAVEHSIDATAprocess2(path2, R, 512, 512, downsample_factor, PSF)
    # test_loader = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)


    cnn = D_F_net(32, 3, 6, 64, 4).cuda()


    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer2 = torch.optim.Adam(cnn.ssf_net.parameters(), lr=LR2, betas=(0.9, 0.999), weight_decay=weight_decay)         # 退化的


    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [20], 0.1)

    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径



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
            lr2 = optimizer2.param_groups[0]['lr']

            pan_d1, pan_d2, half_k, sigma = cnn.spa_and_spe_d(train_lrhs, train_hrpan)


            # 再次采样的损失
            down_loss=1-ssim_func(pan_d1, pan_d2)

            # 模糊核大小和方差损失
            kernel_loss = 5e-4 * (half_k+sigma)

            loss_all2=kernel_loss + down_loss

            optimizer2.zero_grad()
            loss_all2.backward()
            optimizer2.step()

            all_losses2.update(loss_all2.detach().cpu().numpy())


            down_losses.update(down_loss.detach().cpu().numpy())


            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({
                'loss_all2': f'{all_losses2.avg.item():.6f}',
                'down_loss': f'{down_losses.avg.item():.6f}',
                'lr1': f'{lr2:.6f}' , # 确保 lr 是一个单个数值
            })
        scheduler2.step()

        if epoch == 1 or epoch % 5 == 0:
            torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '.pkl')



