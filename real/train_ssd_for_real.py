import os

import torch
from scipy.io import loadmat
from thop import profile, clever_format

from fusion_net import D_F_net
from loss import SSIM, mygradient

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




def total_variation_loss(x):
    return torch.mean(torch.abs(x[:,:, :, :-1] - x[:,:, :, 1:])) + torch.mean(torch.abs(x[:,:, :-1, :] - x[:,:, 1:, :]))

if __name__ == '__main__':
    # 路径参数
    root = os.getcwd() + "/train_save"
    model_name = 'ssd'
    mkdir(os.path.join(root, model_name))
    current_list = os.listdir(os.path.join(root, model_name))
    for i in current_list:
        if len(i) > 1:
            current_list.remove(i)

    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list) + 1
    model_name = os.path.join(model_name, str(train_value))

    data_name = 'real'
    train_path = r"F:\realdata\new_data\reg\mat\train/"

    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    ssim_func = SSIM().cuda()
    # loss_func = nn.MSELoss(reduction='mean')

    downsample_factor = 8
    training_size = 128
    stride = 64
    # LR1 = 1e-4
    LR2 = 1e-2
    EPOCH = 100
    weight_decay = 1e-8  # 我的模型是1e-8
    BATCH_SIZE = 16

    test_epoch = 10
    val_interval = 10  # 每隔val_interval epoch测试一次
    checkpoint_interval = 100

    # print("maxiteration：", maxiteration)

    a, b = 64, 64
    hsi = torch.randn(2, 16, a //8, b // 8).cuda()
    msi = torch.randn(2, 1, a, b).cuda()

    model2 = D_F_net(16, 1, 6, 64, 8).cuda()
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

    train_data = RealdatasetF(train_path, training_size, stride, downsample_factor)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)




    cnn = D_F_net(16, 1, 6, 64, 8).cuda()


    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    optimizer2 = torch.optim.Adam(cnn.ssf_net.parameters(), lr=LR2, betas=(0.9, 0.999), weight_decay=weight_decay)         # 退化的



    # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, [20, 40, 60,80], 0.1)

    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径

    # start_step=0


    # step = start_step
    step = 0  # warm_lr_scheduler要用
    for epoch in range(start_epoch + 1, EPOCH + 1):
        cnn.train()
        all_losses1 = AverageMeter()
        down_losses= AverageMeter()

        loop = tqdm(train_loader, total=len(train_loader))
        for  train_lrhs,train_hrpan in loop:
            train_hrpan, train_lrhs =  train_hrpan.cuda(), train_lrhs.cuda()
            step = step + 1
            lr2 = optimizer2.param_groups[0]['lr']

            pan_d1, pan_d2, half_k, sigma = cnn.spa_and_spe_d(train_lrhs, train_hrpan)


            # 再次采样的损失
            down_loss=1-ssim_func(pan_d1, pan_d2)

            # 模糊核大小和方差损失
            kernel_loss = 5e-4 * half_k
            loss_all=kernel_loss + down_loss

            optimizer2.zero_grad()
            loss_all.backward()
            optimizer2.step()

            all_losses1.update(loss_all.detach().cpu().numpy())
            down_losses.update(down_loss.detach().cpu().numpy())


            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({
                'loss_all2': f'{all_losses1.avg.item():.6f}',
                'down_loss': f'{down_losses.avg.item():.6f}',
                'lr1': f'{lr2:.6f}' , # 确保 lr 是一个单个数值
            })

        if epoch == 1 or epoch % 5 == 0:
            torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '.pkl')

