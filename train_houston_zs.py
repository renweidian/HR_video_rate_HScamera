import hdf5storage

from calculate_metrics import Loss_SAM, Loss_PSNR, Loss_RMSE
from fusion_net_houston import D_F_net

from train_dataloader import *
from torch import nn
from tqdm import tqdm
import time
import pandas as pd
import torch.utils.data as data
from utils import create_F, fspecial
import math
import tifffile as tf           #tifffile是tiff文件的读取库
import hdf5storage as h5
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def mkdir(path):

    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("训练文件夹为：{}".format(path))
    else:
        print('已存在{}'.format(path))


class RealDATAProcess3(Dataset):
    def __init__(self, LR,msi,HR, training_size, stride, downsample_factor):
        """
        :param path:
        :param R: 光谱响应矩阵
        :param training_size:
        :param stride:
        :param downsample_factor:
        :param PSF: 高斯模糊核
        :param num: cave=32个场景
        """
        train_hrhs = []
        train_lrhs = []
        train_hrms = []



        HSI_LR = LR
        MSI = msi
        HRHSI=HR
        for j in range(0, HRHSI.shape[1] - training_size + 1, stride):
            for k in range(0, HRHSI.shape[2] - training_size + 1, stride):
                # if (j+training_size)>800 and k<400:
                #     pass
                # else:
                temp_hrhs = HRHSI[:, j:j + training_size, k:k + training_size]
                temp_hrms = MSI[:, j:j + training_size, k:k + training_size]

                temp_lrhs = HSI_LR[:, int(j / downsample_factor):int((j + training_size) / downsample_factor),
                            int(k / downsample_factor):int((k + training_size) / downsample_factor)]
                # print(temp_hrhs.shape,temp_lrhs.shape,temp_hrms.shape)
                temp_hrhs=temp_hrhs.astype(np.float32)
                temp_lrhs=temp_lrhs.astype(np.float32)
                temp_hrms = temp_hrms.astype(np.float32)
                train_hrhs.append(temp_hrhs)
                train_lrhs.append(temp_lrhs)
                train_hrms.append(temp_hrms)

        train_hrhs = torch.Tensor(np.array(train_hrhs))
        train_lrhs = torch.Tensor(np.array(train_lrhs))
        train_hrms = torch.Tensor(np.array(train_hrms))

        # print(train_hrhs.shape, train_hrms.shape)
        self.train_hrhs_all = train_hrhs
        self.train_lrhs_all = train_lrhs
        self.train_hrms_all = train_hrms

    def __getitem__(self, index):
        train_hrhs = self.train_hrhs_all[index, :, :, :]
        train_lrhs = self.train_lrhs_all[index, :, :, :]
        train_hrms = self.train_hrms_all[index, :, :, :]
        # print(train_hrhs.shape, train_hrms.shape,train_lrhs.shape)
        return  train_lrhs,train_hrms,train_hrhs

    def __len__(self):
        return self.train_hrhs_all.shape[0]


if __name__ == '__main__':
    # 路径参数
    root=os.getcwd()+"/train_save"
    model_name='houston_30'
    mkdir(os.path.join(root,model_name))
    ori_list=os.listdir(os.path.join(root,model_name))
    current_list=[]
    for i in ori_list:
        if len(i)<=2:
            current_list.append(i)

    del ori_list



    current_list.append('0')
    int_list = [int(x) for x in current_list]
    train_value = max(int_list)+1
    model_name=os.path.join(model_name,str(train_value))

    # data_name='icvl'
    # path1 = 'D:\data\Harvard\harvard_train/'
    # path2 = 'D:\data\Harvard\harvard_test/'


    training_size=32
    stride=4
    downsample_factor=4




    """
    # data_name='gf5'
    # HSI_ori = h5.loadmat(r"D:\data\PaviaU\PaviaU.mat")['paviaU']
    # HSI_ori=HSI_ori/np.max(HSI_ori)
    # HSI_ori= np.transpose(HSI_ori, (2, 0, 1))
    # R = h5.loadmat(r"E:\code\fusion_code\data\R.mat")['R']
    # 
    # for band in range(R.shape[0]):
    #     div = np.sum(R[band][:])
    #     for i in range(R.shape[1]):
    #         R[band][i] = R[band][i] / div
    # 
    # 
    # # R=np.transpose(R, (1,0))
    # PSF = fspecial('gaussian', 8, 3)
    # 
    # C,W,H=HSI_ori.shape
    # dw=W//downsample_factor
    # dh=H//downsample_factor
    # HSI_ori=HSI_ori[:,:dw*downsample_factor,:dh*downsample_factor]
    # 
    # 
    # HSI_D1 = Gaussian_downsample(HSI_ori, PSF, downsample_factor)
    # MSI_D1= np.tensordot(R, HSI_ori, axes=([1], [0]))
    # 
    # 
    # C,W,H=HSI_D1.shape
    # dw=W//downsample_factor
    # dh=H//downsample_factor
    # HSI_D11=HSI_D1[:,:dw*downsample_factor,:dh*downsample_factor]
    # 
    # HSI_D2 = Gaussian_downsample(HSI_D11, PSF, downsample_factor)
    # MSI_D2= np.tensordot(R, HSI_D1, axes=([1], [0]))
  """






    data_name = 'houston'
    HSI_ori = h5.loadmat(r"D:\data\HOUSTON\HSI.mat")['HSI']
    HSI_ori = np.double(HSI_ori)

    HSI_ori = HSI_ori / np.max(HSI_ori)
    # train_HSI_ori=HSI_ori[:,:1024,:]
    # test_HSI_ori=HSI_ori

    # R = h5.loadmat(r"E:\code\fusion_code\data\R.mat")['R']
    # R=R[:,:102]
    #
    # for band in range(R.shape[0]):
    #     div = np.sum(R[band][:])
    #     for i in range(R.shape[1]):
    #         R[band][i] = R[band][i] / div

    HSI_ori = np.transpose(HSI_ori, (2, 0, 1))

    # R=np.transpose(R, (1,0))
    # PSF = fspecial('gaussian', 8, 3)
    PSF = np.load('E:\code\demosaic_and_fusion\simulation_fusion\psf\psf_houston_30.npy')
    C, W, H = HSI_ori.shape
    dw = W // downsample_factor**2
    dh = H // downsample_factor**2
    HSI_ori = HSI_ori[:, :dw * downsample_factor * downsample_factor, :dh * downsample_factor* downsample_factor]

    MSI1 = np.mean(HSI_ori[:36, :, :], 0, keepdims=True)
    MSI2 = np.mean(HSI_ori[36:36 * 2, :, :], 0, keepdims=True)
    MSI3 = np.mean(HSI_ori[36 * 2:36 * 3, :, :], 0, keepdims=True)
    MSI4 = np.mean(HSI_ori[36 * 3:36 * 4, :, :], 0, keepdims=True)
    MSI_D1 = np.concatenate([MSI1, MSI2, MSI3,MSI4], 0)
    HSI_D1 = Gaussian_downsample(HSI_ori, PSF, downsample_factor)

    SNRh = 30
    sigma = np.sqrt(np.sum(MSI_D1 ** 2) / (10 ** (SNRh / 10)) / (MSI_D1.shape[0] * MSI_D1.shape[1] * MSI_D1.shape[2]));
    MSI_D1 = MSI_D1 + sigma * np.random.randn(MSI_D1.shape[0], MSI_D1.shape[1], MSI_D1.shape[2])

    SNRh = 35
    sigma = np.sqrt(np.sum(HSI_D1 ** 2) / (10 ** (SNRh / 10)) / (HSI_D1.shape[0] * HSI_D1.shape[1] * HSI_D1.shape[2]));
    HSI_D1 = HSI_D1 + sigma * np.random.randn(HSI_D1.shape[0], HSI_D1.shape[1], HSI_D1.shape[2])

    HR_test=HSI_ori[:,:,:400]
    MSI_test=MSI_D1[:,:,:400]
    HSI_test=HSI_D1[:,:,:400//downsample_factor]

    HR_train = HSI_ori[:, :, 400:]
    MSI_train = MSI_D1[:, :, 400:]
    HSI_train= HSI_D1[:, :, 400//downsample_factor:]

    C, W, H = HSI_train.shape
    PSF2=np.load('E:\code\demosaic_and_fusion\simulation_fusion\psf\psf_houston_30_2.npy')
    dw = W // downsample_factor
    dh = H // downsample_factor

    HSI_D11 = HSI_train[:, :dw * downsample_factor, :dh * downsample_factor]
    MSI_D11= MSI_train[:, :dw * downsample_factor* downsample_factor, :dh * downsample_factor* downsample_factor]


    MSI_D2 = Gaussian_downsample(MSI_D11, PSF2, downsample_factor)
    HSI_D2 = Gaussian_downsample(HSI_D11, PSF2, downsample_factor)

    SNRh = 30
    sigma = np.sqrt(np.sum(MSI_D2 ** 2) / (10 ** (SNRh / 10)) / (MSI_D2.shape[0] * MSI_D2.shape[1] * MSI_D2.shape[2]));
    MSI_D2 = MSI_D2 + sigma * np.random.randn(MSI_D2.shape[0], MSI_D2.shape[1], MSI_D2.shape[2])

    SNRh = 35
    sigma = np.sqrt(np.sum(HSI_D2 ** 2) / (10 ** (SNRh / 10)) / (HSI_D2.shape[0] * HSI_D2.shape[1] * HSI_D2.shape[2]));
    HSI_D2 = HSI_D2 + sigma * np.random.randn(HSI_D2.shape[0], HSI_D2.shape[1], HSI_D2.shape[2])

    print("训练数据处理完成")


    # 训练参数
    loss_func = nn.L1Loss(reduction='mean').cuda()
    # loss_func = nn.MSELoss(reduction='mean')



    stride1 = 16
    LR = 4e-4
    EPOCH =300
    weight_decay=0    # 我的模型是1e-8
    BATCH_SIZE = 8
    psnr_optimal = 40
    rmse_optimal = 3

    test_epoch=5
    val_interval = 5           # 每隔val_interval epoch测试一次
    checkpoint_interval = 100
    # maxiteration = math.ceil(((512 - training_size) // stride + 1) ** 2 * num / BATCH_SIZE) * EPOCH
    # maxiteration = math.ceil(
    #      ((train_w - training_size) // stride + 1) * ((train_h - training_size) // stride + 1) / BATCH_SIZE) * EPOCH
    # print("maxiteration：", maxiteration)

    # warm_lr_scheduler

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

    df2 = pd.DataFrame(columns=['epoch', 'lr', 'train_loss'])  # 列名
    excel_name2 = data_name + '_train_record.csv'
    excel_path2 = os.path.join(path, excel_name2)
    df2.to_csv(excel_path2, index=False)

    train_data=RealDATAProcess3(HSI_D2,MSI_D2,HSI_D11,training_size, stride, downsample_factor)
    train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    maxiteration = math.ceil(len(train_data) / BATCH_SIZE) * EPOCH
    print("maxiteration：", maxiteration)

    decay_power = 1.5
    init_lr2 = 1e-4
    init_lr1 = 1e-4 / 10
    min_lr=0
    warm_iter = math.floor(maxiteration / 40)

    cnn = D_F_net(144,4,6,64,4).cuda()

    # cnn = Cross_Guide_Fusion(144,training_size,training_size,1).cuda()
    # cnn =FusionNet().cuda()
    # cnn=VSR_CAS(channel0=31, factor=8, P=torch.Tensor(R), patch_size=training_size).cuda()
    # cnn=RGBNet(150,6,200).cuda()
    # cnn = Cross_Guide_Fusion(150, training_size, training_size,1).cuda()
    # cnn=MainNet().cuda()
    # cnn=SSRNET('SSRNET',2,4,150).cuda()
    # 模型初始化
    for m in cnn.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    # optimizer = torch.optim.Adam([{'params': cnn.parameters(), 'initial_lr': 1e-4}], lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR,betas=(0.9, 0.999),weight_decay=weight_decay)
    # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor = 0.96, patience = 10,
    #                                                      verbose = True, threshold = 0.00001, threshold_mode ='abs', cooldown = 5, min_lr = 0, eps = 1e-08)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxiteration, eta_min=1e-6, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 300, 700], 0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500,
    #                                                gamma=0.5)   # Fuformer

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100], 0.5)

    start_epoch = 0
    # resume = True
    resume = False
    path_checkpoint = "checkpoints/500_epoch.pkl"  # 断点路径

    # start_step=0
    if resume:
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        cnn.load_state_dict(checkpoint['model_state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # start_step=start_epoch -1
        # scheduler.last_epoch = start_epoch +1 # 设置学习率的last_epoch
        psnr_optimal = checkpoint['psnr_optimal']
        rmse_optimal = checkpoint['rmse_optimal']



    # step = start_step
    step=0   # warm_lr_scheduler要用
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
            lr = optimizer.param_groups[0]['lr']
            # lr2 = optimizer2.param_groups[0]['lr']

            fusion_hsi = cnn.fusion(train_lrhs, train_hrpan)

            loss = loss_func(fusion_hsi, train_hrhs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            all_losses1.update(loss.detach().cpu().numpy())


            loop.set_description(f'Epoch [{epoch}/{EPOCH}]')
            loop.set_postfix({
                'loss_all1': f'{all_losses1.avg.item():.6f}',
                'lr1': f'{lr:.6f}' , # 确保 lr 是一个单个数值
                # 'lr2': f'{lr2:.6f}'  # 确保 lr 是一个单个数值
            })
        scheduler.step()
        # scheduler.step()
        # train_list = [epoch, lr, np.mean(loss_all)]
        # train_record = pd.DataFrame([train_list])
        # train_record.to_csv(excel_path2, mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了




        if ((epoch % val_interval == 0) and (epoch>=test_epoch) ) or epoch==1:
            cnn.eval()
            val_loss=AverageMeter()
            SAM = Loss_SAM()
            RMSE = Loss_RMSE()
            PSNR = Loss_PSNR()
            sam = AverageMeter()
            rmse = AverageMeter()
            psnr = AverageMeter()
            cnn.eval()
            with torch.no_grad():
                # img1 = img1 / img1.max()
                HRHSI = torch.Tensor(HR_test).unsqueeze(0).cuda()
                HSI_LR = torch.Tensor(HSI_test).unsqueeze(0).cuda()
                MSI = torch.Tensor(MSI_test).unsqueeze(0).cuda()
                with torch.no_grad():
                    fusion_hsi, pan_d1, pan_d2, pan_p1, lrhsi_p2, half_k, sigma = cnn(HSI_LR, MSI)
                    test_hr_loss = loss_func(fusion_hsi, HRHSI)  # 只是计算，不更新参数
                val_loss.update(test_hr_loss.detach().cpu().numpy())
                fusion_hsi = fusion_hsi.clamp(0, 1)
                # print(Fuse.shape)
                # prediction=torch.round(prediction*255)/255.0
                sam.update(SAM(np.transpose(HRHSI.squeeze().cpu().detach().numpy(),(1, 2, 0)),np.transpose(fusion_hsi.squeeze().cpu().detach().numpy(),(1, 2, 0))))
                rmse.update(RMSE(HRHSI.squeeze().cpu().permute(1,2,0),fusion_hsi.squeeze().cpu().permute(1,2,0)))
                psnr.update(PSNR(HRHSI.squeeze().cpu().permute(1,2,0),fusion_hsi.squeeze().cpu().permute(1,2,0)))
                fusion_hsi = fusion_hsi.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
                HRHSI = HRHSI.squeeze(0).cpu().permute(1, 2, 0).detach().numpy()
                hdf5storage.savemat(r'E:\code\demosaic_and_fusion\simulation_fusion\result\houston_30\houston.mat', {'fak': fusion_hsi}, format='7.3')

                hdf5storage.savemat(r'E:\code\demosaic_and_fusion\simulation_fusion\result\houston_30\houston.mat', {'rea': HRHSI}, format='7.3')
            if  epoch == 1 or epoch==EPOCH:
                torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_PSNR_best.pkl')

            if torch.abs(psnr_optimal-psnr.avg)<0.15 or psnr.avg>psnr_optimal:
                torch.save(cnn.state_dict(), pkl_path + '/' + str(epoch) + 'EPOCH' + '_PSNR_best.pkl')
            if psnr.avg > psnr_optimal:
                psnr_optimal = psnr.avg

            if torch.abs(rmse.avg-rmse_optimal)<0.15 or rmse.avg<rmse_optimal:
                torch.save(cnn.state_dict(),pkl_path +'/'+ str(epoch) + 'EPOCH' + '_RMSE_best.pkl')
            if rmse.avg < rmse_optimal:
                rmse_optimal = rmse.avg

            print("PSNR:", psnr.avg.cpu().detach().numpy(), "RMSE:", rmse.avg.cpu().detach().numpy())
            print('train: {:.6f}'.format(all_losses1.avg.item()),
                  'test: {:.6f}'.format(val_loss.avg.item()))

            val_list = [epoch, lr, all_losses1.avg.item(), val_loss.avg.item(), rmse.avg.cpu().detach().numpy(),
                        psnr.avg.cpu().detach().numpy()]

            val_data = pd.DataFrame([val_list])
            val_data.to_csv(excel_path,mode='a', header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
            time.sleep(0.1)
