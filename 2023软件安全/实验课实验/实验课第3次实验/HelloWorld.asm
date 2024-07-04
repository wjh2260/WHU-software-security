#/usr/bin/env python
# 训练模型使用的脚本，net1和net2是级联的两个可逆神经网络模型，net3是生成IM模块的模型
import sys
import os
import torch
import torch.nn
import torch.optim
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import tqdm                                     # 显示训练进度
from model import *                             # 从model.py中导入定义的网络模型（应该是net1和net2）
from imp_subnet import *                        # 从imp_subnet.py中导入定义的网络模型（应该是net3）
import torchvision.transforms as T              # TorchVision库中的图像转换模块
import config as c                              # 导入配置文件config.py
from tensorboardX import SummaryWriter          # 用于可视化训练过程
from datasets import trainloader, testloader    # 导入加载训练数据和测试数据的模块
import viz
import modules.module_util as mutil
import modules.Unet_common as common
import warnings
from vgg_loss import VGGLoss                    # 额这个实际上没用上
warnings.filterwarnings("ignore")

# 选择GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 计算图像的峰值信噪比(PSNR)，是衡量图像质量的指标，数值越高表示图像质量越好
# --> 在验证的时候会对比原始秘密图像与提取出的秘密图像、原始载体图像和嵌入后的载体图像
def computePSNR(origin, pred):
    # 将输入转化为浮点类型
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    # 计算均方根误差
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    # 处理极小MSE和极大MSE情况
    if mse < 1.0e-10:
        return 100

    if mse > 1.0e15:
        return -100

    return 10 * math.log10(255.0 ** 2 / mse)


# 生成高斯噪声
def gauss_noise(shape):
    # 使用PyTorch创建一个与输入形状相同的零张量，然后对每个样本应用均值为0、方差为1的高斯分布生成噪声
    noise = torch.zeros(shape).to(device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(device)

    return noise


########### 下面是一组损失函数，用于指导模型训练 #################
##### 其中，L1损失是指平均绝对误差，L2损失是指均方误差(MSE) #############
# 引导损失(确保生成的隐写图像与原始图像相似)
def guide_loss(output, bicubic_image):
    # 使用均方误差损失函数（MSELoss）计算模型输出与双三次插值图像之间的差异
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)

# 重建损失(确保对比提取出的秘密图像与原始秘密图像相似)
def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)

# Imp损失(output 是模型的输出，resi 是残差(模型输出减去输入))
def imp_loss(output, resi):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, resi)
    return loss.to(device)

# 低频损失(确保生成的图像在低频部分保持相似)
# -->代码中没有用到这个损失函数，而是直接提取了图像低频部分然后计算guide_loss视为低频损失
def low_frequency_loss(ll_input, gt_input):
    # 使用 L1 损失函数计算低频分量之间的差异
    loss_fn = torch.nn.L1Loss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)

# 分布损失(noise是模型输出的噪声,计算模型输出的噪声与零噪声之间的差异)
# -->代码中也没有用到这个损失函数
def distr_loss(noise):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(noise, torch.zeros(noise.shape).cuda())
    return loss.to(device)
################# end ###############################


# 获取net网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# 加载模型权重，将网络参数加载到模型中并尝试加载优化器的状态字典
def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


# 初始化net3(IM)网络参数
def init_net3(mod):
    for key, param in mod.named_parameters():
        if param.requires_grad:
            param.data = 0.1 * torch.randn(param.data.shape).to(device)




if __name__ == '__main__':

    #####################
    # Model initialize: #
    #####################
    # 初始化模型
    net1 = Model_1()
    net2 = Model_2()
    net3 = ImpMapBlock()
    # 将模型移动到 GPU
    net1.cuda()
    net2.cuda()
    net3.cuda()
    # 模型参数初始化
    init_model(net1)
    init_model(net2)
    init_net3(net3)
    # 使用 DataParallel 进行并行化，同时训练三个模型
    net1 = torch.nn.DataParallel(net1, device_ids=c.device_ids)
    net2 = torch.nn.DataParallel(net2, device_ids=c.device_ids)
    net3 = torch.nn.DataParallel(net3, device_ids=c.device_ids)
    # 获取并打印模型参数数量
    para1 = get_parameter_number(net1)
    para2 = get_parameter_number(net2)
    para3 = get_parameter_number(net3)
    print(para1)
    print(para2)
    print(para3)
    # 获取可训练参数列表和初始化优化器
    params_trainable1 = (list(filter(lambda p: p.requires_grad, net1.parameters())))
    params_trainable2 = (list(filter(lambda p: p.requires_grad, net2.parameters())))
    params_trainable3 = (list(filter(lambda p: p.requires_grad, net3.parameters())))
    optim1 = torch.optim.Adam(params_trainable1, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    optim2 = torch.optim.Adam(params_trainable2, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    optim3 = torch.optim.Adam(params_trainable3, lr=c.lr3, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    # 初始化学习率调度器
    weight_scheduler1 = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
    weight_scheduler2 = torch.optim.lr_scheduler.StepLR(optim2, c.weight_step, gamma=c.gamma)
    weight_scheduler3 = torch.optim.lr_scheduler.StepLR(optim3, c.weight_step, gamma=c.gamma)
    # 初始化离散小波变换和反离散小波变换
    dwt = common.DWT()
    iwt = common.IWT()

    # 判断是否要加载上一轮训练的模型
    # -->训练过程发生异常，做好调整后可以加载上一轮的模型继续训练（readme中也提到可能会出现梯度爆炸的情况，需要手动停止并调整学习率）
    if c.tain_next:
        load(c.MODEL_PATH + c.suffix_load + '_1.pt', net1, optim1)
        load(c.MODEL_PATH + c.suffix_load + '_2.pt', net2, optim2)
        load(c.MODEL_PATH + c.suffix_load + '_3.pt', net3, optim3)

    # 判断是否要加载预训练模型
    if c.pretrain:
        load(c.PRETRAIN_PATH + c.suffix_pretrain + '_1.pt', net1, optim1)
        load(c.PRETRAIN_PATH + c.suffix_pretrain + '_2.pt', net2, optim2)
        if c.PRETRAIN_PATH_3 is not None:
            load(c.PRETRAIN_PATH_3 + c.suffix_pretrain_3 + '_3.pt', net3, optim3)

    try:
        # 尝试创建一个用于 TensorBoardX 可视化训练过程的 SummaryWriter
        writer = SummaryWriter(log_dir='scalar', comment='hinet', filename_suffix="steg")

        for i_epoch in range(c.epochs):
            # 训练参数的初始化
            i_epoch = i_epoch + c.trained_epoch + 1
            loss_history = []
            loss_history_g1 = []
            loss_history_g2 = []
            loss_history_r1 = []
            loss_history_r2 = []
            loss_history_imp = []
            loss_history_roubst = []
            #################
            #     train:    #
            #################
            vgg_loss = VGGLoss(3, 1, False)
            vgg_loss.to(device)
            # 内层循环迭代每个 mini-batch，并准备训练数据
            for i_batch, data in enumerate(trainloader):
                # 数据准备(将数据移动到GPU并将数据分为 cover 和两个 secret, 并进行dwt变换)
                data = data.to(device)
                """
                在下面的代码中，对`data`进行了一系列的切片操作，将其分成了三个部分，分别赋值给了`cover`、`secret_1`和`secret_2`。
                - `secret_1`：表示从`data`中间部分提取的数据。具体来说，`secret_1`的取值范围是`data`的长度的1/3到2/3之间。这意味着`secret_1`包含了`data`中间1/3长度的数据。
                - `secret_2`：取值范围是`data`的长度的2/3到整体长度之间。这意味着`secret_2`包含了`data`后1/3长度的数据。
                """
                cover = data[:data.shape[0] // 3]  # channels = 3
                secret_1 = data[data.shape[0] // 3: 2 * (data.shape[0] // 3)]
                secret_2 = data[2 * (data.shape[0] // 3): 3 * (data.shape[0] // 3)]
                cover_dwt = dwt(cover)  # channels = 12
                cover_dwt_low = cover_dwt.narrow(1, 0, c.channels_in)  # channels = 3
                secret_dwt_1 = dwt(secret_1)
                secret_dwt_2 = dwt(secret_2)

                # 将cover_dwt和secret_dwt_1沿着通道维度拼接，构建输入数据
                input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24


                #################
                #  第一个前向传播-->net1用于隐藏secret1
                #################
                output_dwt_1 = net1(input_dwt_1)  # channels = 24
                output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                # 获取output_steg_dwt_low_1的目的是为了计算低频损失low_frequency_loss
                output_steg_dwt_low_1 = output_steg_dwt_1.narrow(1, 0, c.channels_in)  # channels = 3
                output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                # 逆dwt变换得到嵌入第一个秘密图像后的载体图像
                output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3


                #################
                #    第二个前向传播-->net3用于生成IM模块，net2用于隐藏secret2
                #################
                if c.use_imp_map:
                    # 如果要使用IM模块，生成imp_map
                    imp_map = net3(cover, secret_1, output_steg_1)  # channels = 3
                else:
                    # 不使用则代入形状相同的零张量
                    imp_map = torch.zeros(cover.shape).cuda()

                # 计算imploss
                impmap_loss = imp_loss(imp_map, cover - output_steg_1)

                # 对imp_map进行dwt
                imp_map_dwt = dwt(imp_map)  # channels = 12
                # 将output_steg_dwt_1和imp_map_dwt沿着通道维度拼接，构建输入数据
                input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)  # 24, without secret2

                # 与第二个秘密图的小波信号 (secret_dwt_2) 拼接，构建完整的输入
                input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 36

                # 与net1同理，net2嵌入第二个图像
                output_dwt_2 = net2(input_dwt_2)  # channels = 36
                output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                output_steg_dwt_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in)  # channels = 3
                output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24

                # get steg2
                output_steg_2 = iwt(output_steg_dwt_2)  # channels = 3
                #print(output_steg_2)

                #----------------鲁棒性模块(见config)-------------------#
                if (c.choice_attack == 1):
                    # 对steg_2加入随机噪声
                    noise = torch.randn_like(output_steg_2) * c.noise_level
                    #print(noise)
                    # 将噪声添加到输出图像
                    output_steg_2_with_noise = output_steg_2 + noise
                    # 使用clamp将像素值限制在合理范围内，避免超过张量的上下限
                    floor_level = torch.min(output_steg_2)
                    upper_level = torch.max(output_steg_2)
                    output_steg_2_with_noise = output_steg_2_with_noise.clamp(floor_level, upper_level)
                    # dwt变化得到反向传播的输入
                    output_steg_dwt_2_with_noise = dwt(output_steg_2_with_noise)
                    output_steg_dwt_low_with_noise = output_steg_dwt_2_with_noise.narrow(1, 0, c.channels_in)  # channels = 3

                elif (c.choice_attack == 2):
                    # 这里用量化操作将输出的浮点数值转换为较低的位数，来模拟PNG图像的8位色深
                    output_steg_2_toPNG = torch.floor(output_steg_2 * 255) / 255.0
                    #print(output_steg_2_toPNG)
                    # dwt变化得到反向传播的输入
                    output_steg_dwt_2_toPNG = dwt(output_steg_2_toPNG)
                    output_steg_dwt_low_toPNG = output_steg_dwt_2_toPNG.narrow(1, 0, c.channels_in)

                else:
                    pass
                #--------------------end------------------------#


                #################
                #  第二个反向传播-->利用net2的逆过程恢复第二个嵌入的秘密图像 
                #################

                # 生成嵌入图像时产生的噪声分布相同的高斯噪声，用于恢复嵌入的图像
                output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
                output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24

                #output_steg_dwt_2=dwt(output_steg_2)
                
                if (c.choice_attack == 1):
                    output_steg_dwt_2 = output_steg_dwt_2_with_noise
                elif (c.choice_attack == 2):
                    output_steg_dwt_2 = output_steg_dwt_2_toPNG
                else:
                    pass
                

                # 将 output_steg_dwt_2 和高斯噪声拼接，得到反向传播的输入
                output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36

                # 使用net2可逆神经网络恢复嵌入的第二个图像
                rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36

                # 得到载体图像的dwt和secret2的dwt
                rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                # rev_secret_dwt_2 = rev_dwt_2.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                rev_secret_dwt_2 = rev_dwt_2.narrow(1, rev_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)

                # 逆dwt恢复steg1和secret2
                rev_steg_1 = iwt(rev_steg_dwt_1).to(device)  # channels = 3
                rev_secret_2 = iwt(rev_secret_dwt_2).to(device)  # channels = 3

                #################
                #   第一个反向传播-->同理利用net1的逆过程恢复第一个嵌入的秘密图像
                #################
                output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

                rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 36

                rev_secret_dwt = rev_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                rev_secret_1 = iwt(rev_secret_dwt).to(device)

                #################
                #     loss:     #
                #################
                # 计算guide_loss\low_frequency_loss\reconstruction_loss
                g_loss_1 = guide_loss(output_steg_1.cuda(), cover.cuda())
                g_loss_2 = guide_loss(output_steg_2.cuda(), cover.cuda())
                
                vgg_on_cov = vgg_loss(cover)
                vgg_on_steg1 = vgg_loss(output_steg_1)
                vgg_on_steg2 = vgg_loss(output_steg_2)
                #vgg_on_secret1 = vgg_loss(secret_1)
                #vgg_on_secret2 = vgg_loss(secret_2)
                #vgg_on_rev_secret1 = vgg_loss(rev_secret_1)
                #vgg_on_rev_secret2 = vgg_loss(rev_secret_2)
                
                #感知损失
                perc_loss = guide_loss(vgg_on_cov, vgg_on_steg1) + guide_loss(vgg_on_cov, vgg_on_steg2)
                #鲁棒损失
                roubst_loss = guide_loss(secret_dwt_1.cuda(),rev_secret_dwt.cuda()) + \
                            guide_loss(secret_dwt_2.cuda(),rev_secret_dwt_2.cuda())
                
                l_loss_1 = guide_loss(output_steg_dwt_low_1.cuda(), cover_dwt_low.cuda())
                l_loss_2 = guide_loss(output_steg_dwt_low_2.cuda(), cover_dwt_low.cuda())
                # 下面rev_secret_1和rev_secret_2是攻击后图像提取的秘密图像
                r_loss_1 = reconstruction_loss(rev_secret_1, secret_1)
                r_loss_2 = reconstruction_loss(rev_secret_2, secret_2)

                # 结合各损失的权重得到总损失后进行反向传播和参数更新
                # -->这里增大r_loss_1和r_loss_2的权重(config)，让网络知道鲁棒性是最重要的
                total_loss = c.lamda_reconstruction_1 * r_loss_1 + c.lamda_reconstruction_2 * r_loss_2 + c.lamda_guide_1 * g_loss_1\
                         + c.lamda_guide_2 * g_loss_2 + c.lamda_low_frequency_1 * l_loss_1 + c.lamda_low_frequency_2 * l_loss_2 \
                         + c.lamda_roubst * roubst_loss
                total_loss = total_loss + 0.01 * perc_loss
                total_loss.backward()

                if c.optim_step_1:
                    optim1.step()

                if c.optim_step_2:
                    optim2.step()

                if c.optim_step_3:
                    optim3.step()

                optim1.zero_grad()
                optim2.zero_grad()
                optim3.zero_grad()

                # 记录损失历史，将每个损失项的值记录到相应的列表中, 便于后续调整
                loss_history.append([total_loss.item(), 0.])
                loss_history_g1.append(g_loss_1.item())
                loss_history_g2.append(g_loss_2.item())
                loss_history_r1.append(r_loss_1.item())
                loss_history_r2.append(r_loss_2.item())
                loss_history_imp.append(impmap_loss.item())
                loss_history_roubst.append(roubst_loss.item())

            #################
            #     val:    #
            #################
            # 模型评估，如果当前 epoch 满足验证频率 (val_freq) 的条件进入验证模式 
            if i_epoch % c.val_freq == 1:
                # 禁用梯度计算
                with torch.no_grad():
                    psnr_s1 = []
                    psnr_s2 = []
                    psnr_c1 = []
                    psnr_c2 = []
                    net1.eval()
                    net2.eval()
                    net3.eval()
                    # 准备验证数据，与训练过程类似不做解释
                    for x in testloader:
                        x = x.to(device)
                        cover = x[:x.shape[0] // 3]  # channels = 3
                        secret_1 = x[x.shape[0] // 3: 2 * x.shape[0] // 3]
                        secret_2 = x[2 * x.shape[0] // 3: 3 * x.shape[0] // 3]

                        cover_dwt = dwt(cover)  # channels = 12
                        secret_dwt_1 = dwt(secret_1)
                        secret_dwt_2 = dwt(secret_2)

                        input_dwt_1 = torch.cat((cover_dwt, secret_dwt_1), 1)  # channels = 24

                        #################
                        #    forward1:   #
                        #################
                        output_dwt_1 = net1(input_dwt_1)  # channels = 24
                        output_steg_dwt_1 = output_dwt_1.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                        output_z_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                        # get steg1
                        output_steg_1 = iwt(output_steg_dwt_1).to(device)  # channels = 3

                        #################
                        #    forward2:   #
                        #################
                        if c.use_imp_map:
                            imp_map = net3(cover, secret_1, output_steg_1)  # channels = 3
                        else:
                            imp_map = torch.zeros(cover.shape).cuda()

                        imp_map_dwt = dwt(imp_map)  # channels = 12
                        input_dwt_2 = torch.cat((output_steg_dwt_1, imp_map_dwt), 1)  # 24, without secret2
                        input_dwt_2 = torch.cat((input_dwt_2, secret_dwt_2), 1)  # 36

                        output_dwt_2 = net2(input_dwt_2)  # channels = 36
                        output_steg_dwt_2 = output_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                        output_z_dwt_2 = output_dwt_2.narrow(1, 4 * c.channels_in, output_dwt_2.shape[1] - 4 * c.channels_in)  # channels = 24

                        # get steg2
                        output_steg_2 = iwt(output_steg_dwt_2).to(device)  # channels = 3

                        #----------------鲁棒性模块(见config)-------------------#
                        if (c.choice_attack == 1):
                            # 对steg_2加入随机噪声
                            noise = torch.randn_like(output_steg_2) * c.noise_level
                            # 将噪声添加到输出图像
                            output_steg_2_with_noise = output_steg_2 + noise
                            # 使用clamp将像素值限制在合理范围内，避免超过张量的上下限
                            floor_level = torch.min(output_steg_2)
                            upper_level = torch.max(output_steg_2)
                            output_steg_2_with_noise = output_steg_2_with_noise.clamp(floor_level, upper_level)
                            output_steg_dwt_2_with_noise = dwt(output_steg_2_with_noise)
                            output_steg_dwt_low_with_noise = output_steg_dwt_2_with_noise.narrow(1, 0, c.channels_in)
                            
                        elif (c.choice_attack == 2):
                            output_steg_2_toPNG = torch.floor(output_steg_2 * 255) / 255.0
                            output_steg_dwt_2_toPNG = dwt(output_steg_2_toPNG)
            
                            output_steg_dwt_low_toPNG = output_steg_dwt_2_toPNG.narrow(1, 0, c.channels_in)
                        else:
                            pass
                        #--------------------end------------------------#

                        #################
                        #   backward2:   #
                        #################

                        output_z_guass_1 = gauss_noise(output_z_dwt_1.shape)  # channels = 12
                        output_z_guass_2 = gauss_noise(output_z_dwt_2.shape)  # channels = 24
                        
                        if (c.choice_attack == 1):
                            output_steg_dwt_2 = output_steg_dwt_2_with_noise
                        elif (c.choice_attack == 2):
                            output_steg_dwt_2 = output_steg_dwt_2_toPNG
                        else:
                            pass
                        
                        #改动在这里：验证部分使用攻击后的图片进行秘密图像的提取，以验证其鲁棒性
                        #后面计算峰值信噪比的时候，原始载体图像和嵌入后的载体图像使用攻击前的嵌入后载体图像，因为攻击是我们人为加的
                        output_rev_dwt_2 = torch.cat((output_steg_dwt_2, output_z_guass_2), 1)  # channels = 36

                        rev_dwt_2 = net2(output_rev_dwt_2, rev=True)  # channels = 36

                        rev_steg_dwt_1 = rev_dwt_2.narrow(1, 0, 4 * c.channels_in)  # channels = 12
                        rev_secret_dwt_2 = rev_dwt_2.narrow(1, output_dwt_2.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12

                        rev_steg_1 = iwt(rev_steg_dwt_1).to(device)  # channels = 3
                        rev_secret_2 = iwt(rev_secret_dwt_2).to(device)  # channels = 3

                        #################
                        #   backward1:   #
                        #################
                        output_rev_dwt_1 = torch.cat((rev_steg_dwt_1, output_z_guass_1), 1)  # channels = 24

                        rev_dwt_1 = net1(output_rev_dwt_1, rev=True)  # channels = 24

                        rev_secret_dwt = rev_dwt_1.narrow(1, rev_dwt_1.shape[1] - 4 * c.channels_in, 4 * c.channels_in)  # channels = 12
                        rev_secret_1 = iwt(rev_secret_dwt).to(device)

                        # 计算峰值信噪比(原始秘密图像与提取出的秘密图像、原始载体图像和嵌入后的载体图像)
                        secret_rev1_255 = rev_secret_1.cpu().numpy().squeeze() * 255
                        secret_rev2_255 = rev_secret_2.cpu().numpy().squeeze() * 255
                        secret_1_255 = secret_1.cpu().numpy().squeeze() * 255
                        secret_2_255 = secret_2.cpu().numpy().squeeze() * 255

                        cover_255 = cover.cpu().numpy().squeeze() * 255
                        steg_1_255 = output_steg_1.cpu().numpy().squeeze() * 255
                        steg_2_255 = output_steg_2.cpu().numpy().squeeze() * 255

                        psnr_temp1 = computePSNR(secret_rev1_255, secret_1_255)
                        psnr_s1.append(psnr_temp1)
                        psnr_temp2 = computePSNR(secret_rev2_255, secret_2_255)
                        psnr_s2.append(psnr_temp2)

                        psnr_temp_c1 = computePSNR(cover_255, steg_1_255)
                        psnr_c1.append(psnr_temp_c1)
                        psnr_temp_c2 = computePSNR(cover_255, steg_2_255)
                        psnr_c2.append(psnr_temp_c2)

                    # 对 PSNR 进行记录和可视化, 目的是跟踪模型在验证集上的性能
                    writer.add_scalars("PSNR", {"S1 average psnr": np.mean(psnr_s1)}, i_epoch)
                    writer.add_scalars("PSNR", {"C1 average psnr": np.mean(psnr_c1)}, i_epoch)
                    writer.add_scalars("PSNR", {"S2 average psnr": np.mean(psnr_s2)}, i_epoch)
                    writer.add_scalars("PSNR", {"C2 average psnr": np.mean(psnr_c2)}, i_epoch)

            # 计算整个 epoch 的平均损失
            epoch_losses = np.mean(np.array(loss_history), axis=0)
            # 记录当前学习率的对数值，将其作为第二个元素添加到 epoch_losses 数组中，这样可以在可视化中追踪学习率的变化。
            epoch_losses[1] = np.log10(optim1.param_groups[0]['lr'])

            # 计算many loss的平均值
            epoch_losses_g1 = np.mean(np.array(loss_history_g1))
            epoch_losses_g2 = np.mean(np.array(loss_history_g2))
            epoch_losses_r1 = np.mean(np.array(loss_history_r1))
            epoch_losses_r2 = np.mean(np.array(loss_history_r2))
            epoch_losses_imp = np.mean(np.array(loss_history_imp))
            epoch_losses_roubst = np.mean(np.array(loss_history_roubst))
            print(f"epoch_losses_g1 : {epoch_losses_g1}")
            print(f"epoch_losses_g2 : {epoch_losses_g2}")
            print(f"epoch_losses_r1 : {epoch_losses_r1}")
            print(f"epoch_losses_r2 : {epoch_losses_r2}")
            print(f"epoch_losses_imp : {epoch_losses_imp}")
            print(f"epoch_losses_roubst : {epoch_losses_roubst}")

            # 使用 viz 对象可视化展示当前 epoch 的损失
            viz.show_loss(epoch_losses)
            writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)
            writer.add_scalars("Train", {"g1_Loss": epoch_losses_g1}, i_epoch)
            writer.add_scalars("Train", {"g2_Loss": epoch_losses_g2}, i_epoch)
            writer.add_scalars("Train", {"r1_Loss": epoch_losses_r1}, i_epoch)
            writer.add_scalars("Train", {"r2_Loss": epoch_losses_r2}, i_epoch)
            writer.add_scalars("Train", {"imp_Loss": epoch_losses_imp}, i_epoch)
            writer.add_scalars("Train", {"roubst_Loss": epoch_losses_roubst}, i_epoch)

            # 如果当前 epoch 大于 0 且是保存的频率的倍数，执行以下操作
            if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
                # 保存网络123的权重和优化器状态
                torch.save({'opt': optim1.state_dict(),
                            'net': net1.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_1' % i_epoch + '.pt')
                torch.save({'opt': optim2.state_dict(),
                            'net': net2.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_2' % i_epoch + '.pt')
                torch.save({'opt': optim3.state_dict(),
                            'net': net3.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i_3' % i_epoch + '.pt')
            # 更新网络1的学习率调度器
            weight_scheduler1.step()
            weight_scheduler2.step()
            weight_scheduler3.step()

        # 保存最终的训练模型
        torch.save({'opt': optim1.state_dict(),
                    'net': net1.state_dict()}, c.MODEL_PATH + 'model_1' + '.pt')
        torch.save({'opt': optim2.state_dict(),
                    'net': net2.state_dict()}, c.MODEL_PATH + 'model_2' + '.pt')
        torch.save({'opt': optim3.state_dict(),
                    'net': net3.state_dict()}, c.MODEL_PATH + 'model_3' + '.pt')
        writer.close()

    # 捕获任何发生的异常
    except:
        if c.checkpoint_on_error:
            # 保存网络123的权重和优化器状态为 'model_ABORT_123'
            torch.save({'opt': optim1.state_dict(),
                        'net': net1.state_dict()}, c.MODEL_PATH + 'model_ABORT_1' + '.pt')
            torch.save({'opt': optim2.state_dict(),
                        'net': net2.state_dict()}, c.MODEL_PATH + 'model_ABORT_2' + '.pt')
            torch.save({'opt': optim3.state_dict(),
                        'net': net3.state_dict()}, c.MODEL_PATH + 'model_ABORT_3' + '.pt')
        raise

    finally:
        viz.signal_stop()
