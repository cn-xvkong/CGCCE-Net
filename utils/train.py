import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
#from catalyst.contrib.nn import Lookahead
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from tqdm import tqdm
from utils import data_loader
import random
from utils.metrics import Evaluator
import logging
from networktest.HGRN import  HGRN
from networktest.HGRT import  HGRT
from networktest.HGRR import  HGRR,HGRR1
from network.Newwork import NWT
from network.Newwork1 import NWTx
from xlstmnet.CDxLSTM import CDxlstmce
from xlstmnet.STxlstmConvNeXt import STxlstmCD

import time

from utils.utils import adjust_lr

start=time.time()

def seed_everything(seed):
    # 设置python内置随机数生成器的种子
    random.seed(seed)
    # 设置环境变量PYTHONHASHSEED，保证hash的结果的一致性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置NumPy的随机数生成器的种子
    np.random.seed(seed)
    # 设置PyTorch的随机数生成器的种子
    torch.manual_seed(seed)
    # 设置PyTorch的CUDA的随机数生成器的种子
    torch.cuda.manual_seed(seed)
    # 保证在不同机器上的训练结果的一致性
    torch.backends.cudnn.deterministic = True
    # 开启CUDNN的benchmark模式，提高训练速度
    torch.backends.cudnn.benchmark = True

def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches):
    print("在train函数内")  # 添加这一行
    vis = visual.Visualization()#创建用于可视化的实例
    vis.create_summary(data_name)#创建与数据集相关的summary
    global best_iou#全局变量，用于保存历史最佳的IoU值
    global best_f1#全局变量，用于保存历史最佳的F1值
    epoch_loss = 0#用于记录每个epoch的损失

    #训练模式
    net.train(True)
    length = 0#记录训练数据集的长度
    st = time.time()#记录训练开始时间，没用到
    #A为img1，B为img2，mask是掩码，即标签

    # 添加日志记录
    log_file_path = os.path.join(save_path, 'training_log.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    logging.info('Epoch\tTrain_Loss\tTrain_IoU\tTrain_Precision\tTrain_Recall\tTrain_F1')

    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()#梯度清零，防止梯度累积
        preds = net(A,B)#模型前向传播，获得预测结果
        # 计算损失，这里使用的是BCEWithLogitsLoss损失函数
        loss = criterion(preds[0], Y) + criterion(preds[1], Y) + criterion(preds[2], Y) + criterion(preds[3], Y)
        # ---- loss function ----
        loss.backward()#反向传播，计算梯度
        optimizer.step()#更新模型参数
        #scheduler.step()#更新学习率调度器
        epoch_loss += loss.item()#累加损失

        output = torch.sigmoid(preds[1])#将模型输出通过sigmoid函数转换为概率
        output[output >= 0.5] = 1#黑
        output[output < 0.5] = 0 #白
        pred = output.data.cpu().numpy().astype(int)#将预测结果转换为numpy数组
        target = Y.cpu().numpy().astype(int)# 将标签转换为numpy数组

        Eva_train.add_batch(target, pred)#计算训练集的评估指标
        length += 1 #记录训练数据集的长度

    #评估指标
    IoU = Eva_train.Intersection_over_Union()[1] # 计算训练集上的IoU
    Pre = Eva_train.Precision()[1]#计算训练集上的Pre
    Recall = Eva_train.Recall()[1]#计算训练集上的Recall
    F1 = Eva_train.F1()[1]        #计算训练集上的F1
    train_loss = epoch_loss / length#计算训练集上的平均损失
    #可视化指标
    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')
    #输出指标
    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU, Pre, Recall, F1))

    print("Begin evaluation")
    #验证模式
    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(val_loader):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]#只需要final map
            output = torch.sigmoid(preds) #将模型输出通过sigmoid函数转换为概率
            output[output >= 0.5] = 1  # 大于等于0.5的设置为1
            output[output < 0.5] = 0   # 小于0.5的设置为0
            pred = output.data.cpu().numpy().astype(int)# 转换为numpy数组
            target = Y.cpu().numpy().astype(int)# 转换为numpy数组

            Eva_val.add_batch(target, pred)    # 计算验证集的评估指标

            length += 1   # 记录验证数据集的长度
    IoU = Eva_val.Intersection_over_Union()# 计算验证集上的IoU
    Pre = Eva_val.Precision()# 计算验证集上的Precision
    Recall = Eva_val.Recall()# 计算验证集上的Recall
    F1 = Eva_val.F1()        # 计算验证集上的F1

    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]   # 记录当前的IoU
    new_f1 = F1[1]
    if new_f1 >= best_f1:
        best_f1 = new_f1
    # if new_iou >= best_iou:# 如果当前的IoU比历史最佳IoU要好
        best_iou = new_iou # 更新历史最佳IoU
        best_epoch = epoch # 更新历史最佳的epoch
        best_net = net.state_dict()# 保存当前模型参数为历史最佳模型
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')# 保存历史最佳模型的参数
    #print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, best_f1))
    logging.info('%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
        epoch, train_loss, IoU[1], Pre[1], Recall[1], F1[1]))
    vis.close_summary()  # 关闭可视化工具


if __name__ == '__main__':
    seed_everything(42)#伪随机数生成器种子为42
    import argparse#通过命令行参数设置训练的超参数

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number') #修改这里！！！
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')#学习率设置为0.0005
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size') #修改这里！！！
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') #修改这里！！！
    parser.add_argument('--data_name', type=str, default='WHU', #修改这里！！！
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='STxlstmCD',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    # 设置训练用GPU
    # if opt.gpu_id == '0':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     print('USE GPU 0')
    # elif opt.gpu_id == '1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #     print('USE GPU 1')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = 'CDData/LEVIR/train/'
        opt.val_root = 'CDData/LEVIR/test/'
    elif opt.data_name == 'WHU':
        opt.train_root = 'CDData/WHU/train/'
        opt.val_root = 'CDData/WHU/test/'
    # elif opt.data_name == 'CDD':
    #     opt.train_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/train/'
    #     opt.val_root = '/data/chengxi.han/data/CDD_ChangeDetectionDataset/Real/subset/val/'
    # elif opt.data_name == 'DSIFN':
    #     opt.train_root = '/data/chengxi.han/data/DSIFN256/train/'
    #     opt.val_root = '/data/chengxi.han/data/DSIFN256/val/'
    # elif opt.data_name == 'SYSU':
    #     opt.train_root = '/data/chengxi.han/data/SYSU-CD/train/'
    #     opt.val_root = '/data/chengxi.han/data/SYSU-CD/val/'
    # elif opt.data_name == 'S2Looking':
    #     opt.train_root = '/data/chengxi.han/data/S2Looking256/train/'
    #     opt.val_root = '/data/chengxi.han/data/S2Looking256/val/'


    #创建训练和验证数据加载器
    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    #初始化评估器，num_class为2表明为二分类问题
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)
    if opt.model_name == 'STxlstmCD':
        model = STxlstmCD().cuda()
    elif opt.model_name == 'HGRN':
        model = HGRN().cuda()
    elif opt.model_name == 'HGRR':
        model = HGRR().cuda()
    elif opt.model_name == 'NWT':
        model = NWT().cuda()
    elif opt.model_name == 'HGRR1':
        model = HGRR1().cuda()
    elif opt.model_name == 'NWTx':
        model = NWTx().cuda()

    #损失函数交叉熵
    criterion = nn.BCEWithLogitsLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    #base_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)

    #AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    def linear_lambda(epoch):
        total_epochs = opt.epoch  # 假设总共训练 100 个 epoch
        return 1 - (epoch / total_epochs)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)
    #余弦退火


    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path# 将命令行参数中提供的路径保存到'save_path'
    if not os.path.exists(save_path):# 如果由'save_path'指定的目录不存在，则创建它
        os.makedirs(save_path)
    data_name = opt.data_name# 从命令行参数获取数据集的名称
    best_iou = 0.0# 初始化一个变量以存储最佳交并比（Intersection over Union，IoU）值
    best_f1 = 0.0

    print("Start train...")
    # args = parser.parse_args()
    # print('现在的数据是：',args.data_name)

    for epoch in range(1, opt.epoch):# 循环遍历时期，从1开始，直到（但不包括）'opt.epoch'
        # 打印优化器中每个参数组的当前学习率
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        #cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # 重置训练和验证评估器的度量指标
        Eva_train.reset()
        Eva_val.reset()
        # 调用'train'函数执行当前时期的训练
        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer, opt.epoch)
        # 使用余弦退火温暖重启调度程序来更新下一个时期的学习率
        lr_scheduler.step()
        # print('现在的数据是：', args.data_name)

end=time.time()
print('程序训练train的时间为:',end-start)