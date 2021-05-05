# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:58:33 2021

@author: User
"""

import os
import time
import numpy as np
import open3d as o3d
from tqdm import tqdm
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from Module_ModelSelector import PointNetCls, PointNet2
from Module_ModelSelector_DataLoader import ModelNet40H5, GetAllModelFromCategory

from Module_Parser import ModelSelectorParser
from Module_Utils import textIO, DrawAxis


def eval_one_epoch(net, testLoader, args):
    net.eval()
    avgClsLoss = 0
    avgL1Loss = 0
    avgLoss = 0
    cnt = 0
    for srcPC, tmpPC, label in tqdm(testLoader):
        if (args.cuda):
            srcPC = srcPC.cuda()
            tmpPC = tmpPC.cuda()
            label = label.cuda()
        clsProbVec, globalFeat, globalFeat2 = net(srcPC, tmpPC)
        clsLoss = F.nll_loss(clsProbVec, label.squeeze())
        l1Loss = F.l1_loss(globalFeat, globalFeat2)
        loss = clsLoss + l1Loss
        avgClsLoss += clsLoss.item()
        avgL1Loss += l1Loss.item()
        avgLoss += loss.item()
        cnt += 1
    return avgLoss / cnt, avgClsLoss / cnt, avgL1Loss / cnt


def train_one_epoch(net, opt, trainLoader, args):
    net.train()
    avgClsLoss = 0
    avgL1Loss = 0
    avgLoss = 0
    cnt = 0
    for srcPC, tmpPC, label in tqdm(trainLoader):
        if (args.cuda):
            srcPC = srcPC.cuda()
            tmpPC = tmpPC.cuda()
            label = label.cuda()
        opt.zero_grad()
        
        clsProbVec, globalFeat, globalFeat2 = net(srcPC, tmpPC)
        clsLoss = F.nll_loss(clsProbVec, label.squeeze())
        l1Loss = F.l1_loss(globalFeat, globalFeat2) if (args.L1Loss) else 0
        loss = clsLoss + l1Loss
        loss.backward()
        
        opt.step()
        
        avgClsLoss += clsLoss.item()
        avgL1Loss += l1Loss.item() if (args.L1Loss) else 0
        avgLoss += loss.item()
        cnt += 1
    return avgLoss / cnt, avgClsLoss / cnt, avgL1Loss / cnt


def train(net, trainLoader, validLoader, textLog, boardLog, args):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    
    bestTrainLoss = 0
    bestTrainEpoch = 0
    bestValidLoss = 0
    bestValidEpoch = 0
    for epoch in range(args.epochs):
        loss, clsLoss, l1Loss = train_one_epoch(net, opt, trainLoader, args)
        scheduler.step()
        
        if (epoch == 0):
            bestTrainLoss = loss
        elif (bestTrainLoss > loss):
            bestTrainLoss = loss
            bestTrainEpoch = epoch
            SaveModel(net, args.saveModelDir, 'model_ModelSelector_best.pth', args.multiCuda)
        textLog.writeLog('train\tepoch %d\tloss:%f\tbest epoch %d\tloss:%f'%(epoch, loss, bestTrainEpoch, bestTrainLoss))
        boardLog.add_scalar('train/loss', loss, epoch)
        boardLog.add_scalar('train/best_loss', bestTrainLoss, epoch)
        boardLog.add_scalar('train/clsLoss', clsLoss, epoch)
        boardLog.add_scalar('train/l1Loss', l1Loss, epoch)
        if (epoch % 10 == 0):
            SaveModel(net, args.saveModelDir, 'model_ModelSelector_%d.pth' %epoch, args.multiCuda)
            loss, clsLoss, l1Loss = eval_one_epoch(net, validLoader, args)
            if (epoch == 0):
                bestValidLoss = loss
            elif (bestValidLoss > loss):
                bestValidLoss = loss
                bestValidEpoch = epoch
            textLog.writeLog('valid\tepoch %d\tloss:%f\tbest epoch %d\tloss:%f'%(epoch, loss, bestValidEpoch, bestValidLoss))
            boardLog.add_scalar('valid/loss', loss, epoch)
            boardLog.add_scalar('valid/best_loss', bestValidLoss, epoch)
            boardLog.add_scalar('valid/clsLoss', clsLoss, epoch)
            boardLog.add_scalar('valid/l1Loss', l1Loss, epoch)


def SaveModel(net, DIR_PATH, modelName, multiCudaF):
    if (multiCudaF):
        torch.save(net.module.state_dict(), os.path.join(DIR_PATH, modelName))
    else:
        torch.save(net.state_dict(), os.path.join(DIR_PATH, modelName))


def CalBestTemplate(net, testLoader, args):
    srcPoints, candidatePointsList, srcPath, candidatePaths = testLoader.GetRandomTestSet(60, srcRotateF=True)
    srcPoints = torch.tensor(srcPoints).view(1, -1, 3)
    if (args.cuda): srcPoints = srcPoints.cuda()
    
    rankList = []
    
    net.eval()
    for i, candidatePoints in enumerate(candidatePointsList):
        candidatePoints = torch.tensor(candidatePoints).view(1, -1, 3)
        if (args.cuda): candidatePoints = candidatePoints.cuda()
        _, srcFeat, tmpFeat = net(srcPoints, candidatePoints)
        if (args.cuda): srcFeat = srcFeat.cpu()
        if (args.cuda): tmpFeat = tmpFeat.cpu()
        srcFeat = srcFeat.detach().numpy().squeeze()
        tmpFeat = tmpFeat.detach().numpy().squeeze()
        loss = np.mean(np.abs(srcFeat - tmpFeat))
        print('%d: %f' %(i, loss))
        rankList.append([candidatePaths[i], loss])
    rankList = sorted(rankList, key=itemgetter(1))
    rankPathList = np.array(rankList)[:,0]
    inRank5F = 0
    if (srcPath in rankPathList[:1]):
        print('Rank 1')
        inRank5F = 1
    elif (srcPath in rankPathList[:3]):
        print('Rank 3')
        inRank5F = 3
    elif (srcPath in rankPathList[:5]):
        print('Rank 5')
        inRank5F = 5
    elif (srcPath in rankPathList[:10]):
        print('Rank 10')
    elif (srcPath in rankPathList[:20]):
        print('Rank 20')
    elif (srcPath in rankPathList[:50]):
        print('Rank 50')
    else:
        print('Out of Rank')
    print(rankList[0])
    if (inRank5F > 0):
        srcPCD = o3d.io.read_triangle_mesh(srcPath)
        maxBound = srcPCD.get_max_bound()
        minBound = srcPCD.get_min_bound()
        length = np.linalg.norm(maxBound - minBound, 2)
        srcPCD.scale(1 / length, center=srcPCD.get_center())
        srcPCD.translate(-srcPCD.get_center())
        srcPCD.paint_uniform_color([1, 0, 0])
        for path, score in rankList[:inRank5F]:
            pcd = o3d.io.read_triangle_mesh(path)
            maxBound = pcd.get_max_bound()
            minBound = pcd.get_min_bound()
            length = np.linalg.norm(maxBound - minBound, 2)
            pcd.scale(1 / length, center=pcd.get_center())
            pcd.translate(-pcd.get_center())
            pcd.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([pcd, srcPCD, DrawAxis(1)], window_name = 'Result')


def initEnv(args):
    try:
        if (not os.path.exists(args.saveModelDir)):
            os.mkdir(args.saveModelDir)
        if (not os.path.exists(args.dataset)):
            raise 'Dataset path error'
        if (args.eval and not os.path.exists(args.modelPath)):
            raise 'Model path error'
        textLog = textIO(args)
        textLog.writeLog(time.ctime())
        return textLog
    except:
        raise 'Unexpected error'


if (__name__ == '__main__'):
    args = ModelSelectorParser()
    textLog = initEnv(args)
    if (args.featModel == 'pointnet'):
        net = PointNetCls(k=40, feature_transform=True)
    elif (args.featModel == 'pointnet2'):
        net = PointNet2(input_feat_dim=3, k=40)
    if (not torch.cuda.is_available() or not args.cuda):
        device = torch.device('cpu')
        args.cuda = False
        args.multiCuda = False
    elif (args.multiCuda and torch.cuda.device_count() > 1):# Use multiple cuda device
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    elif (torch.device(args.cudaDevice)):
        device = torch.device(args.cudaDevice)
        args.multiCuda = False
    else:
        device = torch.device('cpu')
        args.cuda = False
        args.multiCuda = False
    net.to(device)
    textLog.writeLog(args.__str__())
    if (not args.eval):
        boardLog = SummaryWriter(log_dir=args.saveModelDir)
        
        validLoader = DataLoader(ModelNet40H5(dataPartition='test', DIR_PATH=args.dataset, 
                                             srcPointNum=args.inputPoints, 
                                             tmpPointNum=args.inputPoints, 
                                             gaussianNoise=args.gaussianNoise, 
                                             scaling=args.scaling), 
                                batch_size=args.batchSize, shuffle=True)
        
        trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                              srcPointNum=args.inputPoints, 
                                              tmpPointNum=args.inputPoints, 
                                              gaussianNoise=args.gaussianNoise, 
                                              scaling=args.scaling), 
                                batch_size=args.batchSize, shuffle=True)
        
        train(net, trainLoader, validLoader, textLog, boardLog, args)
        
        boardLog.close()
    else:
        testLoader = GetAllModelFromCategory(DIR_PATH='D:\\Datasets\\ModelNet40', category='chair', numOfPoints=args.inputPoints)
        net.load_state_dict(torch.load(args.modelPath))
        CalBestTemplate(net, testLoader, args)
    textLog.close()