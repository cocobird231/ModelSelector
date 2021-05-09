# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:58:33 2021

@author: User
"""

import os
import time
import numpy as np
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
from Module_ModelSelector_DataLoader import ModelNet40H5, ModelSelectorValidDataset

from Module_Parser import ModelSelectorParser
from Module_Utils import textIO


def eval_one_epoch(net, testLoader, args):
    net.eval()
    avgClsLoss = 0
    avgL1Loss = 0
    avgTripletLoss = 0
    avgLoss = 0
    cnt = 0
    for srcPC, tmpPC, negPC, label in tqdm(testLoader):
        if (args.cuda):
            srcPC = srcPC.cuda()
            tmpPC = tmpPC.cuda()
            negPC = negPC.cuda()
            label = label.cuda()
        clsProbVec, globalFeat, globalFeat2, globalFeatNeg = net(srcPC, tmpPC, negPC)
        clsLoss = F.nll_loss(clsProbVec, label.squeeze())
        l1Loss = F.l1_loss(globalFeat, globalFeat2) if (args.L1Loss) else 0
        tripletLoss = F.l1_loss(globalFeat, globalFeat2) + 1 / (F.l1_loss(globalFeat, globalFeatNeg)) if (args.triplet) else 0
        loss = clsLoss + l1Loss + tripletLoss
        avgClsLoss += clsLoss.item()
        avgL1Loss += l1Loss.item() if (args.L1Loss) else 0
        avgTripletLoss += tripletLoss.item() if (args.triplet) else 0
        avgLoss += loss.item()
        cnt += 1
    return avgLoss / cnt, avgClsLoss / cnt, avgL1Loss / cnt, avgTripletLoss / cnt


def train_one_epoch(net, opt, trainLoader, args):
    net.train()
    avgClsLoss = 0
    avgL1Loss = 0
    avgTripletLoss = 0
    avgLoss = 0
    cnt = 0
    for srcPC, tmpPC, negPC, label in tqdm(trainLoader):
        if (args.cuda):
            srcPC = srcPC.cuda()
            tmpPC = tmpPC.cuda()
            negPC = negPC.cuda()
            label = label.cuda()
            
        opt.zero_grad()
        
        clsProbVec, globalFeat, globalFeat2, globalFeatNeg = net(srcPC, tmpPC, negPC)
        clsLoss = F.nll_loss(clsProbVec, label.squeeze())
        l1Loss = F.l1_loss(globalFeat, globalFeat2) if (args.L1Loss) else 0
        tripletLoss = F.l1_loss(globalFeat, globalFeat2) + 1 / (F.l1_loss(globalFeat, globalFeatNeg)) if (args.triplet) else 0
        loss = clsLoss + l1Loss + tripletLoss
        loss.backward()
        
        opt.step()
        
        avgClsLoss += clsLoss.item()
        avgL1Loss += l1Loss.item() if (args.L1Loss) else 0
        avgTripletLoss += tripletLoss.item() if (args.triplet) else 0
        avgLoss += loss.item()
        cnt += 1
    return avgLoss / cnt, avgClsLoss / cnt, avgL1Loss / cnt, avgTripletLoss / cnt


def train(net, trainLoader, validLoader, textLog, boardLog, args):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    
    bestTrainLoss = 0
    bestTrainEpoch = 0
    bestValidLoss = 0
    bestValidEpoch = 0
    for epoch in range(args.epochs):
        loss, clsLoss, l1Loss, tripletLoss = train_one_epoch(net, opt, trainLoader, args)
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
        boardLog.add_scalar('train/tripletLoss', tripletLoss, epoch)
        if (epoch % 10 == 0):
            SaveModel(net, args.saveModelDir, 'model_ModelSelector_%d.pth' %epoch, args.multiCuda)
            loss, clsLoss, l1Loss, tripletLoss = eval_one_epoch(net, validLoader, args)
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
            boardLog.add_scalar('valid/tripletLoss', tripletLoss, epoch)
    return


def SaveModel(net, DIR_PATH, modelName, multiCudaF):
    if (multiCudaF):
        torch.save(net.module.state_dict(), os.path.join(DIR_PATH, modelName))
    else:
        torch.save(net.state_dict(), os.path.join(DIR_PATH, modelName))
    return


def CalBestTemplate(net, testLoader, args):
    net.eval()
    totalRankDict = {'Rank 1' : 0, 'Rank 3' : 0, 'Rank 5' : 0, 
                     'Rank 10' : 0, 'Rank 20' : 0, 'Rank 30' : 0, 'Out of Rank' : 0}
    for srcModelU, catModelUList, pathAns in testLoader:
        srcPCD = srcModelU.model.astype('float32')
        srcPts = torch.tensor(srcPCD).view(1, -1, 3)
        if (args.cuda) : srcPts = srcPts.cuda()
        rankList = []
        for catModelU in catModelUList:
            catPCD = catModelU.model.astype('float32')
            catPts = torch.tensor(catPCD).view(1, -1, 3)
            if (args.cuda) : catPts = catPts.cuda()
            _, srcFeat, tmpFeat, _ = net(srcPts, catPts, srcPts)
            if (args.cuda): srcFeat = srcFeat.cpu()
            if (args.cuda): tmpFeat = tmpFeat.cpu()
            srcFeat = srcFeat.detach().numpy().squeeze()
            tmpFeat = tmpFeat.detach().numpy().squeeze()
            loss = np.mean(np.abs(srcFeat - tmpFeat))
            # print('%s %s: %f' %(srcModelU.path[-20:], catModelU.path[-20:], loss))
            rankList.append([catModelU.path, loss])
        rankList = sorted(rankList, key=itemgetter(1))
        rankPathList = np.array(rankList)[:,0]
        rank = ''
        if (pathAns in rankPathList[:1]):
            rank = 'Rank 1'
        elif (pathAns in rankPathList[:3]):
            rank = 'Rank 3'
        elif (pathAns in rankPathList[:5]):
            rank = 'Rank 5'
        elif (pathAns in rankPathList[:10]):
            rank = 'Rank 10'
        elif (pathAns in rankPathList[:20]):
            rank = 'Rank 20'
        elif (pathAns in rankPathList[:30]):
            rank = 'Rank 30'
        else:
            rank = 'Out of Rank'
        print(srcModelU.path, rank)
        totalRankDict[rank] += 1
    print(totalRankDict)
    return


def initEnv(args):
    try:
        if (not os.path.exists(args.saveModelDir)):
            os.mkdir(args.saveModelDir)
        if (not args.eval and not os.path.exists(args.dataset)):
            raise 'Dataset path error'
        if (args.eval and not os.path.exists(args.modelPath)):
            raise 'Model path error'
        if (args.eval and not os.path.exists(args.validDataset)):
            raise 'validDataset path error'
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
        net = PointNet2(k=40)
    if (not torch.cuda.is_available() or not args.cuda):
        device = torch.device('cpu')
        args.cuda = False
        args.multiCuda = False
    elif (args.multiCuda and torch.cuda.device_count() > 1):# Use multiple cuda device
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    elif (torch.device(args.cudaDevice)):
        device = torch.device(args.cudaDevice)
        torch.cuda.set_device(device.index)
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
                                             scaling=args.scaling, 
                                             triplet=args.triplet), 
                                 batch_size=args.batchSize, shuffle=True)
        
        trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                              srcPointNum=args.inputPoints, 
                                              tmpPointNum=args.inputPoints, 
                                              gaussianNoise=args.gaussianNoise, 
                                              scaling=args.scaling, 
                                              triplet=args.triplet), 
                                 batch_size=args.batchSize, shuffle=True)
        
        train(net, trainLoader, validLoader, textLog, boardLog, args)
        
        boardLog.close()
    else:
        testLoader = ModelSelectorValidDataset(VALID_DIR=args.validDataset, specCatList = args.specCat if (args.specCat != None) else [])
        net.load_state_dict(torch.load(args.modelPath, map_location=device))
        CalBestTemplate(net, testLoader, args)
    textLog.close()