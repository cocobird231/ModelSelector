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
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from Module_ModelSelector import PointNetCls, PointNet2
from Module_ModelSelector_Criterion import ModelSelectorCriterion, GetModelSelectorCriterionLossDict
from Module_ModelSelector_DataLoader import ModelNet40H5, ModelSelectorValidDataset

from Module_Parser import ModelSelectorParser
from Module_Utils import textIO


def eval_one_epoch(net, testLoader, args):
    net.eval()
    avgLossDict = GetModelSelectorCriterionLossDict(args)
    avgLoss = 0
    cnt = 0
    for srcPC, tmpPC, negPC, label in tqdm(testLoader):
        if (args.cuda):
            srcPC = srcPC.cuda()
            tmpPC = tmpPC.cuda()
            negPC = negPC.cuda()
            label = label.cuda()
        clsProbVec, globalFeat, globalFeat2, globalFeatNeg = net(srcPC, tmpPC, negPC)
        loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, globalFeatNeg, clsProbVec, label.squeeze(), args)
        for lossType in lossDict : avgLossDict[lossType] += lossDict[lossType].item()
        avgLoss += loss.item()
        cnt += 1
    for key in avgLossDict : avgLossDict[key] /= cnt
    return avgLoss / cnt, avgLossDict


def train_one_epoch(net, opt, trainLoader, args):
    net.train()
    avgLossDict = GetModelSelectorCriterionLossDict(args)
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
        loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, globalFeatNeg, clsProbVec, label.squeeze(), args)
        loss.backward()
        
        opt.step()
        
        for lossType in lossDict : avgLossDict[lossType] += lossDict[lossType].item()
        avgLoss += loss.item()
        cnt += 1
    for key in avgLossDict : avgLossDict[key] /= cnt
    return avgLoss / cnt, avgLossDict


def train(net, trainLoader, validLoader, textLog, boardLog, args):
    opt = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    if (args.multiLR) : scheduler = MultiStepLR(opt, milestones=args.multiLR, gamma=0.1)
    
    bestTrainLoss = 0
    bestTrainEpoch = 0
    bestValidLoss = 0
    bestValidEpoch = 0
    for epoch in range(args.epochs):
        loss, lossDict = train_one_epoch(net, opt, trainLoader, args)
        if (args.multiLR) : scheduler.step()
        
        if (epoch == 0):
            bestTrainLoss = loss
        elif (bestTrainLoss > loss):
            bestTrainLoss = loss
            bestTrainEpoch = epoch
            SaveModel(net, args.saveModelDir, 'model_ModelSelector_best.pth', args.multiCuda)
        textLog.writeLog('train\tepoch %d\tloss:%f\tbest epoch %d\tloss:%f'%(epoch, loss, bestTrainEpoch, bestTrainLoss))
        boardLog.add_scalar('train/loss', loss, epoch)
        boardLog.add_scalar('train/best_loss', bestTrainLoss, epoch)
        for key in lossDict : boardLog.add_scalar('train/%s' %key, lossDict[key], epoch)
        if (epoch % 10 == 0):
            SaveModel(net, args.saveModelDir, 'model_ModelSelector_%d.pth' %epoch, args.multiCuda)
            loss, lossDict = eval_one_epoch(net, validLoader, args)
            if (epoch == 0):
                bestValidLoss = loss
            elif (bestValidLoss > loss):
                bestValidLoss = loss
                bestValidEpoch = epoch
            textLog.writeLog('valid\tepoch %d\tloss:%f\tbest epoch %d\tloss:%f'%(epoch, loss, bestValidEpoch, bestValidLoss))
            boardLog.add_scalar('valid/loss', loss, epoch)
            boardLog.add_scalar('valid/best_loss', bestValidLoss, epoch)
            for key in lossDict : boardLog.add_scalar('valid/%s' %key, lossDict[key], epoch)
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
        # print(srcModelU.path, rank)
        totalRankDict[rank] += 1
    print(totalRankDict)
    return


def initEnv(args):
    try:
        if (not os.path.exists(args.saveModelDir)):
            os.mkdir(args.saveModelDir)
        if (not os.path.exists(args.saveLogDir)):
            os.mkdir(args.saveLogDir)
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


def initDevice(args):
    if (not args.cuda or not torch.cuda.is_available()):
        device = torch.device('cpu')
        args.cuda = False
        args.multiCuda = False
    elif (torch.device(args.cudaDevice)):
        device = torch.device(args.cudaDevice)
        torch.cuda.set_device(device.index)
    else:
        device = torch.device('cpu')
        args.cuda = False
        args.multiCuda = False
    return device, args


def initListArgs(args):
    if (args.multiLR) : args.multiLR = args.multiLR if len(args.multiLR) > 0 else None
    if (args.specCat) : args.specCat = args.specCat if len(args.specCat) > 0 else None
    return args


if (__name__ == '__main__'):
    args = ModelSelectorParser()
    textLog = initEnv(args)
    device, args = initDevice(args)
    args = initListArgs(args)
    if (args.featModel == 'pointnet') : net = PointNetCls(k=40, feature_transform=True)
    elif (args.featModel == 'pointnet2') : net = PointNet2(k=40)

    if (args.multiCuda and torch.cuda.device_count() > 1):# Use multiple cuda device
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    net.to(device)
    
    textLog.writeLog(args.__str__())
    if (not args.eval):
        boardLog = SummaryWriter(log_dir=args.saveModelDir)
        
        validLoader = DataLoader(ModelNet40H5(dataPartition='test', DIR_PATH=args.dataset, 
                                             srcPointNum=args.inputPoints, 
                                             tmpPointNum=args.inputPoints, 
                                             gaussianNoise=args.gaussianNoise, 
                                             scaling=args.scaling, 
                                             triplet=args.triplet or args.tripletL2 or args.tripletMg), 
                                 batch_size=args.batchSize, shuffle=True)
        
        trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                              srcPointNum=args.inputPoints, 
                                              tmpPointNum=args.inputPoints, 
                                              gaussianNoise=args.gaussianNoise, 
                                              scaling=args.scaling, 
                                              triplet=args.triplet or args.tripletL2 or args.tripletMg), 
                                 batch_size=args.batchSize, shuffle=True)
        
        train(net, trainLoader, validLoader, textLog, boardLog, args)
        
        boardLog.close()
    else:
        testLoader = ModelSelectorValidDataset(VALID_DIR=args.validDataset, specCatList = args.specCat if (args.specCat) else [])
        net.load_state_dict(torch.load(args.modelPath, map_location=device))
        CalBestTemplate(net, testLoader, args)
    textLog.close()