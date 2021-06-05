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

from Module_PointNetSeries import PointNetCls, PointNetFeat, PointNetComp, \
    PointNet2Cls, PointNet2Feat, PointNet2Comp, PointNet2Comp2, DGCNN, DGCNNFeat
from Module_ModelSelector_Criterion import ModelSelectorCriterion, GetModelSelectorCriterionLossDict
from Module_ModelNet40Series_DataLoader import ModelNet40H5, ModelSelectorValidDataset, GetModelNet40H5ReturnType

from Module_Parser import ModelSelectorParser
from Module_Utils import textIO

sepModelList = ['pointnet', 'pointnet2', 'dgcnn']
intModelList = ['pointnetComp', 'pointnet2Comp', 'pointnet2Comp2']
featModelList = ['pointnetFeat', 'pointnet2Feat', 'dgcnnFeat']
acceptModelList = [*sepModelList, *intModelList, *featModelList]
# Seperate models: pointnet, pointnet2, dgcnn               -> cls, feat = net(pc) or cls = net(pc)
# Integrate models: pointnetComp, pointnet2Comp             -> cls, feat, feat2 = net(pc1, pc2)
# Validation only: pointnetFeat, pointnet2Feat, dgcnnFeat   -> feat = net(pc)

def eval_one_epoch(net, testLoader, args):
    net.eval()
    avgLossDict = GetModelSelectorCriterionLossDict(args)
    avgLoss = 0
    cnt = 0
    for package in tqdm(testLoader):
        # srcPC, label              : loaderType=cls
        # srcPC, tmpPC, label       : loaderType=glob2
        # srcPC, tmpPC, negPC, label: loaderType=triplet
        
        if (args.DP):
            for i in range(0, len(package) - 1):
                randPointsPerBatch = int(np.random.uniform(128, args.inputPoints))
                package[i] = package[i][:, :randPointsPerBatch, :]
        
        srcPC = package[0].cuda() if (args.cuda) else package[0]
        if (len(package) == 2):
            label = package[1].cuda() if (args.cuda) else package[1]
        elif (len(package) == 3):
            tmpPC = package[1].cuda() if (args.cuda) else package[1]
            label = package[2].cuda() if (args.cuda) else package[2]
        elif (len(package) == 4):
            tmpPC = package[1].cuda() if (args.cuda) else package[1]
            negPC = package[2].cuda() if (args.cuda) else package[2]
            label = package[3].cuda() if (args.cuda) else package[3]
        
        # To be continue: triplet method inclusive
        if (args.sepModel):
            clsProbVec, globalFeat = net(srcPC)
            if (args.featLoss) : _, globalFeat2 = net(tmpPC)
            else : globalFeat2 = None
            loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, None, clsProbVec, label, args)
        else:
            clsProbVec, globalFeat, globalFeat2 = net(srcPC, tmpPC)
            loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, None, clsProbVec, label, args)
        
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
    for package in tqdm(trainLoader):
        # srcPC, label              : loaderType=cls
        # srcPC, tmpPC, label       : loaderType=glob2
        # srcPC, tmpPC, negPC, label: loaderType=triplet
        
        if (args.DP):
            for i in range(0, len(package) - 1):
                randPointsPerBatch = int(np.random.uniform(128, args.inputPoints))
                package[i] = package[i][:, :randPointsPerBatch, :]
        
        srcPC = package[0].cuda() if (args.cuda) else package[0]
        if (len(package) == 2):
            label = package[1].cuda() if (args.cuda) else package[1]
        elif (len(package) == 3):
            tmpPC = package[1].cuda() if (args.cuda) else package[1]
            label = package[2].cuda() if (args.cuda) else package[2]
        elif (len(package) == 4):
            tmpPC = package[1].cuda() if (args.cuda) else package[1]
            negPC = package[2].cuda() if (args.cuda) else package[2]
            label = package[3].cuda() if (args.cuda) else package[3]
        
        # To be continue: triplet method inclusive
        opt.zero_grad()
        if (args.sepModel):
            clsProbVec, globalFeat = net(srcPC)
            if (args.featLoss) : _, globalFeat2 = net(tmpPC)
            else : globalFeat2 = None
            # loss = F.nll_loss(clsProbVec, label.squeeze())
            # lossDict = {'clsLoss' : loss}
            loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, None, clsProbVec, label, args)
        else:
            clsProbVec, globalFeat, globalFeat2 = net(srcPC, tmpPC)
            loss, lossDict = ModelSelectorCriterion(globalFeat, globalFeat2, None, clsProbVec, label, args)
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
            SaveModel(net, 'best', args)
        textLog.writeLog('train\tepoch %d\tloss:%f\tbest epoch %d\tloss:%f'%(epoch, loss, bestTrainEpoch, bestTrainLoss))
        boardLog.add_scalar('train/loss', loss, epoch)
        boardLog.add_scalar('train/best_loss', bestTrainLoss, epoch)
        for key in lossDict : boardLog.add_scalar('train/%s' %key, lossDict[key], epoch)
        if (epoch % 10 == 0):
            SaveModel(net, epoch, args)
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


def SaveModel(net, ver, args):
    if (not args.modelName) : args.modelName = args.modelType
    saveModelName = os.path.join(args.saveModelDir, '{}_{}.pth'.format(args.modelName, ver))
    if (args.multiCuda):
        torch.save(net.module.state_dict(), saveModelName)
        if (args.sepModel):
            saveModelName = os.path.join(args.saveModelDir, '{}_feat_{}.pth'.format(args.modelName, ver))
            torch.save(net.features.module.state_dict(), saveModelName)
    else:
        torch.save(net.state_dict(), saveModelName)
        if (args.sepModel):
            saveModelName = os.path.join(args.saveModelDir, '{}_feat_{}.pth'.format(args.modelName, ver))
            torch.save(net.features.state_dict(), saveModelName)
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
            
            if (args.modelType in featModelList):
                srcFeat = net(srcPts)
                tmpFeat = net(catPts)
            elif (args.modelType in sepModelList):
                _, srcFeat = net(srcPts)
                _, tmpFeat = net(catPts)
            elif (args.modelType in intModelList):
                if (args.modelType == 'pointnet2Comp2'):
                    _, srcFeat = net(srcPts)
                    _, tmpFeat = net(catPts)
                else:
                    _, srcFeat, tmpFeat = net(srcPts, catPts)
            
            if (args.cuda): srcFeat = srcFeat.cpu()
            if (args.cuda): tmpFeat = tmpFeat.cpu()
            srcFeat = srcFeat.detach().numpy().squeeze()
            tmpFeat = tmpFeat.detach().numpy().squeeze()
            loss = np.mean(np.abs(srcFeat - tmpFeat))
            # print('%s %s: %f' %(srcModelU.path[-20:], catModelU.path[-20:], loss))
            rankList.append([catModelU.path, loss])
        rankList = sorted(rankList, key=itemgetter(1))
        rankPathList = np.array(rankList)[:,0]
        if (pathAns in rankPathList[:1]):
            totalRankDict['Rank 1'] += 1
        if (pathAns in rankPathList[:3]):
            totalRankDict['Rank 3'] += 1
        if (pathAns in rankPathList[:5]):
            totalRankDict['Rank 5'] += 1
        if (pathAns in rankPathList[:10]):
            totalRankDict['Rank 10'] += 1
        if (pathAns in rankPathList[:20]):
            totalRankDict['Rank 20'] += 1
        if (pathAns in rankPathList[:30]):
            totalRankDict['Rank 30'] += 1
        else:
            totalRankDict['Out of Rank'] += 1
        # print(srcModelU.path, rank)
    textLog.writeLog('#{}'.format(totalRankDict))
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
        if (args.modelType not in acceptModelList):
            raise 'modelType error.\n\tChoices:{}'.format(acceptModelList)
        if (args.modelType in featModelList and not args.eval):
            raise 'Model *Feat can only use in validation mode'
        if (args.modelType in featModelList and args.eval) : args.sepModel = True 
        if (args.modelType in sepModelList) : args.sepModel = True
        if (args.L1Loss or args.L2Loss or args.tripletMg) : args.featLoss = True
        
        args.loaderType = GetModelNet40H5ReturnType(args)
        
        textLog = textIO(args)
        textLog.writeLog(time.ctime())
        textLog.writeLog(args.__str__())
        return textLog, args
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
    if (args.multiCuda and torch.cuda.device_count() > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        args.multiCuda = False
    return device, args


def initListArgs(args):
    if (args.multiLR) : args.multiLR = args.multiLR if len(args.multiLR) > 0 else None
    if (args.specCat) : args.specCat = args.specCat if len(args.specCat) > 0 else None
    return args


if (__name__ == '__main__'):
    args = ModelSelectorParser(acceptModelList)
    args = initListArgs(args)
    device, args = initDevice(args)
    textLog, args = initEnv(args)
    
    randSeed = int(time.clock() * 10000)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(randSeed)
    torch.cuda.manual_seed_all(randSeed)
    np.random.seed(randSeed)
    
    if (args.modelType == 'pointnetComp') : net = PointNetComp(40, True)#           cls, feat1, feat2 = net(pc1, pc2)
    elif (args.modelType == 'pointnetFeat') : net = PointNetFeat(True, True)#       feat = net(pc)
    elif (args.modelType == 'pointnet') : net = PointNetCls(retGlobFeat = True)#    cls, feat = net(pc)
    elif (args.modelType == 'pointnet2Comp') : net = PointNet2Comp(3, 40)#          cls, feat1, feat2 = net(pc1, pc2)
    elif (args.modelType == 'pointnet2Feat') : net = PointNet2Feat(3)#              feat = net(pc)
    elif (args.modelType == 'pointnet2') : net = PointNet2Cls(retGlobFeat = True)#  cls, feat = net(pc)
    elif (args.modelType == 'dgcnnFeat') : net = DGCNNFeat(512, 20)#                feat = net(pc)
    elif (args.modelType == 'dgcnn') : net = DGCNN(retGlobFeat = True)#             cls, feat = net(pc)
    elif (args.modelType == 'pointnet2Comp2'):
        if (args.eval) : net = PointNet2Comp2(3, 40, 'cls')#                        cls, feat = net(pc)
        else : net = PointNet2Comp2(3, 40, args.loaderType)#                        cls, feat1, feat2 = net(pc1, pc2)
    
    if (args.multiCuda) : net = nn.DataParallel(net)
    net.to(device)
    
    if (not args.eval):
        boardLog = SummaryWriter(log_dir=args.saveModelDir)
        
        validLoader = DataLoader(ModelNet40H5(dataPartition='test', DIR_PATH=args.dataset, 
                                             srcPointNum=args.inputPoints, 
                                             tmpPointNum=args.inputPoints, 
                                             gaussianNoise=args.gaussianNoise, 
                                             scaling=args.scaling, 
                                             retType=args.loaderType), 
                                 batch_size=args.batchSize, shuffle=True)
        
        trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                              srcPointNum=args.inputPoints, 
                                              tmpPointNum=args.inputPoints, 
                                              gaussianNoise=args.gaussianNoise, 
                                              scaling=args.scaling, 
                                              retType=args.loaderType), 
                                 batch_size=args.batchSize, shuffle=True)
        
        train(net, trainLoader, validLoader, textLog, boardLog, args)
        
        boardLog.close()
    else:
        testLoader = ModelSelectorValidDataset(VALID_DIR=args.validDataset, specCatList = args.specCat if (args.specCat) else [])
        net.load_state_dict(torch.load(args.modelPath, map_location=device))
        CalBestTemplate(net, testLoader, args)
    textLog.close()