# -*- coding: utf-8 -*-
"""
Created on Mon May 10 02:30:40 2021

@author: cocob
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def ModelSelectorCriterion(srcFeat, tmpFeat, negFeat, clsProbVec, label, args):
    lossDict = dict()
    clsLoss = F.nll_loss(clsProbVec, label.squeeze())
    lossDict['clsLoss'] = clsLoss
    if (args.L1Loss) : lossDict['l1Loss'] = F.l1_loss(srcFeat, tmpFeat) * args.weight
    if (args.L2Loss) : lossDict['l2Loss'] = F.mse_loss(srcFeat, tmpFeat) * args.weight
    # if (args.tripletMg):
    #     posLoss = F.mse_loss(srcFeat, tmpFeat)
    #     negLoss = F.mse_loss(srcFeat, negFeat)
    #     distLoss = posLoss - negLoss + args.tripletMg
    #     lossDict['tripletMg'] = distLoss if distLoss.item() > 0 else torch.tensor(0).to(srcFeat)
    if (args.tripletMg):
        triplet_loss = nn.TripletMarginLoss(margin=args.tripletMg, p=2)
        lossDict['tripletMg'] = triplet_loss(srcFeat, tmpFeat, negFeat) * args.weight
    loss = 0
    for key in lossDict:
        loss += lossDict[key]
    return loss, lossDict

def GetModelSelectorCriterionLossDict(args):
    lossDict = dict()
    lossDict['clsLoss'] = 0
    if (args.L1Loss) : lossDict['l1Loss'] = 0
    if (args.L2Loss) : lossDict['l2Loss'] = 0
    if (args.tripletMg) : lossDict['tripletMg'] = 0
    return lossDict