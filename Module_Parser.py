# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 01:14:23 2021

@author: cocob
"""

import argparse

def ModelSelectorParser(acceptModelList):
    parser = argparse.ArgumentParser(description='ModelSelector')
    # Required arguments
    parser.add_argument('-d', '--dataset',      required=False, type=str, metavar='PATH', 
                        default='D:\\Datasets\\modelnet40_ply_hdf5_2048', help='ModelNet40_HDF5 dataset path')
    parser.add_argument('-b', '--batchSize',    required=False, type=int, metavar='N', 
                        default=8, help='Set batch size')
    parser.add_argument('-e', '--epochs',       required=False, type=int, metavar='N', 
                        default=200, help='Set epoch size')
    parser.add_argument('-m', '--modelPath',    required=False, type=str, metavar='PATH', 
                        default='models/model_ModelSelector_best_pointnet_scaling_noL1.pth', help='Pre-trained model path for ModelSelector')# Windows test
    parser.add_argument('-t', '--modelType',    required=False, type=str, metavar='N', 
                        default='pointnet', choices=acceptModelList, help='Feature extractor')# Windows test
    # Training arguments
    parser.add_argument('--eval', action='store_true', 
                        default=False, help='Run evaluation mode')# Windows test
    parser.add_argument('--startEpoch', type=int, 
                        default=0, help='Start epoch')
    parser.add_argument('--multiLR', type=int, nargs='+', 
                        help='Use multi learning rate with input steps')
    # Device settings
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')# Windows test
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device, ignore while --multiCuda flag is true')
    parser.add_argument('--multiCuda', action='store_true', 
                        default=False, help='Using multiple cuda device, ignore while --cuda flag is false')
    # Training settings
    parser.add_argument('--inputPoints', type=int, 
                        default=2048, help='Input points (max: 2048)')
    parser.add_argument('--gaussianNoise', action='store_true', 
                        default=False, help='Add Gaussian noise into dataset during training')
    parser.add_argument('--trainView', action='store_true', 
                        default=False, help='Training via viewed point cloud')
    parser.add_argument('--scaling', action='store_true', 
                        default=False, help='Training via scaled point cloud')
    parser.add_argument('--DP', action='store_true', 
                        default=False, help='Random dropout point cloud during training. When true, ignore batchSize and inputPoints.')
    parser.add_argument('--modelName', type=str, 
                        default=None, help='Save model file name: {NAME}.pth (default: featModel)')
    parser.add_argument('--saveModelDir', type=str, 
                        default='models', help='Path for model saving')
    parser.add_argument('--logName', type=str, 
                        default='log_ModelSelector.txt', help='Log file name')
    parser.add_argument('--saveLogDir', type=str, 
                        default='models', help='Path for log saving')
    parser.add_argument('--sepModel', action='store_true', help='Program implement only.')
    # Loss options
    parser.add_argument('--L1Loss', action='store_true', 
                        default=False, help='Using L1 loss')
    parser.add_argument('--L2Loss', action='store_true', 
                        default=False, help='Using L2 loss')
    parser.add_argument('--tripletMg', type=float, 
                        default=None, help='Using MSE triplet loss with given margin')
    parser.add_argument('--featLoss', action='store_true', help='Program implement only.')
    parser.add_argument('--loaderType', type=str, help='Program implement only.')
    
    # Validation options
    parser.add_argument('--validDataset', type=str, 
                        default='D:/Datasets/ModelNet40_ModelSelector_VALID', help='ModelNet40_ModelSelector_VALID dataset path')
    parser.add_argument('--specCat', type=str, nargs='+', help='Validating on specific categories')
    
    return parser.parse_args()

def PointNetLKParser():
    parser = argparse.ArgumentParser(description='PointNetLK')

    # required.
    parser.add_argument('-d', '--dataset', default='D:\\Datasets\\ModelNet40_VALID_1024_2', required=False, type=str, 
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('-c', '--clsModelPath', default='models/model_PointNetLK_classifier_feat_best.pth_oldSaveVer', required=False, type=str, 
                        metavar='PATH', help='path to trained model file (default: null (no-use))')
    parser.add_argument('-m', '--lkModelPath', default='models/model_PointNetLK_best.pth', required=False, type=str,
                        metavar='PATH', help='path to classifier feature (default: null (no-use))')

    # settings for PointNet-LK
    parser.add_argument('--max_iter', default=20, type=int, 
                        metavar='N', help='max-iter on LK. (default: 20)')
    parser.add_argument('--dim_k', default=1024, type=int, 
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'], 
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float, 
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')

    # settings for on testing
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')# Windows test
    parser.add_argument('--cudaDevice', default='cuda:0', type=str, 
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    parser.add_argument('--logName', type=str, 
                        default='log_PointNetLK.txt', help='Log file name')
    parser.add_argument('--saveLogDir', type=str, 
                        default='log', help='Path for log saving')
    return parser.parse_args()

def PointNet2Parser():
    parser = argparse.ArgumentParser(description='PointNet2')
    # Required arguments
    parser.add_argument('-d', '--dataset',      required=False, type=str, metavar='PATH', 
                        default='D:\\Datasets\\modelnet40_ply_hdf5_2048', help='ModelNet40_HDF5 dataset path')
    parser.add_argument('-b', '--batchSize',    required=False, type=int, metavar='N', 
                        default=8, help='Set batch size')
    parser.add_argument('-e', '--epochs',       required=False, type=int, metavar='N', 
                        default=200, help='Set epoch size')
    # Device settings
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')# Windows test
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device, ignore while --multiCuda flag is true')
    # Training settings
    parser.add_argument('--inputPoints', type=int, 
                        default=2048, help='Input points (max: 2048)')
    parser.add_argument('--gaussianNoise', action='store_true', 
                        default=False, help='Add Gaussian noise into dataset during training')
    parser.add_argument('--trainView', action='store_true', 
                        default=False, help='Training via viewed point cloud')
    parser.add_argument('--scaling', action='store_true', 
                        default=False, help='Training via scaled point cloud')
    parser.add_argument('--RD', type=float, 
                        default=None, help='Random dropout point cloud during training')
    parser.add_argument('--saveModelDir', type=str, 
                        default='result', help='Path for model saving')
    parser.add_argument('--logName', type=str, 
                        default='log_pointnet2.txt', help='Log file name')
    parser.add_argument('--saveLogDir', type=str, 
                        default='result', help='Path for log saving')
    return parser.parse_args()