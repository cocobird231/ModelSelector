# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 01:14:23 2021

@author: cocob
"""

import argparse

def ModelSelectorParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',      required=False, type=str, metavar='PATH', 
                        default='D:\\Datasets\\modelnet40_ply_hdf5_2048', help='ModelNet40_HDF5 dataset path')
    parser.add_argument('-b', '--batchSize',    required=False, type=int, metavar='N', 
                        default=24, help='Set batch size')
    parser.add_argument('-e', '--epochs',       required=False, type=int, metavar='N', 
                        default=200, help='Set epoch size')
    parser.add_argument('-m', '--modelPath',    required=False, type=str, metavar='PATH', 
                        default='models/#model_ModelSelector_best.pth', help='Pre-trained model path for ModelSelector')
    parser.add_argument('-f', '--featModel',    required=False, type=str, metavar='N', 
                        default='pointnet2', choices=['pointnet', 'pointnet2'], help='Feature extractor')
    
    parser.add_argument('--eval', action='store_true', 
                        default=False, help='Run evaluation mode')
    parser.add_argument('--startEpoch', type=int, 
                        default=0, help='Start epoch')
    parser.add_argument('--cuda', action='store_true', 
                        default=False, help='Training via cuda device, ignore while cuda device not found')
    parser.add_argument('--cudaDevice', type=str, 
                        default='cuda:0', help='Select cuda device, ignore while --multiCuda flag is true')
    parser.add_argument('--multiCuda', action='store_true', 
                        default=False, help='Using multiple cuda device, ignore while --cuda flag is false')
    parser.add_argument('--inputPoints', type=int, 
                        default=1024, help='Input points')
    parser.add_argument('--gaussianNoise', action='store_true', 
                        default=False, help='Add Gaussian noise into dataset during training')
    parser.add_argument('--trainView', action='store_true', 
                        default=False, help='Training via viewed point cloud')
    parser.add_argument('--scaling', action='store_true', 
                        default=False, help='Training via scaled point cloud')
    parser.add_argument('--saveModelDir', type=str, 
                        default='models', help='Path for model saving')
    parser.add_argument('--logName', type=str, 
                        default='log_ModelSelector.txt', help='Log file name')
    parser.add_argument('--L1Loss', action='store_true', 
                        default=False, help='Using L1 loss')
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()