# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:19:15 2021

@author: User
"""

import os
import csv
import glob
import h5py
import time
import numpy as np
import open3d as o3d

from torch.utils.data import Dataset

from Module_Utils import Rigid, jitter_pointcloud, scaling_pointCloud, WalkModelNet40ByCatName, ModelUtils


#############################################################
#                       Training Dataset
#############################################################
class ModelNet40H5(Dataset):
    def __init__(self, DIR_PATH : str, dataPartition = 'None', 
                 tmpPointNum = 1024, srcPointNum = 1024, 
                 gaussianNoise = True, randView = False, scaling = False, 
                 angleRange = 90, translationRange = 0.5, scalingRange = 0.2):
        
        self.data, self.label = self.load_data(DIR_PATH, dataPartition)
        self.tmpPointNum = tmpPointNum
        self.srcPointNum = srcPointNum
        self.gaussianNoise = gaussianNoise
        self.randView = randView
        self.scaling = scaling
        self.angleRange = angleRange
        self.translationRange = translationRange
        self.scalingRange = scalingRange
                
    def load_data(self, DIR_PATH, dataPartition):
        all_data = []
        all_label = []
        dataNamePattern = '/ply_data*.h5'
        if (dataPartition != 'None'):
            dataNamePattern = ('/ply_data_%s*.h5' %dataPartition)
        print(dataNamePattern)
        for h5_name in glob.glob(DIR_PATH + dataNamePattern):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        rigidAB = Rigid()
        rigidAB.getRandomRigid(self.angleRange, self.translationRange)
        
        pc = self.data[item]
        pc1 = np.random.permutation(pc)[:self.tmpPointNum]
        pc2 = np.random.permutation(pc)[:self.srcPointNum]
        pc2 = ((rigidAB.rotation @ pc2.T).T + rigidAB.translation)
        if (self.gaussianNoise):
            pc1 = jitter_pointcloud(pc1)
            pc2 = jitter_pointcloud(pc2)
        if (self.scaling):
            pc1 = scaling_pointCloud(pc1)
            pc2 = scaling_pointCloud(pc2)
        # Output pc1, pc2: N x 3
        return pc1.astype('float32'), pc2.astype('float32'), self.label[item]



#############################################################
#                   Random Testing Dataset (to be optimize)
#############################################################
class GetAllModelFromCategory():
    def __init__(self, DIR_PATH : str, category : str, numOfPoints : int):
        assert DIR_PATH and category, 'Must input dataset path and category name'
        self.filePathList = self.GetModelPathByCategory(DIR_PATH, category)
        self.points = numOfPoints
        
    
    def GetModelPathByCategory(self, DIR_PATH : str, category : str):
        SEARCH_DIR = os.path.join(DIR_PATH, category)
        filePathList = []
        for dirpath, dirname, filename in os.walk(SEARCH_DIR):
            for offFile in filename:
                if '.off' in offFile:
                    filePathList.append(os.path.join(dirpath, offFile))
        return filePathList
    
    def GetRandomTestSet(self, numOfCandidate, srcRotateF = False):
        np.random.seed(int(time.clock()))
        candidatePathList = np.random.permutation(self.filePathList)[:numOfCandidate]
        srcPath = candidatePathList[0]
        candidatePathList = np.random.permutation(candidatePathList)
        
        rigid = Rigid()
        rigid.getRandomRigid()
        
        srcMesh = o3d.io.read_triangle_mesh(srcPath)
        maxBound = srcMesh.get_max_bound()
        minBound = srcMesh.get_min_bound()
        length = np.linalg.norm(maxBound - minBound, 2)
        srcMesh = srcMesh.translate(-srcMesh.get_center())
        srcMesh = srcMesh.scale(1/length, center=srcMesh.get_center())
        if (srcRotateF) : srcMesh = srcMesh.rotate(rigid.rotation, center=srcMesh.get_center())
        
        srcPCD = jitter_pointcloud(np.asarray(srcMesh.sample_points_uniformly(self.points).points)).astype('float32')
        candidatePCDList = []
        for path in candidatePathList:
            tmpMesh = o3d.io.read_triangle_mesh(path)
            maxBound = tmpMesh.get_max_bound()
            minBound = tmpMesh.get_min_bound()
            length = np.linalg.norm(maxBound - minBound, 2)
            tmpMesh = tmpMesh.translate(-tmpMesh.get_center())
            tmpMesh = tmpMesh.scale(1/length, center=tmpMesh.get_center())
            candidatePCDList.append(jitter_pointcloud(np.asarray(tmpMesh.sample_points_uniformly(self.points).points)).astype('float32'))
        
        return srcPCD, candidatePCDList, srcPath, candidatePathList



#############################################################
#               ModelSelector Validation Dataset
#############################################################
class ModelSelectorValidDataset():
    def __init__(self, VALID_DIR : str):# -> VALID_DIR: Directory path for ModelNet40_ModelSelector_VALID
        self.srcPCDList, self.srcPathList, self.ansPathList, self.catList, self.catPCDsDict = self.getModelsFromModelSelectorVALIDDataset(VALID_DIR)
        assert (len(self.srcPCDList) == len(self.srcPathList) == len(self.ansPathList) == len(self.catList)), 'Data length error'
        self.size = len(self.srcPCDList)
    
    def getModelsFromModelSelectorVALIDDataset(self, VALID_DIR : str):
        srcdir = os.path.join(VALID_DIR, '_src')
        catList = []
        srcPCDList = []# Stored each testing srcPCD
        catPCDsDict = dict()# Each category's PCD: {'cat' : [ModelUtils(PCD, cat, path),...,ModelUtils(PCD, cat, path)]}
        srcPathList = []# Path for srcPCD in /_src directory
        ansPathList = []# Path for srcPCD in /'cat' directory
        with open(os.path.join(srcdir, '_Association.csv'), 'r', encoding = 'utf-8', newline = '') as f:
            csvReader = csv.reader(f)
            for row in csvReader:
                catList.append(row[0])
                srcPathList.append(row[1])
                ansPathList.append(row[2])
                srcPCDList.append(ModelUtils(np.asarray(o3d.io.read_point_cloud(row[1]).points), row[0], row[1]))
        catSet = set(catList)
        for cat in catSet:
            filePathList = WalkModelNet40ByCatName(VALID_DIR, cat, '.pcd')
            modelList = []
            for file in filePathList:
                modelList.append(ModelUtils(np.asarray(o3d.io.read_point_cloud(file).points), cat, file))
            catPCDsDict[cat] = modelList
        return srcPCDList, srcPathList, ansPathList, catList, catPCDsDict
    
    def getModelListByCat(self, category : str):
        return self.catPCDsDict[category]
    
    def getAllCatModelDict(self):
        return self.catPCDsDict
    
    def __iter__(self):
        self.cnt = 0
        return self
    
    def __next__(self):
        if (self.cnt < self.size):
            _temp = self.cnt
            self.cnt += 1
            return self.srcPCDList[_temp], self.getModelListByCat(self.srcPCDList[_temp].label), self.ansPathList[_temp]
        else:
            raise StopIteration


if __name__ == '__main__':

    import sys
    loader = ModelSelectorValidDataset('D:/Datasets/ModelNet40_ModelSelector_VALID')
    cnt = 1
    for srcModelU, catModelUList, pathAns in loader:
        print(srcModelU)
        print(catModelUList)
        print(pathAns)
        cnt += 1
        if (cnt > 5) : break
    sys.exit(0)
    from Module_Parser import ModelSelectorParser
    from torch.utils.data import DataLoader
    args = ModelSelectorParser()
    loader = GetAllModelFromCategory(DIR_PATH='D:\\Datasets\\ModelNet40', category='chair', numOfPoints=args.inputPoints)
    print(loader.GetRandomTestSet(3))
    
    trainLoader = DataLoader(ModelNet40H5(dataPartition='train', DIR_PATH=args.dataset, 
                                          srcPointNum=args.inputPoints, 
                                          tmpPointNum=args.inputPoints, 
                                          gaussianNoise=args.gaussianNoise, 
                                          scaling=args.scaling), 
                            batch_size=1, shuffle=True)
    cnt = 1
    for pc1, pc2, label in trainLoader:
        pc1 = pc1.numpy().squeeze()
        pc2 = pc2.numpy().squeeze()
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1)
        pcd2.points = o3d.utility.Vector3dVector(pc2)
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name = 'Result')
        cnt += 1
        if (cnt > 5) : break