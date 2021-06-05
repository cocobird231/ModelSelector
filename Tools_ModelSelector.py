# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:24:07 2021

@author: cocob
"""
import os
from shutil import copyfile
import csv
import time
import open3d as o3d
import numpy as np

from Module_Utils import jitter_pointcloud, scaling_pointCloud, rotate_pointCloud
from Module_ModelNet40Series_DataLoader import WalkModelNet40CatDIR, WalkModelNet40ByCatName


def GenUnitSpherePCDFromMesh(modelPath : str, pointSize : int = 0):
    mesh = o3d.io.read_triangle_mesh(modelPath)
    if (mesh.is_empty()) : raise 'Empty geometry'
    maxBound = mesh.get_max_bound()
    minBound = mesh.get_min_bound()
    length = np.linalg.norm(maxBound - minBound, 2)
    mesh = mesh.scale(1 / length, center = mesh.get_center())
    mesh = mesh.translate(-mesh.get_center())
    return mesh.sample_points_uniformly(pointSize)


def FixMeshFileHeader(modelPath : str, backupF : bool = True, showInfo : bool = False):
    content = []
    with open(modelPath, 'r') as f:
        content = f.readlines()
    if (backupF) : copyfile(modelPath, modelPath + '_old')
    # Fix off file header with seperate first line into two lines, 
    # which contents OFF header and model info respectively.
    content[0] = content[0][3:]
    with open(modelPath, 'w') as f:
        f.write('OFF\n')
        for c in content:
            f.write(c)
    if (showInfo) : print('New file:', modelPath)
    return


def GenModelSelecotrValidDatasetFromModelNet40(DIR_PATH, SAVE_DIR_PATH, categories = 'all', numOfModelsPerCat = 50, pointSize = 1024):
    # Search model category
    catList = WalkModelNet40CatDIR(DIR_PATH)
    # Search model in each category
    catModelPathDict = dict()
    for cat in catList:
        np.random.seed(int(time.clock() * 10000))
        filePathList = WalkModelNet40ByCatName(DIR_PATH, cat, '.off')
        filePathList = np.random.permutation(filePathList)
        catModelPathDict[cat] = filePathList
    # Create point cloud from file
    catPCDsDict = dict()
    for catKey in catModelPathDict:
        pcdList = []
        modelCnt = 0
        print('Processing category:', catKey, end = ' ')
        for modelPath in catModelPathDict[catKey]:
            try:
                pcdList.append(GenUnitSpherePCDFromMesh(modelPath, pointSize))
            except BaseException:
                FixMeshFileHeader(modelPath)
                pcdList.append(GenUnitSpherePCDFromMesh(modelPath, pointSize))
            modelCnt += 1
            if (modelCnt >= numOfModelsPerCat) : break
        print('numbers:', len(pcdList))
        catPCDsDict[catKey] = pcdList
    # Save PCD into directory
    if (not os.path.exists(SAVE_DIR_PATH)) : os.mkdir(SAVE_DIR_PATH)
    for catKey in catPCDsDict:
        catdir = os.path.join(SAVE_DIR_PATH, catKey)
        if (not os.path.exists(catdir)) : os.mkdir(catdir)
        for i, pcd in enumerate(catPCDsDict[catKey]):
            o3d.io.write_point_cloud(os.path.join(catdir, '%s_%04d.pcd' %(catKey, i)), pcd)
    return


def GenSrcPCDFromValidDataset(SAVE_DIR_PATH, numOfModelsPerCat = 10, randDP = False):
    # Search model category
    catList = WalkModelNet40CatDIR(SAVE_DIR_PATH)
    if ('_src' in catList) : catList.remove('_src')
    srcdir = os.path.join(SAVE_DIR_PATH, '_src')
    if (not os.path.exists(srcdir)) : os.mkdir(srcdir)
    # Search model in each category
    catModelNameDict = dict()
    for cat in catList:
        np.random.seed(int(time.clock() * 10000))
        fileNameList = WalkModelNet40ByCatName(SAVE_DIR_PATH, cat, '.pcd', 'name')
        fileNameList = np.random.permutation(fileNameList)[:numOfModelsPerCat]
        catModelNameDict[cat] = fileNameList
    # Operating scaling, jittering and roatation on src PCD
    outputCatDict = dict()# {'cat1' : {'srcPath1' : 'dstPath1',..., 'srcPathN' : 'dstPathN'}}
    for catKey in catModelNameDict:
        srcPathAssociationDict = dict()# {'srcPath1' : 'dstPath1',..., 'srcPathN' : 'dstPathN'}
        for i, pcdName in enumerate(catModelNameDict[catKey]):
            pcd = o3d.io.read_point_cloud(os.path.join(SAVE_DIR_PATH, catKey, pcdName))
            pcdPts = np.asarray(pcd.points)
            if (randDP):
                randPointSize = int(np.random.uniform(128, 2048))
                pcdPts = pcdPts[:randPointSize, :]
            pcdPts = rotate_pointCloud(pcdPts)
            pcdPts = jitter_pointcloud(pcdPts)
            pcdPts = scaling_pointCloud(pcdPts)
            pcd_new = o3d.geometry.PointCloud()
            pcd_new.points = o3d.utility.Vector3dVector(pcdPts)
            # o3d.visualization.draw_geometries([pcd, pcd_new, DrawAxis(1)], window_name = 'Result')
            # print(pcdPts.shape)
            modelName = '%s_%04d.pcd' %(catKey, i)
            srcPathAssociationDict[modelName] = pcdName
            o3d.io.write_point_cloud(os.path.join(srcdir, modelName), pcd_new)
        outputCatDict[catKey] = srcPathAssociationDict
    with open(os.path.join(srcdir, '_Association.csv'), 'w', encoding = 'utf-8', newline = '') as f:
        csvWriter = csv.writer(f)
        for catKey in outputCatDict:
            for key in outputCatDict[catKey]:
                csvWriter.writerow([catKey, key, outputCatDict[catKey][key]])
    return


if (__name__ == '__main__'):
    # ModelSelectorValid
    # MODELNET40_DIR = 'D:/Datasets/ModelNet40'
    # VALID_DIR = 'D:/Datasets/ModelNet40_ModelSelector_VALID'
    # sT = time.clock()
    # GenModelSelecotrValidDatasetFromModelNet40(MODELNET40_DIR, VALID_DIR)
    # GenSrcPCDFromValidDataset(VALID_DIR)
    # print('Total use', time.clock() - sT, 'sec')
    
    # ModelSelectorValidDP
    MODELNET40_DIR = 'D:/Datasets/ModelNet40'
    VALID_DIR = 'D:/Datasets/ModelNet40_ModelSelector_VALID_DP'
    sT = time.clock()
    GenModelSelecotrValidDatasetFromModelNet40(MODELNET40_DIR, VALID_DIR, pointSize=2048)
    GenSrcPCDFromValidDataset(VALID_DIR, randDP=True)
    print('Total use', time.clock() - sT, 'sec')