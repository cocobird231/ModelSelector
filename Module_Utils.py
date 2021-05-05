# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 22:13:08 2021

@author: User
"""

import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

DEG2RAD = 3.1415926 / 180.0

def ICPIter(templatePC, targetPC, initTransform, iterSize = 50, iterStep = 0.2):
    ICP_TRANSFORM = o3d.pipelines.registration.registration_icp(
        templatePC, targetPC, iterStep, initTransform, 
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iterSize))
    return ICP_TRANSFORM

def ICPMotion(ICP_PC, tempPC, defaultTrans, iterSize, iterStep):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window('Viewer', 600, 600)
    
    ctr = viewer.get_view_control()
    ctr.set_zoom(1)
    ctr.set_front((1, 0, 0))
    ctr.set_up((0, 0, 1))
    ctr.set_lookat((0, 0, 0))
    
    viewer.add_geometry(ICP_PC)
    viewer.add_geometry(tempPC)
    for i in range(iterSize):
        ICP_TRANSFORM = ICPIter(ICP_PC, tempPC, defaultTrans, 1, iterStep)
        ICP_PC.transform(ICP_TRANSFORM.transformation)
        viewer.update_geometry(ICP_PC)
        viewer.poll_events()
        viewer.update_renderer()
    # viewer.run()
    viewer.remove_geometry(ICP_PC)
    viewer.remove_geometry(tempPC)

def DrawAxis(length = 10):
    points = [[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]]
    lines = [[0, 1], [0, 2], [0, 3]]# x, y, z
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]# r, g, b
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def GetZAxisRotateMat(ang : float):# The ang must be an angle value, follow permutation x-y-z
    return R.from_euler('xyz', [0, 0, ang], degrees = True).as_matrix()

class Rigid():
    def __init__(self, rotation = 0, translation = 0, eulerAng = []):
        self.rotation = rotation
        self.translation = translation
        self.eulerAng = eulerAng
    
    def getRandomRigid(self, angleRange = 90, translationRange = 0.5):
        anglex = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        angley = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        anglez = np.random.uniform(-angleRange, angleRange) * DEG2RAD
        self.eulerAng = np.asarray([anglez, angley, anglex]).astype('float32')
        self.rotation = R.from_euler('zyx', self.eulerAng).as_matrix().astype('float32')
        self.translation = np.array([np.random.uniform(-translationRange, translationRange), 
                                     np.random.uniform(-translationRange, translationRange), 
                                     np.random.uniform(-translationRange, translationRange)]).astype('float32')


    def getInvRigid(self):
        rotation_inv = self.rotation.T
        translation_inv = -rotation_inv.dot(self.translation)
        eulerAng_inv = -self.eulerAng[::-1]
        return Rigid(rotation_inv, translation_inv, eulerAng_inv)

class UnitModelUtils:
    def __init__(self, modelPath = '', translation = np.zeros(3), rotation = np.eye(3), scale = 1.0, label = 'None'):
        self.translation = translation
        self.rotation = rotation
        self.scale = scale
        self.modelPath = modelPath
        self.label = label

class textIO:
    def __init__(self, args):
        self.f = open(os.path.join(args.saveModelDir, args.logName), 'a')
    
    def writeLog(self, string, printF = True):
        print(string)
        self.f.write(string + '\n')
        self.f.flush()
    
    def close(self, nextLineF = True):
        if (nextLineF):
            self.f.write('\n')
        self.f.close()
    
    def __del__(self):
        self.f.close()
        print('textIO destructor called')