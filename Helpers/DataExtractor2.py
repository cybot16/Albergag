#!/usr/bin/env python

import os
import numpy as np
import dlib
import cv2
from Machines.Align_face import FaceAligner


def ImgProc(img, path, c):
    file_name = 'face'
    E = FaceAligner(img)
    faceAligned = E.AlignFace(31)
    cv2.imwrite(path+'/'+file_name+str(c)+".jpg", faceAligned)

def PrepEnv(path):
    os.chdir('../../')
    rootpath = os.getcwd()
    ProcPath = r"ProcessedData"
    if not os.path.exists(ProcPath):
        os.makedirs(ProcPath)
    for (dirpath, dirnames, filenames) in os.walk(path):
        print '[INFO] Accessing the provided folder.'
        for i in dirnames:
            print '[INFO] Accessing folder: '+i
            os.chdir(os.getcwd()+'/'+ProcPath)
            if not os.path.exists(i):
                os.makedirs(i)
            os.chdir(rootpath)
            os.chdir(path)
            c=0
            for (dirpath1, dirnames1, filenames1) in os.walk(i):
                os.chdir(rootpath)
                for j in filenames1:
                    img = cv2.imread(path+'/'+dirpath1+'/'+j)
                    ImgProc(img,ProcPath+'/'+i,c)
                    c+=1
                break
        break

__name__ = '__main__'
__all__ = ["ImgProc","PrepEnv"]