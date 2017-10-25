#!/usr/bin/env python

from Align_face import FaceAligner
import os
import numpy as np
import dlib
import cv2
import pickle

def ImgProc(img, path, c):
    file_name = 'face'
    E = FaceAligner(img,flag=1)
    faceAligned = E.AlignFace(31)
    if type(faceAligned) == int:
        return
    cv2.imwrite(path+'/'+file_name+str(c)+".jpg", faceAligned)

def pe(path):
    os.chdir('../')
    rootpath = os.getcwd()
    print rootpath
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

def Loader(file):
    # os.chdir('../')
    with open(file,'rb') as f:
        return pickle.load(f)

def Dumper(path, obj):
    # os.chdir('../')
    with open(path,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# __name__ = '__main__'
__all__ = ["ImgProc", "pe", "Loader", "Dumper"]
