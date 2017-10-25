import os
import numpy as np
import cv2
from Align_face import FaceAligner
from DataExtractor import *

__all__ = ["PrepEnv","TrainingData","TestingData","LoadData","PrepOne"]

# def get_bin(x, n):
#     return format(x, 'b').zfill(n)

# d = 0
# X_train = np.empty((1))
# Y_train = np.empty((1))
# X_test = np.empty((1))
# Y_test = np.empty((1))

# def create_list(s):
#     global d
#     l = []
#     l += [0] * d
#     l[s.index('1')] = 1
#     return l

def PrepEnv(path, flag=0):
    d = 0
    Label_map = {}
    # global X_train
    # global Y_train
    # global X_test
    # global Y_test
    if not flag:
        c = 0
        os.chdir('../')
        rootpath = os.getcwd()
        for (dirpath, dirnames, filenames) in os.walk(path):
            os.chdir(path)
            path_v = os.getcwd()
            for i in dirnames:
                for (dirpath1, dirnames1, filenames1) in os.walk(i):
                    if filenames1:
                        d += 1
                        Label_map[d] = i
                        for j in filenames1:
                            c+=1
                os.chdir(path_v)
        X_train = np.empty((c,31,31),dtype=np.float32)
        Y_train = np.empty((c))
        print Label_map
        Dumper('../Data/People/people.pkl',Label_map)
        return X_train, Y_train
    else:
        os.chdir('../')
        c = 0
        for (dirpath, dirnames, filenames) in os.walk(path):
            os.chdir(path)
            path_v = os.getcwd()
            for i in dirnames:
                for (dirpath1, dirnames1, filenames1) in os.walk(i):
                    if filenames1:
                        for j in filenames1:
                            c+=1
                os.chdir(path_v)
        X_test = np.empty((c,31,31),dtype=np.float32)
        Y_test = np.empty((c))
        return X_test, Y_test

def TrainingData(path,X_train,Y_train):
    os.chdir('../')
    root_path = os.getcwd()
    c = 0
    tot = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        print '[INFO] Accessing the provided folder.'
        print '[INFO] Preparing Data, please wait.'
        os.chdir(path)
        path_v = os.getcwd()
        for i in dirnames:
            print '[INFO]  person: '+i
            for (dirpath1, dirnames1, filenames1) in os.walk(i):
                if filenames1:
                    c += 1
                    for j in filenames1:
                        npar = cv2.imread(os.getcwd()+'/'+i+'/'+j)
                        npar = cv2.cvtColor(npar,cv2.COLOR_BGR2GRAY)
                        X_train[tot] = npar
                        Y_train[tot] = c
                        tot += 1
            os.chdir(path_v)
    os.chdir(root_path)
    np.save('Data/CSV/X_train',X_train)
    np.save('Data/CSV/Y_train',Y_train)

def PrepOne(img):
    E = FaceAligner(img)
    Face = E.AlignFace(31)
    if type(Face) == int:
        return 0
    return cv2.cvtColor(Face,cv2.COLOR_BGR2GRAY)

def TestingData(path,X_test,Y_test):
    os.chdir('../')
    root_path = os.getcwd()
    c = 0
    tot = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        print '[INFO] Accessing the provided folder.'
        print '[INFO] Preparing Data, please wait.'
        os.chdir(path)
        path_v = os.getcwd()
        for i in dirnames:
            print '[INFO]  person: '+i
            for (dirpath1, dirnames1, filenames1) in os.walk(i):
                if filenames1:
                    c += 1
                    for j in filenames1:
                        npar = cv2.imread(os.getcwd()+'/'+i+'/'+j)
                        npar = cv2.cvtColor(npar,cv2.COLOR_BGR2GRAY)
                        X_test[tot] = npar
                        Y_test[tot] = c
                        tot += 1
            os.chdir(path_v)
    os.chdir(root_path)
    np.save('Data/CSV/X_test',X_test)
    np.save('Data/CSV/Y_test',Y_test)

def LoadData(path='../Data/CSV/'):
    X = np.load(path+'X_train.npy')
    Y = np.load(path+'Y_train.npy')
    # X_test_load = np.load(path+'X_test.npy')
    # Y_test_load = np.load(path+'Y_test.npy')
    return X, Y