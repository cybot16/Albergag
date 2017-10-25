#!/usr/bin/env python

from Machines.DataRepresentation import *
from Machines.DataExtractor import *
from Machines.deepID import *
import cv2
import numpy as np
import os


root_path = ''

class Brain():
    def __init__(self):
        print "[+] My brain is starting up, one second and we'll have fun together ;)!"
        return
    def DataExt(self, path='Data/RAWData'):
        pe(path)
    def Train(self):
        X_train, Y_train = PrepEnv('ProcessedData')
        Training()
    def Solve(self, Image):
        img = DataRep(mode=0,img=Image)
        img = np.asarray(img, dtype=np.float32)
        c = Solver(img)
        print c
        Names = Loader('../Data/People/people.pkl')
        print Names
        print "The person in the picture is "+Names[c]
    def Correct(self, Image):
        img = DataRep(mode=0, img=Image)
        img = np.asarray(img, dtype=np.float32)
        Correction(img)

def DataRep(path1='ProcessedData', path2='Test', mode=1, img=None, *args, **kargs):
    if mode:
        X, Y = PrepEnv(path1)
        os.chdir(root_path)
        # X_test, Y_test = PrepEnv(path2,1)
        # os.chdir(root_path)
        TrainingData(path1, X, Y)
        # os.chdir(root_path)
        # TestingData(path2, X_test, Y_test)
    else:
        return PrepOne(img)

def video():
    print "[INFO] Hello, when you are ready press 'c' and let the magic happen."
    Video = cv2.VideoCapture(0)
    while 1:
        ret, frame = Video.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            return frame


def main():
    global root_path
    root_path = os.getcwd()
    print "[INFO] Welcome to our facial recognition program or if you don't have\n       a trained model press 'y' or else press 'n'."
    B = Brain()
    inp = 'f'
    while(inp != 'y' and inp != 'n'):
        inp = raw_input("[INFO] if it's the first time you run the program\n--->")
    if(inp == 'y'):
        print '[INFO] Please provide the directory in which you have your raw Data.'
        print '       Which means random pictures, not necessarily cropped face pictures.'
        print '       Your directory should have for each person a dedicated folder.'
        pathtoext = raw_input("--->")
        B.DataExt(pathtoext)
        os.chdir(root_path)
        DataRep()
        os.chdir(root_path)
        PrepEnv('ProcessedData')
        os.chdir(root_path)
    while(inp != 'y' and inp != 'n'):
        inp = raw_input("[INFO] Do you want to train your model some more? ('y' or 'n')\n--->")
    if inp == 'y':
        B.Train()
    while 1:
        if 'y' in raw_input():
            Img = video()
        else:
            print "You are on "+os.getcwd()
            name = raw_input('-->')
            # os.chdir('../Data/Prediction/'+name)
            # for (dirpath, dirnames, filenames) in os.walk('.'):
            #     for i in filenames:
            Img = cv2.imread(name)
        cv2.destroyAllWindows()
        os.chdir(root_path)
        B.Solve(Img)
        answer = raw_input('Is the answer correct?\n-->')
        if 'n' in answer:
            B.Correct(Img)
    return

if __name__ == '__main__':
    main()