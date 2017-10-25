#!/usr/bin/env python

from threading import Thread
from threading import Semaphore
from Machines.DataRepresentation import *
from Machines.DataExtractor import *
from Machines.deepID import *
from Machines.Align_face import FaceAligner
import cv2
import numpy as np
import os

s = Semaphore(15)

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
        if type(img) == int:
            # print 'No face was detected'
            return
        img = np.asarray(img, dtype=np.float32)
        c = Solver(img)
        print c
        Names = Loader('../Data/People/people.pkl')
        print Names
        return Names[c]
    def Correct(self, Image):
        img = DataRep(mode=0, img=Image)
        img = np.asarray(img, dtype=np.float32)
        Correction(img)

def video(Video):
    print "[INFO] Hello, when you are ready press 'c' and let the magic happen."
    while 1:
        ret, frame = Video.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            return frame

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

def Show(frame):
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

def fadet(Video):
    while 1:
        s.acquire()
        ret, frame = Video.read()
        E = FaceAligner(frame,flag=1)
        face = E.DetectFace()
        Show(frame)
        s.release()

def Realtime(B, frame):
    s.acquire()
    ret, frame = Video.read()
    cv2.putText(frame,B.Solve(frame), (0,0), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    Show(frame)
    s.release()

def main():
    global root_path
    Video = cv2.VideoCapture(0)
    root_path = os.getcwd()
    print '[INFO] Welcome to our facial recognition program.'
    B = Brain()
    # B.DataExt()
    # os.chdir(root_path)
    # DataRep()
    # os.chdir(root_path)
    # PrepEnv('ProcessedData')
    # os.chdir(root_path)
    # B.Train()
    while 1:
        inpu = raw_input()
        if 'y' in inpu:
            Img = video(Video)
        elif 'r' in inpu:
            t2 = Thread(target=Realtime, args=(B, Video,))
            t1 = Thread(target=fadet, args=(Video,))
            t1.start()
            t2.start()
            os.chdir(root_path)
            t1.run()
            t2.run()
        else:
            print "You are on "+os.getcwd()
            name = raw_input('-->')
            # os.chdir('../Data/Prediction/'+name)
            # for (dirpath, dirnames, filenames) in os.walk('.'):
            #     for i in filenames:
            Img = cv2.imread(name)
        cv2.destroyAllWindows()
        os.chdir(root_path)
        print B.Solve(Img)
        answer = raw_input('Is the answer correct?\n-->')
        if 'n' in answer:
            B.Correct(Img)
    return

if __name__ == '__main__':
    main()