#! /usr/bin/env python

import cv2
import os

def takepicture(name, path):
    os.chdir('../'+path)
    video = cv2.VideoCapture(0)
    if not os.path.exists(name):
        os.makedirs(name)
    c = 0
    os.chdir(name)
    print 'You are on '+os.getcwd()
    while 1:
        ret, frame = video.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(2) & 0xFF == ord('c'):
            print '[+] Picture number '+str(c+1)+' has been added to your directory.'
            cv2.imwrite('imagse'+str(c)+'.jpg',frame)
            c+=1
        if cv2.waitKey(2) & 0xFF == ord('q'):
            print '[-] Quiting the program, thank you for using it.'
            break

def main():
    name=raw_input('[+] Welcome to the picture taker program, \
    please provide us with your name.\n-->')
    print "[+] Thank you very much "+name+", now a window will pop-up\
     and you will see your face in it, don't worry, you are not being hacked"
    print "[INFO] To take a picture press 'c', to quit press 'q'"
    takepicture(name, 'Data/RAWData/')

if __name__ == '__main__':
    main()
