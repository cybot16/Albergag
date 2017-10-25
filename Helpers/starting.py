# Facial Detection System
# Import required Modules, NumPy and OpenCV

import numpy as np
import cv2

# Define the main function

def main():

# Set file for Haar Cascades Classifier on face detection

    face_cascade = cv2.CascadeClassifier('~/Documents/Studies/Image Processing/\
    Facial Recognition/Data/HaarDB/HaarCascade_FrontalFace_Default.xml')
    Video = cv2.VideoCapture(0)

    while 1:
        ret, frame = Video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    Video.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
