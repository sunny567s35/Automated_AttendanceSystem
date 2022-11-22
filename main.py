import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

# from PIL import ImageGrab

path = 'Training_images'
images = []
KnownNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    KnownNames.append(os.path.splitext(cl)[0])
print(KnownNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList



#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)

#save encoding along with their names in dictionary data
data = {"encodings":encodeListKnown , "names":KnownNames}
print('Encoding Complete')

#use picle to save daa in to a file for later use
f = open("face_encodings","wb")
f.write(pickle.dumps(data))
f.close()
