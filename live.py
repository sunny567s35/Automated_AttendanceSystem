import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pickle

#---------------------------------------------------------------------------------------------
# get default video FPS
# fps = cap.get(cv2.CAP_PROP_FPS)
 
# # get total number of video framesqq
# num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# print(dlib.DLIB_USE_CUDA)
#---------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(1)

data = pickle.loads(open('face_encodings',"rb").read())
nameList = []
print(nameList)
#-----------------------------------------------------------added FPS counter
print("FPS : {:0.3}".format(cap.get(cv2.CAP_PROP_FPS)))
#=============================================================
def markAttendance(name):
    with open('Attendance.csv', 'a') as f:
        
        if name not in nameList :
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            nameList.append(name)
        


while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    #------------------------------added number of upsamples
    facesCurFrame = face_recognition.face_locations(imgS,number_of_times_to_upsample=2)
    #------------------------------added number of jitters
    encodesCurFrame = face_recognition.face_encodings(imgS,known_face_locations= facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        mat = False
        matches = face_recognition.compare_faces(data['encodings'], encodeFace)
        faceDis = face_recognition.face_distance(data['encodings'], encodeFace)
        # print(faceDis)
        # print(matches)
        matchIndex = np.argmin(faceDis)
        
        if(faceDis[matchIndex] <= 0.47):
            
            mat = True

        if mat == True:
            name = data['names'][matchIndex].lower()
            
        elif mat ==False:
            name = "Buchodu!"
            
# print(name)
        y1, x2, y2, x = faceLoc
        y1, x2, y2, x = y1 * 4, x2 * 4, y2 * 4, x * 4
        cv2.rectangle(img, (x, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if(name!='Buchodu!'):
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
     # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap().release()
cv2.destroyAllWindows()
    