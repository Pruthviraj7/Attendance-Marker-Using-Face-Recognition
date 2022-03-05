import cv2
from face_recognition.api import face_encodings
import numpy as np
import face_recognition
import os
from datetime import date, datetime
path="Project\TESTfaces"
images=[]
personName=[]
myList=os.listdir(path)
print(myList)

for curr_img in myList:
    #Adding Path of Images
    current_Img=cv2.imread(f"{path}/{curr_img}")   
    images.append(current_Img) 
    #Selecting Name of Person From Images
    personName.append(os.path.splitext(curr_img)[0]) 
#Printing Person name on terminal
print(personName) 

#Function for Encoding faces
def faceEncodings(images):   
    encodeList=[]
    for img in images:
        #Converting Image to RGB
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
         #Taking Images for encoding
        encode=face_recognition.face_encodings(img)[0]
        #Appending Encoded Image to Empty List
        encodeList.append(encode) 
    return encodeList
#Called face encoding by assigning it to variable encodelistknown
encodeListKnown=faceEncodings(images)
print("All Encodings Complete")

#Function for entering name in csv file
def attendance(name): 
    #Taking csv file in read and append mode
    with open('Attendance.csv','r+') as f: 
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            #Entering names from the images
            entry=line.split(',') 
            nameList.append(entry[0])
        if name not in nameList:
            time_now=datetime.now()
            #Entering entry in Hour:Minute:Second Format
            timeSr=time_now.strftime("%H:%M:%S") 
            #Entering entry in Day:Month:Year Format 
            dateSr=time_now.strftime("%d/%m/%Y") 
            f.writelines(f"\n{name},{timeSr},{dateSr}")

encodeListKnown=faceEncodings(images) 

cap=cv2.VideoCapture(0) #laptop id value is 0
while True:
    #Reading frame of Camera
    ret,frame=cap.read() 
    #Resizing dimensions of Face from camera
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    #Converting Camera image into RGB 
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB) 

    #Detecting faces from the camera
    facesCurrentFrame=face_recognition.face_locations(faces) 
    encodesCurrentFrame=face_recognition.face_encodings(faces,facesCurrentFrame)

    #Checking if the encoded face and the face in image are same or not
    #zip function for doing both task at same time
    for encodeFace, faceLoc in zip(encodesCurrentFrame,facesCurrentFrame): 
        #Comparing Video face & distance with the image provided in system
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace) 
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace) 

        matchIndex=np.argmin(faceDis)
        #Checking if the image and Video image are matching or not
        if matches[matchIndex]:
            #Recognizing name and adding it to csv file in UPPERCASE
            name=personName[matchIndex].upper() 
            y1,x2,y2,x1=faceLoc 
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            #Creating rectangle around the face
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2) 
            #Creating rectangle below rectangle for name section 
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            #Adding text to rectangle frame in during the video image
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 
            attendance(name)
    
    cv2.imshow("Camera",frame)
    #Pressing enter will close all windows
    if cv2.waitKey(10)==13:
        break
cap.release()
cv2.destroyAllWindows()