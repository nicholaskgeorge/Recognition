import cv2
import numpy as np
import os
import pygame
import time
from PIL import Image
from random import randint

class Recognition():
    """
    This class works with the rasppi camera inorder to do simple recognision of
    faces or play a certian song when the person is seen."""

    def __init__(self):
        self.threshold = 77
        self.names = [None]+self.getnames()
        self.killthread = False
        self.sounds = [[],['MasterIntro.mp3'],[],['Breakemystride.mp3'],['JennyJenny.mp3'],[]]
    """
    def addsound():
        person = input('\n Please enter the name of the person you would like to add the audio for ==> ')
        #if person in self.names and self.sounds:

        else:
            print('\n That person is not in the data base')
    """

    """
    Names of people are stored in a file. This function adds new faces to the
    list or recognisable faces.
    """
    def addface(self):
        name = input('\n Please enter the name of the new face and press return ==>  ')
        if self.updatenames(name)==True:
            #collecting data
            numsamples = 200
            cam = cv2.VideoCapture(0)
            cam.set(3, 640) # set video width
            cam.set(4, 480) # set video height
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            face_id = len(self.names)-1
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            # Initialize individual sampling face count
            count = 0
            while(True):
                ret, img = cam.read()
                img = cv2.flip(img, -1) # flip video image vertically
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    cv2.imshow('image', img)
                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= numsamples: # Take 30 face sample and stop video
                     break
            # Do a bit of cleanup
            print("\n [INFO] Exiting Program and cleanup stuff")
            cam.release()
            cv2.destroyAllWindows()

            """Training new face"""

            path = 'dataset'
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            #function to get the images and label data
            print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces,ids = self.getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            # Save the model into trainer/trainer.yml
            recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
            # Print the numer of faces trained and end program
            print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
            print(name+"'s face is ready for recognition!")

    def schedcallrecognise(self,people, maxcalls=1):
        print('Scheduled recognition running')
        self.killthread = False
        beencalled = {}
        numcalled = {}
        for i in people:
            beencalled[i]= False
        for i in people:
            numcalled[i]= 0
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        #iniciate id counter
        id = 0
        timeholder = ['None']+[0]*(len(self.names)-1)
        certify=['None']+[0]*(len(self.names)-1)
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        #seen = False
        seen = [0]*(len(self.names)-1)
        present = [0]*(len(self.names)-1)
        for i in range(len(self.names)-1):
            seen.append(False)
        for i in range(len(self.names)-1):
            present.append(False)
        while False in beencalled.values() and not self.killthread:
            ret, img =cam.read()
            img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < self.threshold):
                    present[id] = True
                    certify[id]+=1
                    idnum = id
                    id = self.names[id]
                    confidence = "  {0}%".format(round(100 - confidence))

                    if time.perf_counter()-timeholder[idnum]>10 and certify[idnum]>8:
                        if seen[idnum] == False and id in people and beencalled[id]==False:
                            numcalled[id]+=1
                            if numcalled[id] >= maxcalls:
                                beencalled[id]=True
                            certify[idnum]=0
                            self.playsound(self.sounds[idnum])
                            seen[idnum] = True
                        timeholder[idnum] = time.perf_counter()

                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

            for i in range(1,len(present)):
                if present[i] == False:
                    seen[i] = False
                present[i] = False


            cv2.imshow('camera',img)
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        self.killthread = True
        print('Scheduled Recognition finished')

    def callrecognise(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        #iniciate id counter
        id = 0
        timeholder = [0]*(len(self.names))
        certify= [0]*(len(self.names))
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        #seen = False
        seen = [0]*(len(self.names))
        present = [0]*(len(self.names))
        for i in range(len(self.names)):
            seen.append(False)
        for i in range(len(self.names)):
            present.append(False)
        while True:
            ret, img =cam.read()
            img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence > self.threshold):
                    idnum = 0
                    id = 'Unknown'
                else:
                    idnum = id
                    id = self.names[id]
                present[idnum] = True
                certify[idnum] += 1
                confidence = "  {0}%".format(round(100 - confidence))
                if time.perf_counter()-timeholder[idnum]>10 and certify[idnum]>8:
                    if seen[idnum] == False:
                        certify[idnum]=0
                        choice = randint(0,len(self.sounds[idnum])-1)
                        self.playsound(self.sounds[idnum][choice])
                        seen[idnum] = True
                    timeholder[idnum] = time.perf_counter()

                # else:
                #     id = "unknown"
                #     confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

            for i in range(len(present)):
                if present[i] == False:
                    seen[i] = False
                present[i] = False


            cv2.imshow('camera',img)
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    def recognise(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascadePath);
        font = cv2.FONT_HERSHEY_SIMPLEX
        #iniciate id counter
        id = 0
        # names related to ids: example ==> Marcelo: id=1,  etc
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video widht
        cam.set(4, 480) # set video height
        # Define min window size to be recognized as a face
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        while True:
            ret, img =cam.read()
            img = cv2.flip(img, -1) # Flip vertically
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )
            for(x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                print(id)
                # Check if confidence is less them 100 ==> "0" is perfect match
                if (confidence < self.threshold):
                    id = self.names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

            cv2.imshow('camera',img)
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:
                break
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    def getnames(self):
        file = open('/home/pi/Desktop/Recognition/RecongiseInfo/Names.txt','r')
        raw = file.readlines()
        file.close()
        info = raw[0][raw[0].index(':')+1:].strip()
        if info == '':
            return []
        info = info.split(" ")
        return info
    def updatenames(self,name):
        currentnames = self.getnames()
        if name not in currentnames:
            file = open('/home/pi/Desktop/Recognition/RecongiseInfo/Names.txt','r')
            raw = file.readlines()
            file.close
            new = raw[:]
            new[0]= new[0]+' '+name
            updated = ''.join(new)
            file = open('/home/pi/Desktop/Recognition/RecongiseInfo/Names.txt','w')
            file.write(updated)
            file.close()
            self.names.append(name)
            return True
        else:
            print('This name is already taken please choose a diffrent one')
            return False
    def playsound(self,sound):
        pygame.mixer.init()
        pygame.mixer.music.load(sound)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
           continue
    def getImagesAndLabels(self,path):
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids

if __name__ == '__main__':
    rec = Recognition()
    rec.callrecognise()
