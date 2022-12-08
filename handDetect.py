import cv2
import os
from PIL import Image
from pynput.mouse import Button, Controller
from handUtils import HandDetector
import time
import pyautogui
import numpy as np

mouse = Controller()
press = 1

wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 5

pTime = 0
plocX, plocY = 0, 0  # previous location
clocX, clocY = 0, 0  # current location

camera = cv2.VideoCapture(0)
camera.set(3, wCam)
camera.set(4, hCam)
hand_detector = HandDetector()
wScr, hScr = pyautogui.size()

# Face Recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')


def getImagesWithLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[1].split(".")[0])
        faces = detector.detectMultiScale(imageNp)
        for x, y, w, h in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids


def face_detect(img_detect):
    gray = cv2.cvtColor(img_detect, cv2.COLOR_BGR2GRAY)
    face = detector.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in face:
        cv2.rectangle(img_detect, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
        ids, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 80:
            # cv2.putText(img, "Id: "+str(id), (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            return ids


print("Welcome to virtual gesture control application\n")
print("1. Record face information\n2. Start virtual gesture control.")
choice = input("Enter your choice: ")
choice = int(choice)

while choice == 2 and not os.path.isfile("training/training.xml"):
    print("You haven't record your face information\n")
    choice = input("Enter choice again: ")
    choice = int(choice)

if choice == 1:
    # face information record and training
    faceDetect = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
    ids = input("Enter id number: ")
    num = 0
    while True:
        num += 1
        success, image = camera.read()
        if success:
            image = cv2.flip(image, 1)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                if not os.path.isdir("DataSet"):
                    os.makedirs("Dataset")
                cv2.imwrite("DataSet/" + str(ids) + "." + str(num) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Face_information_capture", image)
            if num >= 25:
                break

    #  face training
    faces, Ids = getImagesWithLabels('Dataset')
    recognizer.train(faces, np.array(Ids))
    if not os.path.isdir("training"):
        os.makedirs("training")
    recognizer.save('training/training.xml')
    print("Successfully recorded")

recognizer.read('training/training.xml')
while choice == 2:
    success, img = camera.read()
    id = face_detect(img)
    if success:
        img = cv2.flip(img, 1)
        hand_detector.process(img)
        position = hand_detector.find_position(img)
        print(position)

        if id:
            #  right hand for mouse control
            #  draw dots on fingertips
            finger_tips = [4, 8, 12, 16, 20]  # code for each fingertip
            if position['Right'] and (not position['Left']):
                rThumb_tip = position['Right']. get(4, None)
                rIndex_tip = position['Right'].get(8, None)
                rMiddle_tip = position['Right'].get(12, None)
                rRing_tip = position['Right'].get(16, None)
                rPinky_tip = position['Right'].get(20, None)
                rTip_position = [rThumb_tip, rIndex_tip, rMiddle_tip, rRing_tip, rPinky_tip]

                for tip in rTip_position:
                    if tip:
                        cv2.circle(img, (tip[0], tip[1]), 5, (0, 0, 255), cv2.FILLED)

                if rThumb_tip and rIndex_tip:
                    thumb_index_middle_x = int((rThumb_tip[0] + rIndex_tip[0])/2)
                    thumb_index_middle_y = int((rThumb_tip[1] + rIndex_tip[1])/2)
                    # draw line between index finger and thumb and show middle dot
                    cv2.line(img, (rIndex_tip[0], rIndex_tip[1]), (rThumb_tip[0], rThumb_tip[1]), (255, 255, 255), 3)
                    cv2.circle(img, (thumb_index_middle_x, thumb_index_middle_y), 5, (255, 0, 0), cv2.FILLED)

                    # frame reduction
                    cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
                    # convert coordinate for cursor moving
                    cursor_x = np.interp(thumb_index_middle_x, (frameR, wCam-frameR), (0, wScr))
                    cursor_y = np.interp(thumb_index_middle_y, (frameR, hCam-frameR), (0, hScr))
                    # Smoothen values
                    clocX = plocX + (cursor_x - plocX) / smoothening
                    clocY = plocY + (cursor_y - plocY) / smoothening

                    # move cursor
                    pyautogui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY

                    if press and ((rThumb_tip[1] - rIndex_tip[1]) <= 10):
                        mouse.press(Button.left)
                        press = 0

                    if (press==0) and ((rThumb_tip[1] - rIndex_tip[1]) >= 40):
                        mouse.release(Button.left)
                        press = 1

                    if press and ((rThumb_tip[1] - rMiddle_tip[1]) <= 10):
                        mouse.click(Button.right, 1)

                    if press and ((rThumb_tip[1] - rRing_tip[1]) <= 10):   # zoom in
                        with pyautogui.hold('ctrl'):
                            mouse.scroll(0, 2)

                    if press and ((rThumb_tip[1] - rPinky_tip[1]) <= 10):  # zoom out
                        with pyautogui.hold('ctrl'):
                            mouse.scroll(0, -2)

            if position['Left'] and (not position['Right']):
                lThumb_tip = position['Left'].get(4, None)
                lIndex_tip = position['Left'].get(8, None)
                lMiddle_tip = position['Left'].get(12, None)
                lRing_tip = position['Left'].get(16, None)
                lPinky_tip = position['Left'].get(20, None)
                lTip_position = [lThumb_tip, lIndex_tip, lMiddle_tip, lRing_tip, lPinky_tip]

                for tip in lTip_position:
                    if tip:
                        cv2.circle(img, (tip[0], tip[1]), 5, (0, 0, 255), cv2.FILLED)

                if lThumb_tip[1] - lIndex_tip[1] <= 10:
                    pyautogui.press('up')

                elif lThumb_tip[1] - lMiddle_tip[1] <= 10:
                    pyautogui.press('down')

                elif lThumb_tip[1] - lRing_tip[1] <= 10:
                    pyautogui.press('left')

                elif lThumb_tip[1] - lPinky_tip[1] <= 10:
                    pyautogui.press('right')

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
