import cv2
from faceMesh import *
import cvzone
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(640, 480, [20, 50], invert=True)

pointListEyeLeft = [33, 7, 163, 144, 145, 153, 154, 155, 173, 157, 158, 159, 160, 161, 246]
pointListEyeRight = [362, 382, 381, 380, 374, 373,390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
pointListFace = [10, 152, 227, 454]
pointList = pointListEyeLeft + pointListEyeRight + pointListFace
ratioLeftList = []
ratioRightList = []
blinkCounter = 0
counter = 0
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)
    if faces:
        face = faces[0]
        for point in pointList:
            cv2.circle(img, face[point], 2, (0, 255, 0), cv2.FILLED)

        #left eye
        leftUp = face[159]
        leftDown = face[145]
        leftRight = face[133]
        leftLeft = face[33]
        lengthHorLeft, _ = detector.findDistance(leftUp, leftDown)
        lengthVerLeft, _ =detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftRight, leftLeft, (0, 0, 255), 3)
        cv2.line(img, leftUp, leftDown, (0, 0, 255), 3)
        ratioLeft = int(100 * (lengthHorLeft / lengthVerLeft))
        ratioLeftList.append(ratioLeft)
        if len(ratioLeftList) > 10:
            ratioLeftList.pop(0)

        #right eye
        rightUp = face[386]
        rightDown = face[374]
        rightRight = face[398]
        rightLeft = face[263]
        lengthHorRight, _ = detector.findDistance(rightUp, rightDown)
        lengthVerRight, _ =detector.findDistance(rightLeft, rightRight)
        cv2.line(img, rightRight, rightLeft, (0, 0, 255), 3)
        cv2.line(img, rightUp, rightDown, (0, 0, 255), 3)
        ratioRight = int(100 * (lengthHorRight / lengthVerRight))
        ratioRightList.append(ratioRight)
        if len(ratioRightList) > 10:
            ratioRightList.pop(0)

        #face 
        topPoint = face[10]
        bottomPoint = face[152]
        leftPoint = face[227]
        rightPoint = face[454]
        lengthHorFace, _ = detector.findDistance(topPoint, bottomPoint)
        lengthVerFace, _ =detector.findDistance(leftPoint, rightPoint)
        print(int(100 * (lengthHorFace / lengthVerFace)))


        ratioAvg = min(sum(ratioLeftList)/len(ratioLeftList), sum(ratioRightList)/len(ratioRightList))

        if ratioAvg < 25 and counter == 0:
            blinkCounter += 1
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0

        cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100), scale=3, thickness=5, offset=20)
        imgPlot = plotY.update(ratioAvg)
        img = cv2.resize(img, (640, 480))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        imgPlot = plotY.update(100)
        img = cv2.resize(img, (640, 480))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)

    cv2.imshow("Image", imgStack)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break