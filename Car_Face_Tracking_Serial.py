import cv2
import Modules.Module_Face_Detecte as odm
import Modules.Module_Serial_Window as sm

cap = cv2.VideoCapture(0)
ser = sm.initConnection("COM11", 9600)
faceCascade = cv2.CascadeClassifier("Data/haarcascade_frontalface_default.xml")

def findCenter(imgObjects, objects):
    cx, cy = -1, -1
    if len(objects) != 0:
        x, y, w, h = objects[0][0]
        cx = x + w//2
        cy = y + h//2
        cv2.circle(imgObjects, (cx, cy), 2, (0, 255, 0), cv2.FILLED)
        ############## Vẽ điểm trung tâm khuôn mặt
        ih, iw, ic = imgObjects.shape
        cv2.line(imgObjects, (iw//2, cy), (cx, cy), (0, 255, 0), 1)
        cv2.line(imgObjects, (cx, ih//2), (cx, cy), (0, 255, 0), 1)
        ############# Vẽ line từ tâm mặt đến trục X, Y
        data = str(cx-iw//2)
        #sm.SendData(ser, data)
        print(data)
    return cx, cy, imgObjects

while True:
    suscess, img = cap.read()
    img = cv2.resize(img, (0, 0), None, 0.4, 0.4)
    imgObjects, objects = odm.faceDetect(img, faceCascade, 1.08, 10)
    cx, cy, imgObjects = findCenter(imgObjects, objects)
    ###########################
    h, w, c = imgObjects.shape
    cv2.line(imgObjects, (w // 2, 0), (w // 2, h), (255, 0, 255), 1)
    cv2.line(imgObjects, (0, h // 2), (w, h // 2), (255, 0, 255), 1)
    ########################## Vẽ trục X, Y
    img = cv2.resize(imgObjects, (0, 0), None, 3, 3)
    cv2.imshow("Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

