import cv2
import numpy as np
import Modules.Module_Center_Point as TamAnh

cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    x_center, y_center = TamAnh.ToaDoTam(img)
    ###############  Red color
    low_red = np.array([161, 155, 84])
    hight_red = np.array([179, 255, 255])
    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_frame, low_red, hight_red)
    cv2.imshow("Mask", red_mask)
    ###############
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        x_medium = int(x + w/2)
        y_medium = int(y + h/2)
        #########
        cv2.circle(img, (x_medium, y_medium), 20, (0, 255, 0))
        cv2.line(img, (x_center, y_center), (x_medium, y_medium), (0, 255, 255), 2)
        break

    cv2.imshow("Camera", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()