import numpy as np 
import cv2 

cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read()

	if ret==False:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow("video frame", frame)
	cv2.imshow("gray frame", gray)

	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed==ord('q'):
		break

cv2.destroyAllWindows()