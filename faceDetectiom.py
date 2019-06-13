# 1. Read and show video stream, capture images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image and store in numpy array
# 4. Repeat the above for multile people to generate trainig data


import numpy as np 
import cv2 

#init camera
cap = cv2.VideoCapture(0)

#face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'

file_name = input("enter name ")

while True:

	ret, frame = cap.read()

	if ret==False:
		continue

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	faces = sorted(faces, key = lambda f:f[2]*f[3] )

	#pick last face because it the largest face acc to f[2]*f[3]
	for face in faces[-1:]:
		x, w, y, h = face
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

		#Extrat region of interest
		offset = 10
		face_section = frame[y-offset: y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("video frame", frame)

	key_pressed = cv2.waitKey(1)&0xFF
	if key_pressed==ord('q'):
		break

#Convert face list array into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

#save data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("data sucessfully save at" + dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()